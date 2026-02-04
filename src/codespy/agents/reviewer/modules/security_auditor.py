"""Security vulnerability analyzer module."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.agents.reviewer.modules.helpers import get_language, is_speculative, MIN_CONFIDENCE
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class CodeSecuritySignature(dspy.Signature):
    """Analyze code changes for VERIFIED security vulnerabilities.

    You are a security expert reviewing code changes.
    You have access to tools that let you explore the codebase to VERIFY your findings.

    CRITICAL RULES:
    - ONLY report vulnerabilities you can VERIFY using the available tools
    - Before reporting any vulnerability, USE the tools to verify your assumptions
    - DO NOT speculate about potential issues you cannot verify
    - If you cannot trace the vulnerability with evidence, do NOT report it
    - Quality over quantity: prefer 0 reports over 1 speculative report

    VERIFICATION WORKFLOW:
    1. Analyze the diff and full content for potential security issues
    2. For each suspected vulnerability, VERIFY using tools:
       - Use read_file to examine input validation, sanitization, or auth code
       - Use find_function_calls to trace data flow from user input to sinks
       - Use find_function_usages to see how sensitive data is handled
       - Use find_callers to trace where dangerous operations are invoked
       - Use search_literal to find related security patterns (escaping, encoding)
    3. Only report issues that are CONFIRMED by your verification

    VULNERABILITIES to look for (with verification) but not limited to:
    - Injection attacks (SQL, command, XSS, etc.): verify input reaches dangerous sink without sanitization
    - Authentication and authorization issues: verify auth check is actually missing by reading the code
    - Sensitive data exposure: verify sensitive data is actually exposed, not just accessed
    - Insecure cryptographic practices: verify weak crypto by checking the actual algorithm used
    - Security misconfigurations: verify misconfiguration by checking actual config values
    - Input validation issues: verify input is not validated by checking validation code
    - Path traversal vulnerabilities: verify path input is not validated/sanitized
    - Race conditions: verify shared state access without synchronization
    - Memory safety issues: verify unsafe memory operations (buffer overflows, use-after-free, etc.)

    DO NOT report:
    - Hypothetical vulnerabilities without evidence
    - Issues that might exist in code you haven't verified
    - "Could be vulnerable if..." scenarios

    For each issue, provide:
    - A clear title
    - Severity (critical, high, medium, low, info)
    - Detailed description of the vulnerability
    - The affected code location
    - A suggested fix
    - CWE ID if applicable
    """

    diff: str = dspy.InputField(
        desc="The code diff showing changes (unified diff format)"
    )
    full_content: str = dspy.InputField(
        desc="The full file content after changes"
    )
    filename: str = dspy.InputField(
        desc="Path to the file being analyzed"
    )
    language: str = dspy.InputField(
        desc="Programming language of the file"
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="VERIFIED security issues found. Empty list if none confirmed."
    )


class DependencySecuritySignature(dspy.Signature):
    """Analyze package dependencies for known security vulnerabilities using OSV database.

    You are a security expert reviewing package dependencies. You have access to:
    - Filesystem tools to read manifest and lock files
    - OSV (Open Source Vulnerabilities) tools to query real vulnerability data

    STEPS:
    1. Use the read_file tool to read the manifest file at manifest_path
    2. If lock_file_path is provided, use read_file to read the lock file as well
    3. Extract dependency names and versions from the files
    4. Use OSV tools to scan dependencies for REAL vulnerabilities:
       - For Python/PyPI: use scan_pypi_package(name, version)
       - For JavaScript/npm: use scan_npm_package(name, version)
       - For Go: use scan_go_package(name, version)
       - For Java/Maven: use scan_maven_package(group_id, artifact_id, version)
       - For Ruby/RubyGems: use scan_rubygems_package(name, version)
       - For Rust/Cargo: use scan_cargo_package(name, version)
       - Or use scan_dependencies(list) for batch scanning multiple packages
    5. Create issues based on the ACTUAL vulnerabilities returned by OSV

    The OSV tools return real CVE/GHSA IDs, severity scores, and fix recommendations.
    Only report vulnerabilities that are actually found by OSV queries.

    For each issue found, provide:
    - A clear title identifying the vulnerable dependency
    - Severity (critical, high, medium, low, info) - use the severity from OSV if available
    - Detailed description including the actual CVE/GHSA ID
    - The affected version and recommended fixed version from OSV
    - CWE ID if available from OSV data
    """

    manifest_path: str = dspy.InputField(
        desc="Path to the manifest file (e.g., package.json, go.mod, pyproject.toml). Use read_file tool to read it."
    )
    lock_file_path: str = dspy.InputField(
        desc="Path to the lock file if available (e.g., package-lock.json, go.sum). Empty string if none. Use read_file tool to read it."
    )
    package_manager: str = dspy.InputField(
        desc="Package manager name (e.g., npm, pip, go, cargo)"
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="Dependency vulnerability issues found. Empty list if none."
    )


class SecurityAuditor(dspy.Module):
    """Analyzes code for security vulnerabilities using DSPy."""

    category = IssueCategory.SECURITY

    def __init__(self) -> None:
        """Initialize the security auditor."""
        super().__init__()

    async def _create_mcp_tools(self, repo_path: Path) -> tuple[list[Any], list[Any]]:
        """Create DSPy tools from filesystem, parser, and OSV MCP servers.

        Args:
            repo_path: Path to the repository root

        Returns:
            Tuple of (tools, contexts) for cleanup
        """
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        repo_path_str = str(repo_path)

        # Add filesystem tools for reading files and exploring structure
        tools.extend(await connect_mcp_server(
            tools_dir / "filesystem" / "server.py",
            [repo_path_str],
            contexts
        ))

        # Add tree-sitter tools for parsing code structure
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "treesitter" / "server.py",
            [repo_path_str],
            contexts
        ))

        # Add ripgrep tools for searching code patterns
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "ripgrep" / "server.py",
            [repo_path_str],
            contexts
        ))

        # Add OSV tools for querying real vulnerability data
        tools.extend(await connect_mcp_server(
            tools_dir / "cyber" / "osv" / "server.py",
            [],
            contexts
        ))

        return tools, contexts

    async def aforward(self, scopes: Sequence[ScopeResult], repo_path: Path) -> list[Issue]:
        """Analyze scopes for security vulnerabilities and return issues.

        Args:
            scopes: The scopes containing changed files to analyze
            repo_path: Path to the cloned repository for reading manifest files

        Returns:
            List of security issues found across all scopes
        """
        all_issues: list[Issue] = []
        tools, contexts = await self._create_mcp_tools(repo_path)

        # Create ReAct agents with code exploration tools
        code_security_agent = dspy.ReAct(
            signature=CodeSecuritySignature,
            tools=tools,
            max_iters=10,
        )
        dependency_security_agent = dspy.ReAct(
            signature=DependencySecuritySignature,
            tools=tools,
            max_iters=5,
        )

        try:
            # 1. Run code security analysis for each changed file
            for scope in scopes:
                for file in scope.changed_files:
                    if not file.patch:
                        logger.debug(f"Skipping {file.filename}: no patch available")
                        continue

                    try:
                        result = await code_security_agent.acall(
                            diff=file.patch or "",
                            full_content=file.content or "",
                            filename=file.filename,
                            language=get_language(file),
                            category=self.category,
                        )
                        issues = [
                            issue for issue in result.issues
                            if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
                        ]
                        all_issues.extend(issues)
                        logger.debug(f"  Code security in {file.filename}: {len(issues)} issues")
                    except Exception as e:
                        logger.error(f"Error analyzing {file.filename}: {e}")

            # 2. Run dependency security analysis for each scope with package manifest
            for scope in scopes:
                if not scope.package_manifest:
                    continue
                manifest = scope.package_manifest
                try:
                    result = await dependency_security_agent.acall(
                        manifest_path=manifest.manifest_path,
                        lock_file_path=manifest.lock_file_path or "",
                        package_manager=manifest.package_manager,
                        category=self.category,
                    )
                    issues = [
                        issue for issue in result.issues
                        if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
                    ]
                    all_issues.extend(issues)
                    logger.debug(f"  Dependency security in {manifest.manifest_path}: {len(issues)} issues")
                except Exception as e:
                    logger.error(f"Error analyzing dependencies in {manifest.manifest_path}: {e}")
        finally:
            await cleanup_mcp_contexts(contexts)
        return all_issues

    def forward(self, scopes: Sequence[ScopeResult], repo_path: Path) -> list[Issue]:
        """Analyze scopes for security vulnerabilities (sync wrapper).

        Args:
            scopes: The scopes containing changed files to analyze
            repo_path: Path to the cloned repository for reading manifest files

        Returns:
            List of security issues found across all scopes
        """
        return asyncio.run(self.aforward(scopes, repo_path))