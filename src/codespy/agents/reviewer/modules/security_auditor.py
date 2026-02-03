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
    """Analyze code changes for security vulnerabilities.

    You are a security expert reviewing code changes. Identify potential security
    vulnerabilities including but not limited to:
    - Injection attacks (SQL, command, XSS, etc.)
    - Authentication and authorization issues
    - Sensitive data exposure
    - Insecure cryptographic practices
    - Security misconfigurations
    - Input validation issues
    - Path traversal vulnerabilities
    - Race conditions
    - Memory safety issues

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
        desc="Security issues found. Empty list if none."
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
        """Initialize the security auditor with chain-of-thought reasoning."""
        super().__init__()

    async def _create_mcp_tools(self, repo_path: Path) -> tuple[list[Any], list[Any]]:
        """Create DSPy tools from filesystem and OSV MCP servers.

        Args:
            repo_path: Path to the repository root

        Returns:
            Tuple of (tools, contexts) for cleanup
        """
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        repo_path_str = str(repo_path)
        
        # Add filesystem tools for reading manifest files
        tools.extend(await connect_mcp_server(
            tools_dir / "filesystem" / "server.py", 
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
        # Create agents once outside loops for better performance
        code_security_agent = dspy.ChainOfThought(CodeSecuritySignature)
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
                        result = code_security_agent(
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
