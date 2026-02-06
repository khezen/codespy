"""Security vulnerability analyzer module."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.agents.reviewer.modules.helpers import is_speculative, MIN_CONFIDENCE
from codespy.config import get_settings
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class CodeSecuritySignature(dspy.Signature):
    """Analyze code changes for VERIFIED security vulnerabilities.

    You are a security expert reviewing code changes.
    You have access to tools that let you explore the codebase to VERIFY your findings.

    INPUT:
    - scope: A ScopeResult containing:
      * subroot: Path relative to repo root (e.g., "packages/auth")
      * scope_type: Type of scope (library, service, application, script)
      * language: Primary programming language
      * changed_files: List of ChangedFile objects, each with:
        - filename: Path to the file
        - patch: The diff showing exactly what changed (additions/deletions)

    TOKEN EFFICIENCY:
    - The patch shows exactly what changed - analyze it FIRST before using tools
    - Use read_file ONLY when you need context outside the diff (e.g., checking if validation exists elsewhere)
    - Prefer targeted searches (find_function_calls, search_literal) over reading entire files
    - Stop exploring once you have enough evidence to confirm or dismiss an issue

    CRITICAL RULES:
    - Analyze ALL changed files in the scope
    - ONLY report vulnerabilities you can VERIFY using the available tools
    - Before reporting any vulnerability, USE the tools to verify your assumptions
    - DO NOT speculate about potential issues you cannot verify
    - If you cannot trace the vulnerability with evidence, do NOT report it
    - Quality over quantity: prefer 0 reports over 1 speculative report

    VERIFICATION WORKFLOW:
    1. Review each changed file's patch in scope.changed_files - the diff shows what changed
    2. For suspected vulnerabilities that need verification beyond the patch:
       - Use find_function_calls to trace data flow from user input to sinks
       - Use find_function_usages to see how sensitive data is handled
       - Use search_literal to find related security patterns (escaping, encoding)
       - Use read_file ONLY if you need broader context not visible in the patch
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

    scope: ScopeResult = dspy.InputField(
        desc="Full scope context including all changed files, scope type, subroot, and language"
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="VERIFIED security issues found across all changed files in scope. Empty list if none confirmed."
    )


class SupplyChainSecuritySignature(dspy.Signature):
    """Analyze supply chain security: artifacts (Dockerfiles, etc.) and dependencies.

    You are a security expert reviewing supply chain security aspects of a project.
    You have access to:
    - Filesystem tools to read files and explore the codebase
    - OSV (Open Source Vulnerabilities) tools to query real vulnerability data

    You will analyze TWO types of supply chain security concerns:

    ## 1. ARTIFACT SECURITY (Dockerfiles, CI configs, etc.)

    For each artifact provided, check for:
    - Running as root: Missing USER instruction or explicit USER root
    - Secrets in build: Hardcoded passwords, API keys, tokens in ENV or ARG
    - Insecure base images: Using :latest tag, unverified base images
    - Package manager issues: Not pinning versions, not cleaning cache
    - COPY/ADD risks: Copying sensitive files (.env, credentials, private keys)
    - Exposed ports: Unnecessary exposed ports
    - Shell injection: Unquoted variables in RUN commands
    - Privilege escalation: Unnecessary --privileged or capabilities

    ## 2. DEPENDENCY SECURITY (package manifests)

    If manifest info is provided:
    1. Use read_file to read the manifest file at manifest_path
    2. If lock_file_path is provided, read it as well
    3. Extract ALL dependencies with their names and versions

    4. **CRITICAL: Use BATCH scanning to scan ALL dependencies in a SINGLE call**
       ALWAYS use scan_dependencies() instead of individual scan calls:

       scan_dependencies([
           {"name": "requests", "ecosystem": "PyPI", "version": "2.25.0"},
           {"name": "django", "ecosystem": "PyPI", "version": "3.1.0"},
           ...all other dependencies...
       ])

       Ecosystem values by package manager:
       - Python (pip/poetry/pipenv) → "PyPI"
       - JavaScript/Node.js (npm/yarn/pnpm) → "npm"
       - Go (go mod) → "Go"
       - Java/Maven → "Maven" (name format: "groupId:artifactId")
       - Ruby (bundler) → "RubyGems"
       - Rust (cargo) → "crates.io"

       DO NOT call individual scan tools (scan_pypi_package, scan_npm_package, etc.)
       for multiple packages - this wastes iterations. Use scan_dependencies() for ALL.

    5. Only report vulnerabilities actually found by OSV queries

    ## VERIFICATION RULES
    - Check for actual security issues, not hypotheticals
    - For dependencies, only report CVE/GHSA IDs returned by OSV
    - For artifacts, verify the issue exists in the content provided

    For each issue, provide:
    - A clear title
    - Severity (critical, high, medium, low, info)
    - Detailed description (include CVE/GHSA IDs for dependencies)
    - The affected location or dependency version
    - A suggested fix (include fixed version for dependencies)
    - CWE ID if applicable
    """

    artifacts: list[dict] = dspy.InputField(
        desc="List of artifact dicts with keys: path, artifact_type, content. Empty list if no artifacts."
    )
    manifest_path: str = dspy.InputField(
        desc="Path to manifest file (e.g., package.json, go.mod). Empty string if none."
    )
    lock_file_path: str = dspy.InputField(
        desc="Path to lock file (e.g., package-lock.json, go.sum). Empty string if none."
    )
    package_manager: str = dspy.InputField(
        desc="Package manager name (e.g., npm, pip, go). Empty string if no manifest."
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="VERIFIED supply chain security issues. Empty list if none confirmed."
    )


class SecurityAuditor(dspy.Module):
    """Analyzes code for security vulnerabilities using DSPy."""

    category = IssueCategory.SECURITY

    def __init__(self) -> None:
        """Initialize the security auditor."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

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
        caller = "security_auditor"

        # Add filesystem tools for reading files and exploring structure
        tools.extend(await connect_mcp_server(
            tools_dir / "filesystem" / "server.py",
            [repo_path_str],
            contexts,
            caller,
        ))

        # Add tree-sitter tools for parsing code structure
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "treesitter" / "server.py",
            [repo_path_str],
            contexts,
            caller,
        ))

        # Add ripgrep tools for searching code patterns
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "ripgrep" / "server.py",
            [repo_path_str],
            contexts,
            caller,
        ))

        # Add OSV tools for querying real vulnerability data
        tools.extend(await connect_mcp_server(
            tools_dir / "cyber" / "osv" / "server.py",
            [],
            contexts,
            caller,
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

        # Check which signatures are enabled
        code_security_enabled = self._settings.is_signature_enabled("code_security")
        supply_chain_enabled = self._settings.is_signature_enabled("supply_chain")

        # Create ReAct agents with per-signature config
        code_security_agent = None
        supply_chain_agent = None

        if code_security_enabled:
            code_security_max_iters = self._settings.get_max_iters("code_security")
            code_security_agent = dspy.ReAct(
                signature=CodeSecuritySignature,
                tools=tools,
                max_iters=code_security_max_iters,
            )

        if supply_chain_enabled:
            supply_chain_max_iters = self._settings.get_max_iters("supply_chain")
            supply_chain_agent = dspy.ReAct(
                signature=SupplyChainSecuritySignature,
                tools=tools,
                max_iters=supply_chain_max_iters,
            )

        try:
            for scope in scopes:
                # 1. Run code security analysis if enabled and there are changed files
                if code_security_agent and scope.changed_files:
                    try:
                        logger.debug(f"Analyzing scope {scope.subroot} with {len(scope.changed_files)} files")
                        # Track code_security signature costs separately
                        async with SignatureContext("code_security", self._cost_tracker):
                            result = await code_security_agent.acall(
                                scope=scope,
                                category=self.category,
                            )
                        issues = [
                            issue for issue in result.issues
                            if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
                        ]
                        all_issues.extend(issues)
                        logger.debug(f"  Code security in scope {scope.subroot}: {len(issues)} issues")
                    except Exception as e:
                        logger.error(f"Error analyzing scope {scope.subroot}: {e}")

                # 2. Run supply chain security analysis if enabled (artifacts + dependencies)
                if supply_chain_agent:
                    # Check if we should scan unchanged artifacts/manifests
                    scan_unchanged = self._settings.get_scan_unchanged("supply_chain")

                    # Collect artifacts with their content (filter by has_changes if scan_unchanged=False)
                    artifacts_data: list[dict] = []
                    for artifact in scope.artifacts:
                        # Skip unchanged artifacts if scan_unchanged is False
                        if not scan_unchanged and not artifact.has_changes:
                            logger.debug(f"Skipping unchanged artifact: {artifact.path}")
                            continue
                        artifact_full_path = repo_path / artifact.path
                        if not artifact_full_path.exists():
                            logger.warning(f"Artifact file not found, skipping: {artifact.path}")
                            continue
                        artifacts_data.append({
                            "path": artifact.path,
                            "artifact_type": artifact.artifact_type,
                            "content": artifact_full_path.read_text(),
                        })
                    # Get manifest info (skip if unchanged and scan_unchanged=False)
                    manifest = scope.package_manifest
                    should_scan_manifest = manifest and (scan_unchanged or manifest.dependencies_changed)
                    manifest_path = manifest.manifest_path if should_scan_manifest else ""
                    lock_file_path = (manifest.lock_file_path or "") if should_scan_manifest else ""
                    package_manager = manifest.package_manager if should_scan_manifest else ""
                    if manifest and not should_scan_manifest:
                        logger.debug(f"Skipping unchanged manifest: {manifest.manifest_path}")
                    # Run supply chain analysis if there's anything to analyze
                    if artifacts_data or manifest_path:
                        try:
                            logger.debug(
                                f"Analyzing supply chain in scope {scope.subroot}: "
                                f"{len(artifacts_data)} artifacts, manifest={bool(manifest_path)}"
                            )
                            # Track supply_chain signature costs separately
                            async with SignatureContext("supply_chain", self._cost_tracker):
                                result = await supply_chain_agent.acall(
                                    artifacts=artifacts_data,
                                    manifest_path=manifest_path,
                                    lock_file_path=lock_file_path,
                                    package_manager=package_manager,
                                    category=self.category,
                                )
                            issues = [
                                issue for issue in result.issues
                                if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
                            ]
                            all_issues.extend(issues)
                            logger.debug(f"  Supply chain security in scope {scope.subroot}: {len(issues)} issues")
                        except Exception as e:
                            logger.error(f"Error analyzing supply chain in scope {scope.subroot}: {e}")
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