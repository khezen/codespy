"""Supply chain security auditor module (Dockerfiles and dependencies)."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.agents.reviewer.modules.helpers import MIN_CONFIDENCE, resolve_scope_root, strip_prefix, restore_repo_paths
from codespy.config import get_settings
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class SupplyChainSecuritySignature(dspy.Signature):
    """Analyze supply chain security: Dockerfiles and dependencies.

    You are a busy Principal Engineer reviewing supply chain security. Focus on critical risks only.
    Be extremely terse. Use imperative mood.
    You have access to:
    - Filesystem tools to read files and explore the scope (rooted at scope directory)
    - OSV (Open Source Vulnerabilities) tools to query real vulnerability data

    All file paths are relative to the scope root directory (the current tool root).
    Tools are restricted to this scope — you cannot access files outside it.

    You will analyze TWO types of supply chain security concerns:

    ## 1. DOCKERFILE SECURITY

    FIRST, search for Dockerfiles in the scope using filesystem tools:
    - Use get_tree or list_directory to find Dockerfile, Dockerfile.*, *.Dockerfile, Containerfile
    - Common locations: root (""), docker/, build/, .docker/ directories
    - Once found, use read_file to get the contents

    For each Dockerfile found, check for:
    - Running as root: Missing USER instruction or explicit USER root
    - Secrets in build: Hardcoded passwords, API keys, tokens in ENV or ARG
    - Insecure base images: Using :latest tag, unverified base images
    - Package manager issues: Not pinning versions, not cleaning cache
    - COPY/ADD risks: Copying sensitive files (.env, credentials, private keys)
    - Exposed ports: Unnecessary exposed ports
    - Shell injection: Unquoted variables in RUN commands
    - Privilege escalation: Unnecessary --privileged or capabilities

    OUTPUT RULES:
    - Do not enumerate individual dependencies in reasoning steps. Scan them in batch and only mention those with actual findings.
    - Keep each reasoning step to 1-2 sentences. Never copy source code into issues—use line numbers.
    - Empty list if no verified supply chain issues. No approval text ("LGTM", "looks good").
    - description: ≤25 words, imperative tone, no filler ("Fix X", "Pin Y").
    - No polite or conversational language ("I suggest", "Please consider", "Great").
    - Do not populate code_snippet—use line numbers instead.
    - File paths in issues must be relative to the scope root.

    ## 2. DEPENDENCY SECURITY (package manifests)

    If manifest info is provided:
    1. Use read_file to read the manifest file at manifest_path (relative to scope root)
    2. Extract ALL dependencies with their names and versions
    3. Use OSV tools to scan dependencies for vulnerabilities
    4. Only report vulnerabilities actually found by OSV queries

    ## AVAILABLE OSV TOOLS

    You have access to these OSV tools for querying vulnerability data:

    **Batch Scanning (PREFERRED for multiple packages):**
    - scan_dependencies(dependencies) - Scan multiple packages in a single call
      Example: scan_dependencies([{"name": "requests", "ecosystem": "PyPI", "version": "2.25.0"}, ...])
      Ecosystem values by package manager:
       - Python (pip/poetry/pipenv) → "PyPI"
       - JavaScript/Node.js (npm/yarn/pnpm) → "npm"
       - Go (go mod) → "Go"
       - Java/Maven → "Maven" (name format: "groupId:artifactId")
       - Ruby (bundler) → "RubyGems"
       - Rust (cargo) → "crates.io"

    **Individual Package Scanning:**
    - scan_package(name, ecosystem, version) - Scan any package (generic)
    - scan_pypi_package(name, version) - Scan Python/PyPI packages
    - scan_npm_package(name, version) - Scan npm/Node.js packages
    - scan_go_package(name, version) - Scan Go modules
    - scan_maven_package(group_id, artifact_id, version) - Scan Java/Maven packages
    - scan_rubygems_package(name, version) - Scan Ruby gems
    - scan_cargo_package(name, version) - Scan Rust/Cargo crates

    **Query Tools:**
    - query_package(name, ecosystem, version) - Query vulnerabilities for a specific package
    - query_purl(purl, version) - Query using Package URL (e.g., 'pkg:pypi/requests')
    - query_commit(commit_hash) - Query vulnerabilities affecting a git commit
    - get_vulnerability(osv_id) - Get full details of a vulnerability by ID (e.g., 'GHSA-xxxx' or 'CVE-2021-xxxx')

    ## VERIFICATION RULES
    - Check for actual security issues, not hypotheticals
    - For dependencies, only report CVE/GHSA IDs returned by OSV
    - For Dockerfiles, verify the issue exists by reading the file content

    For each issue, provide:
    - A clear title
    - Severity (critical, high, medium, low, info)
    - Detailed description (include CVE/GHSA IDs for dependencies)
    - The affected location or dependency version
    - A suggested fix (include fixed version for dependencies)
    - CWE ID if applicable
    """

    scope_subroot: str = dspy.InputField(
        desc="Scope root indicator. Always '.' since tools are already rooted at scope directory."
    )
    manifest_path: str = dspy.InputField(
        desc="Path to manifest file relative to scope root (e.g., package.json, go.mod). Empty string if none."
    )
    lock_file_path: str = dspy.InputField(
        desc="Path to lock file relative to scope root (e.g., package-lock.json, go.sum). Empty string if none."
    )
    package_manager: str = dspy.InputField(
        desc="Package manager name (e.g., npm, pip, go). Empty string if no manifest."
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="Verified supply chain issues only. Titles <10 words. Descriptions ≤25 words, imperative. "
        "File paths must be relative to scope root. Empty list if none."
    )


class SupplyChainAuditor(dspy.Module):
    """Analyzes supply chain security (Dockerfiles and dependencies) using DSPy.

    MCP tools are scope-restricted: for each scope, filesystem/parser tools are
    rooted at repo_path/scope.subroot so the agent cannot access files outside
    the scope. OSV tools (no filesystem root) are shared across scopes.
    """

    category = IssueCategory.SECURITY

    def __init__(self) -> None:
        """Initialize the security auditor."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    def _needs_analysis(self, scopes: Sequence[ScopeResult]) -> bool:
        """Check if any scope has supply-chain-relevant changes.

        Returns True if any scope has dependency changes or Dockerfile modifications.
        Avoids spinning up MCP servers when there's nothing to audit.
        """
        return any(self._scope_needs_analysis(s) for s in scopes)

    def _scope_needs_analysis(self, scope: ScopeResult) -> bool:
        """Check if a specific scope has supply-chain-relevant changes.

        Returns True if the scope has dependency changes or Dockerfile modifications.
        """
        scan_unchanged = self._settings.get_scan_unchanged("supply_chain")
        if scope.package_manifest:
            if scan_unchanged or scope.package_manifest.dependencies_changed:
                return True
        for cf in scope.changed_files:
            fname = cf.filename.rsplit("/", 1)[-1].lower()
            if "dockerfile" in fname or fname == "containerfile":
                return True
        return False

    async def _create_scoped_tools(
        self, scope_root: Path
    ) -> tuple[list[Any], list[Any]]:
        """Create scope-restricted DSPy tools from filesystem and parser MCP servers.

        Args:
            scope_root: Path to the scope root directory (repo_path / scope.subroot)

        Returns:
            Tuple of (tools, contexts) for cleanup
        """
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        scope_root_str = str(scope_root)
        caller = "supply_chain_auditor"

        # Add filesystem tools for reading files and exploring structure
        tools.extend(await connect_mcp_server(
            tools_dir / "filesystem" / "server.py",
            [scope_root_str],
            contexts,
            caller,
        ))

        # Add tree-sitter tools for parsing code structure
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "treesitter" / "server.py",
            [scope_root_str],
            contexts,
            caller,
        ))

        # Add ripgrep tools for searching code patterns
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "ripgrep" / "server.py",
            [scope_root_str],
            contexts,
            caller,
        ))

        return tools, contexts

    async def _create_osv_tools(self) -> tuple[list[Any], list[Any]]:
        """Create OSV tools (no filesystem root, can be shared).

        Returns:
            Tuple of (tools, contexts) for cleanup
        """
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        caller = "supply_chain_auditor"

        tools.extend(await connect_mcp_server(
            tools_dir / "cyber" / "osv" / "server.py",
            [],
            contexts,
            caller,
        ))

        return tools, contexts

    async def aforward(self, scopes: Sequence[ScopeResult], repo_path: Path) -> list[Issue]:
        """Analyze scopes for supply chain security vulnerabilities and return issues.

        For each scope, filesystem/parser tools are created rooted at
        repo_path/scope.subroot so the agent can only access files within the
        scope boundary. OSV tools are shared across scopes since they don't
        access the filesystem.

        Args:
            scopes: The scopes containing changed files to analyze
            repo_path: Path to the cloned repository for reading manifest files

        Returns:
            List of security issues found across all scopes
        """
        # Check if supply chain signature is enabled
        if not self._settings.is_signature_enabled("supply_chain"):
            logger.debug("Skipping supply_chain: disabled")
            return []

        # Check if any scope has supply-chain-relevant changes
        if not self._needs_analysis(scopes):
            logger.info("Skipping supply chain analysis: no dependency changes or Dockerfiles modified")
            return []

        all_issues: list[Issue] = []
        supply_chain_max_iters = self._settings.get_max_iters("supply_chain")

        # Create OSV tools once (shared across scopes, no filesystem root)
        osv_tools, osv_contexts = await self._create_osv_tools()

        try:
            for scope in scopes:
                # Skip scopes with no supply-chain-relevant changes
                if not self._scope_needs_analysis(scope):
                    continue

                # Check if we should scan unchanged manifests
                scan_unchanged = self._settings.get_scan_unchanged("supply_chain")

                # Get manifest info (skip if unchanged and scan_unchanged=False)
                manifest = scope.package_manifest
                should_scan_manifest = manifest and (scan_unchanged or manifest.dependencies_changed)

                # Convert manifest paths to scope-relative
                if should_scan_manifest:
                    manifest_path = strip_prefix(manifest.manifest_path, scope.subroot)
                    lock_file_path = (
                        strip_prefix(manifest.lock_file_path, scope.subroot)
                        if manifest.lock_file_path
                        else ""
                    )
                    package_manager = manifest.package_manager
                else:
                    manifest_path = ""
                    lock_file_path = ""
                    package_manager = ""
                    if manifest:
                        logger.debug(f"Skipping unchanged manifest: {manifest.manifest_path}")

                # Scope-restrict filesystem/parser tools to the scope's subroot
                scope_root = resolve_scope_root(repo_path, scope.subroot)
                scoped_tools, scoped_contexts = await self._create_scoped_tools(scope_root)

                try:
                    # Combine scoped filesystem tools with shared OSV tools
                    all_tools = scoped_tools + osv_tools
                    supply_chain_agent = dspy.ReAct(
                        signature=SupplyChainSecuritySignature,
                        tools=all_tools,
                        max_iters=supply_chain_max_iters,
                    )

                    logger.debug(
                        f"Analyzing supply chain in scope {scope.subroot}: "
                        f"manifest={bool(manifest_path)}"
                    )
                    # Track supply_chain signature costs separately
                    async with SignatureContext("supply_chain", self._cost_tracker):
                        result = await supply_chain_agent.acall(
                            scope_subroot=".",
                            manifest_path=manifest_path,
                            lock_file_path=lock_file_path,
                            package_manager=package_manager,
                            category=self.category,
                        )
                    issues = [
                        issue for issue in result.issues
                        if issue.confidence >= MIN_CONFIDENCE
                    ]
                    # Restore repo-root-relative paths in reported issues
                    restore_repo_paths(issues, scope.subroot)
                    all_issues.extend(issues)
                    logger.debug(f"  Supply chain security in scope {scope.subroot}: {len(issues)} issues")
                except Exception as e:
                    logger.error(f"Error analyzing supply chain in scope {scope.subroot}: {e}")
                finally:
                    await cleanup_mcp_contexts(scoped_contexts)
        finally:
            await cleanup_mcp_contexts(osv_contexts)

        logger.info(f"Security audit found {len(all_issues)} issues")
        return all_issues

    def forward(self, scopes: Sequence[ScopeResult], repo_path: Path) -> list[Issue]:
        """Analyze scopes for supply chain security vulnerabilities (sync wrapper).

        Args:
            scopes: The scopes containing changed files to analyze
            repo_path: Path to the cloned repository for reading manifest files

        Returns:
            List of security issues found across all scopes
        """
        return asyncio.run(self.aforward(scopes, repo_path))