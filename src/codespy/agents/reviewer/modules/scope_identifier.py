"""Scope identifier module for detecting code scopes in repositories."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]

from codespy.agents.reviewer.models import PackageManifest, ScopeResult, ScopeType
from codespy.tools.filesystem.client import FileSystem
from codespy.tools.github.models import ChangedFile, PullRequest
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


# Package manifest mappings
MANIFEST_FILES = {
    "package.json": ("npm", ["package-lock.json", "yarn.lock", "pnpm-lock.yaml"]),
    "pyproject.toml": ("pip", ["poetry.lock", "uv.lock"]),
    "requirements.txt": ("pip", ["requirements.lock"]),
    "go.mod": ("go", ["go.sum"]),
    "Cargo.toml": ("cargo", ["Cargo.lock"]),
    "pom.xml": ("maven", []),
    "build.gradle": ("gradle", ["gradle.lockfile"]),
    "build.gradle.kts": ("gradle", ["gradle.lockfile"]),
    "composer.json": ("composer", ["composer.lock"]),
    "Gemfile": ("bundler", ["Gemfile.lock"]),
}


class ScopeIdentifierSignature(dspy.Signature):
    """Identify code scopes in a repository for a pull request.

    You have tools to explore the repository filesystem and analyze code.
    Your goal is to identify logical code boundaries (scopes) that organize the codebase.

    EXPLORATION STRATEGY:
    1. Start by examining the changed files list to understand what areas are affected
    2. Use get_tree to explore the repository structure (start with depth 3-4)
    3. Look for package manifest files (package.json, go.mod, pyproject.toml, etc.)
    4. Identify logical boundaries based on directory structure and conventions

    SCOPE TYPE CLASSIFICATION:
    - LIBRARY: Shared code that others import. Look for:
      * Directories named: libs/, packages/, shared/, common/, core/
      * Multiple consumers importing from this scope
      * Generic/reusable code patterns

    - SERVICE: Isolated microservice with explicit APIs. Look for:
      * Directories named: services/, microservices/, svc/
      * Own package manifest, often with server/API code
      * HTTP handlers, gRPC definitions, message consumers

    - APPLICATION: Standalone app or frontend. Look for:
      * Directories named: apps/, web/, frontend/, mobile/
      * Entry points (main.go, index.ts, App.tsx)
      * UI components, routing, state management

    - SCRIPT: Build/deployment scripts, tooling. Look for:
      * Directories named: scripts/, bin/, tools/, hack/, ci/, .github/
      * Shell scripts, Makefiles, Dockerfiles
      * CI/CD configuration, deployment scripts

    MONO-REPO PATTERNS:
    - Look for workspaces configuration in root package.json
    - Check for lerna.json, pnpm-workspace.yaml, turbo.json
    - Examine if directories under packages/, services/, apps/ have their own manifests

    IMPORTANT RULES:
    - Each scope should represent a distinct, buildable/deployable unit
    - Don't create overlapping scopes (parent contains child)
    - Prefer the most specific scope that contains the changed files
    - For flat repos without sub-projects, use "." as the single scope
    - Include package manifest info if found (important for security scanning)
    """

    changed_files: str = dspy.InputField(
        desc="JSON array of changed file paths from the PR"
    )
    pr_title: str = dspy.InputField(desc="PR title for additional context")
    pr_description: str = dspy.InputField(desc="PR description for additional context")

    scopes_json: str = dspy.OutputField(
        desc="""JSON array of identified scopes. Each scope should have:
        {
            "subroot": "path/to/scope" or "." for root,
            "scope_type": "library" | "service" | "application" | "script",
            "confidence": 0.0-1.0,
            "language": "detected language or null",
            "package_manifest": {
                "manifest_path": "relative path to manifest",
                "lock_file_path": "relative path to lock file or null",
                "package_manager": "npm" | "pip" | "go" | etc.
            } or null,
            "reason": "Brief explanation of why this scope was identified"
        }
        Return at least one scope. For simple repos, return [{"subroot": ".", ...}]."""
    )


class ScopeIdentifier(dspy.Module):
    """Agentic scope identifier using ReAct pattern with MCP tools.

    This module uses an LLM agent to explore the repository structure
    and identify logical code scopes for focused code review.
    """

    def __init__(self) -> None:
        """Initialize the scope identifier."""
        super().__init__()

    async def _create_mcp_tools(self, repo_path: Path) -> tuple[list[Any], list[Any]]:
        """Create DSPy tools from MCP servers.

        Args:
            repo_path: Path to the repository root.

        Returns:
            Tuple of (dspy_tools, context_managers_to_keep_open)
        """
        tools: list[Any] = []
        contexts: list[Any] = []
        # Get paths to MCP server modules
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        fs_mcp_path = tools_dir / "filesystem" / "mcp.py"
        rg_mcp_path = tools_dir / "parsers" / "ripgrep" / "mcp.py"
        ts_mcp_path = tools_dir / "parsers" / "treesitter" / "mcp.py"
        gh_mcp_path = tools_dir / "github" / "mcp.py"
        # Connect to MCP servers
        repo_path_str = str(repo_path)
        tools.extend(await connect_mcp_server(fs_mcp_path, [repo_path_str], contexts))
        tools.extend(await connect_mcp_server(rg_mcp_path, [repo_path_str], contexts))
        tools.extend(await connect_mcp_server(ts_mcp_path, [repo_path_str], contexts))
        tools.extend(await connect_mcp_server(gh_mcp_path, [], contexts))
        return tools, contexts

    def _parse_scopes_json(self, scopes_json: str) -> list[dict]:
        """Parse scopes JSON from agent output."""
        try:
            # Try to extract JSON from the response
            if "```json" in scopes_json:
                start = scopes_json.find("```json") + 7
                end = scopes_json.find("```", start)
                scopes_json = scopes_json[start:end].strip()
            elif "```" in scopes_json:
                start = scopes_json.find("```") + 3
                end = scopes_json.find("```", start)
                scopes_json = scopes_json[start:end].strip()

            # Find JSON array
            start = scopes_json.find("[")
            end = scopes_json.rfind("]") + 1
            if start >= 0 and end > start:
                scopes_json = scopes_json[start:end]

            return json.loads(scopes_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse scopes JSON: {e}")
            return []

    def _detect_manifest_in_scope(
        self, subroot: str, changed_files: list[str], fs: FileSystem
    ) -> PackageManifest | None:
        """Detect package manifest in a scope directory."""
        for manifest_name, (pkg_manager, lock_files) in MANIFEST_FILES.items():
            manifest_path = f"{subroot}/{manifest_name}" if subroot != "." else manifest_name
            if fs.exists(manifest_path):
                # Check for lock file
                lock_file_path = None
                for lock_name in lock_files:
                    lock_path = f"{subroot}/{lock_name}" if subroot != "." else lock_name
                    if fs.exists(lock_path):
                        lock_file_path = lock_path
                        break

                # Check if dependencies changed
                deps_changed = manifest_path in changed_files
                if lock_file_path:
                    deps_changed = deps_changed or lock_file_path in changed_files

                return PackageManifest(
                    manifest_path=manifest_path,
                    lock_file_path=lock_file_path,
                    package_manager=pkg_manager,
                    dependencies_changed=deps_changed,
                )

        return None

    def _assign_files_to_scopes(
        self, scopes: list[ScopeResult], changed_files: list[ChangedFile]
    ) -> list[ChangedFile]:
        """Assign changed files to their best-matching scopes.

        Args:
            scopes: List of identified scopes (will be modified in place)
            changed_files: List of changed files from the PR

        Returns:
            List of orphan files that couldn't be assigned to any scope
        """
        assigned_files: set[str] = set()

        # Sort scopes by subroot length (longest first) for most specific matching
        sorted_scopes = sorted(scopes, key=lambda s: len(s.subroot), reverse=True)

        for changed_file in changed_files:
            filepath = changed_file.filename

            # Find the most specific scope that contains this file
            for scope in sorted_scopes:
                if scope.subroot == ".":
                    continue  # Handle root scope last

                # Check if file belongs to this scope
                scope_prefix = scope.subroot.rstrip("/") + "/"
                if filepath.startswith(scope_prefix) or filepath == scope.subroot:
                    scope.changed_files.append(changed_file)
                    scope.has_changes = True
                    assigned_files.add(filepath)
                    break

        # Collect orphan files
        orphan_files = [f for f in changed_files if f.filename not in assigned_files]

        # Assign orphans to root scope if it exists
        root_scope = next((s for s in scopes if s.subroot == "."), None)
        if root_scope and orphan_files:
            root_scope.changed_files.extend(orphan_files)
            root_scope.has_changes = True
            orphan_files = []

        return orphan_files

    async def aforward(self, pr: PullRequest, repo_path: Path) -> list[ScopeResult]:
        """Identify scopes in the repository for the given PR (async version).

        Args:
            pr: Pull request with changed files
            repo_path: Path to the cloned repository

        Returns:
            List of ScopeResult objects with changed files assigned
        """
        # Create MCP tools
        tools, contexts = await self._create_mcp_tools(repo_path)

        try:
            # Initialize ReAct agent with MCP tools
            agent = dspy.ReAct(
                signature=ScopeIdentifierSignature,
                tools=tools,
                max_iters=15,
            )

            # Prepare input for the agent
            changed_file_paths = [f.filename for f in pr.changed_files]
            changed_files_json = json.dumps(changed_file_paths, indent=2)

            logger.info(f"Identifying scopes for {len(changed_file_paths)} changed files...")

            try:
                # Run the ReAct agent asynchronously
                result = await agent.acall(
                    changed_files=changed_files_json,
                    pr_title=pr.title or "No title",
                    pr_description=pr.body or "No description",
                )

                # Parse agent output
                scope_dicts = self._parse_scopes_json(result.scopes_json)

            except Exception as e:
                logger.error(f"Agent failed: {e}")
                # Fallback: return single root scope
                scope_dicts = [
                    {
                        "subroot": ".",
                        "scope_type": "application",
                        "confidence": 0.5,
                        "language": None,
                        "reason": f"Fallback scope due to agent error: {e}",
                    }
                ]

        finally:
            # Clean up MCP connections
            await cleanup_mcp_contexts(contexts)

        # Create FileSystem for manifest detection
        fs = FileSystem(repo_path)

        # Convert to ScopeResult objects
        scopes: list[ScopeResult] = []
        for sd in scope_dicts:
            try:
                scope_type = ScopeType(sd.get("scope_type", "application"))
            except ValueError:
                scope_type = ScopeType.APPLICATION

            # Detect package manifest
            manifest = None
            if "package_manifest" in sd and sd["package_manifest"]:
                pm = sd["package_manifest"]
                manifest = PackageManifest(
                    manifest_path=pm.get("manifest_path", ""),
                    lock_file_path=pm.get("lock_file_path"),
                    package_manager=pm.get("package_manager", "unknown"),
                    dependencies_changed=False,  # Will be updated below
                )
            else:
                # Try to auto-detect manifest
                manifest = self._detect_manifest_in_scope(
                    sd.get("subroot", "."), changed_file_paths, fs
                )

            scope = ScopeResult(
                subroot=sd.get("subroot", "."),
                scope_type=scope_type,
                has_changes=False,  # Will be updated when files are assigned
                is_dependency=False,
                confidence=sd.get("confidence", 0.8),
                language=sd.get("language"),
                package_manifest=manifest,
                changed_files=[],
                reason=sd.get("reason", "Identified by agent"),
            )
            scopes.append(scope)

        # Ensure we have at least a root scope
        if not scopes:
            scopes.append(
                ScopeResult(
                    subroot=".",
                    scope_type=ScopeType.APPLICATION,
                    has_changes=True,
                    is_dependency=False,
                    confidence=0.5,
                    language=None,
                    package_manifest=self._detect_manifest_in_scope(".", changed_file_paths, fs),
                    changed_files=[],
                    reason="Default root scope",
                )
            )

        # Assign changed files to scopes
        orphan_files = self._assign_files_to_scopes(scopes, pr.changed_files)

        # Handle any remaining orphan files with a new root scope
        if orphan_files:
            root_scope = ScopeResult(
                subroot=".",
                scope_type=ScopeType.APPLICATION,
                has_changes=True,
                is_dependency=False,
                confidence=0.5,
                language=None,
                package_manifest=self._detect_manifest_in_scope(".", changed_file_paths, fs),
                changed_files=orphan_files,
                reason="Fallback scope for files not matching identified sub-projects",
            )
            scopes.append(root_scope)

        # Remove scopes with no changes (unless they're dependency scopes for libraries)
        scopes = [
            s
            for s in scopes
            if s.has_changes or (s.is_dependency and s.scope_type == ScopeType.LIBRARY)
        ]

        # Validate: all PR files must be covered
        total_assigned = sum(len(s.changed_files) for s in scopes)
        if total_assigned != len(pr.changed_files):
            logger.error(
                f"File assignment mismatch: {total_assigned} assigned vs {len(pr.changed_files)} in PR"
            )
            # Log which files are missing
            assigned_set = {f.filename for s in scopes for f in s.changed_files}
            missing = [f.filename for f in pr.changed_files if f.filename not in assigned_set]
            logger.error(f"Missing files: {missing}")

        logger.info(f"Identified {len(scopes)} scopes covering {total_assigned} files")

        return scopes

    def forward(self, pr: PullRequest, repo_path: Path) -> list[ScopeResult]:
        """Identify scopes in the repository for the given PR (sync wrapper).

        Args:
            pr: Pull request with changed files
            repo_path: Path to the cloned repository

        Returns:
            List of ScopeResult objects with changed files assigned
        """
        return asyncio.run(self.aforward(pr, repo_path))