"""Scope resolver module - merged deterministic analysis + ReAct agent refinement.

This module combines deterministic scope identification with a ReAct agent
for intelligent refinement. The agent uses filesystem and search tools to
explore the codebase and make informed scope decisions, replacing the
previous ChainOfThought predictor that relied on a static repo tree.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.context_safe import ContextSafe
from codespy.agents.memory.hippocampus import ContextMemory, Hippocampus
from codespy.agents.reviewer.models import (
    PRContext,
    PackageManifest,
    ReviewContext,
    ScopeResult,
    ScopeType,
)
from codespy.config import get_settings
from codespy.config_memory import get_memory_store
from codespy.tools.git.client import get_client
from codespy.tools.git.models import ChangedFile, MergeRequest, should_review_file
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server
from codespy.agents.reviewer.modules.manifest_parser import (
    extract_package_name,
    extract_dependencies,
    PACKAGE_MANAGER_TO_ECOSYSTEM,
    infer_repo_from_name,
)

logger = logging.getLogger(__name__)

# Files to read at each directory level
SKILL_FILES: list[str] = ["AGENTS.md", "CLAUDE.md", "SKILL.md"]

# Subdirectories to scan for .md files (limited depth)
SKILL_DIRS: list[str] = [".kilo/agent", ".claude", ".agent", ".ai", ".cursor", ".codex"]


def collect_skills(repo_path: Path, subroot: str) -> str | None:
    """Collect hierarchical skills from root down to scope subroot.

    Reads instruction files at each ancestor directory level.
    Returns concatenated content (root-first) or None if nothing found.
    """
    # Build path hierarchy: [".", "packages", "packages/auth"]
    levels: list[str] = ["."]
    if subroot and subroot != ".":
        parts = subroot.split("/")
        for i in range(1, len(parts) + 1):
            levels.append("/".join(parts[:i]))

    sections: list[str] = []

    for level in levels:
        level_path = repo_path if level == "." else repo_path / level

        # Read standalone skill files
        for filename in SKILL_FILES:
            filepath = level_path / filename
            if filepath.is_file():
                content = filepath.read_text(errors="ignore").strip()
                if content:
                    header = filename if level == "." else f"{level}/{filename}"
                    sections.append(f"=== {header} ===\n{content}")

        # Read .md files from skill directories
        for skill_dir in SKILL_DIRS:
            dir_path = level_path / skill_dir
            if dir_path.is_dir():
                for md_file in sorted(dir_path.glob("*.md")):
                    content = md_file.read_text(errors="ignore").strip()
                    if content:
                        rel = f"{level}/{skill_dir}/{md_file.name}" if level != "." else f"{skill_dir}/{md_file.name}"
                        sections.append(f"=== {rel} ===\n{content}")

    return "\n\n".join(sections) if sections else None


def _deepest_common_folder(scopes: list[ScopeResult], repo_slug: str) -> str:
    """Compute the deepest common ancestor directory across all scope subroots.

    Args:
        scopes: List of scope results
        repo_slug: Repository slug for fallback path

    Returns:
        Deepest common ancestor path (e.g., "/repo/scope/subroot/")
    """
    subroots = [s.subroot for s in scopes]
    if not subroots or any(sr in (".", "") for sr in subroots):
        return f"/{repo_slug}/"
    try:
        common = os.path.commonpath(subroots)
    except ValueError:
        common = ""
    if not common or common == ".":
        return f"/{repo_slug}/"
    return f"/{repo_slug}/{common.strip('/')}/"

# Exact filename matches -> package manager
MANIFEST_FILES: dict[str, str] = {
    # Go
    "go.mod": "go",
    # JavaScript/TypeScript
    "package.json": "npm",
    # Python
    "pyproject.toml": "pip",
    "setup.py": "pip",
    "setup.cfg": "pip",
    "Pipfile": "pip",
    # Rust
    "Cargo.toml": "cargo",
    # Java/Kotlin/Scala
    "pom.xml": "maven",
    "build.gradle": "gradle",
    "build.gradle.kts": "gradle",
    "build.sbt": "sbt",
    # PHP
    "composer.json": "composer",
    # Ruby
    "Gemfile": "bundler",
    # .NET / C# / F#
    "Directory.Build.props": "dotnet",
    "Directory.Packages.props": "dotnet",
    # Swift
    "Package.swift": "swift",
    # Dart/Flutter
    "pubspec.yaml": "pub",
    # Elixir
    "mix.exs": "mix",
    # Clojure
    "deps.edn": "clojure",
    "project.clj": "leiningen",
    # Haskell
    "stack.yaml": "stack",
    "cabal.project": "cabal",
    # OCaml
    "dune-project": "dune",
    # Zig
    "build.zig.zon": "zig",
    # Perl
    "cpanfile": "cpan",
    "Makefile.PL": "cpan",
    # R
    "DESCRIPTION": "r",
    # Helm charts
    "Chart.yaml": "helm",
}

# Glob patterns for manifests that use variable filenames
MANIFEST_GLOBS: dict[str, str] = {
    "*.csproj": "dotnet",
    "*.fsproj": "dotnet",
    "*.vbproj": "dotnet",
    "*.sln": "dotnet",
    "*.cabal": "cabal",
}

# Lock file -> manifest mapping
LOCK_TO_MANIFEST: dict[str, str] = {
    "go.sum": "go.mod",
    "package-lock.json": "package.json",
    "yarn.lock": "package.json",
    "pnpm-lock.yaml": "package.json",
    "poetry.lock": "pyproject.toml",
    "uv.lock": "pyproject.toml",
    "Pipfile.lock": "Pipfile",
    "Cargo.lock": "Cargo.toml",
    "Gemfile.lock": "Gemfile",
    "composer.lock": "composer.json",
    "pubspec.lock": "pubspec.yaml",
    "mix.lock": "mix.exs",
    "packages.lock.json": "*.csproj",
    "paket.lock": "paket.dependencies",
}


# Scope indicator directories
SCOPE_INDICATORS: dict[ScopeType, list[str]] = {
    ScopeType.LIBRARY: [
        "lib/", "libs/", "libraries/",
        "pkg/", "packages/",
        "shared/", "common/", "core/",
        "modules/", "mod/",
        "sdk/", "sdks/",
        "components/",
        "plugins/", "extensions/", "addons/",
        "framework/",
    ],
    ScopeType.SERVICE: [
        "services/", "service/", "svc/",
        "microservices/",
        "api/", "apis/",
        "server/", "servers/",
        "backend/", "backends/",
        "gateway/", "gateways/",
        "proxy/", "proxies/",
        "workers/", "worker/",
        "jobs/", "cron/", "schedulers/",
        "consumers/", "producers/",
        "functions/", "lambdas/", "lambda/",
        "cmd/",
    ],
    ScopeType.APPLICATION: [
        "apps/", "app/", "applications/",
        "web/", "www/",
        "frontend/", "frontends/",
        "client/", "clients/",
        "ui/",
        "dashboard/", "admin/",
        "portal/", "console/",
        "mobile/", "native/",
        "ios/", "android/",
        "electron/", "desktop/",
        "site/", "website/",
    ],
    ScopeType.SCRIPT: [
        "scripts/", "script/",
        "bin/",
        "tools/", "tooling/",
        "hack/",
        "make/",
        "ci/", "cd/",
        ".github/", ".gitlab/",
        ".circleci/", ".buildkite/",
        ".azure-pipelines/",
        "infra/", "infrastructure/",
        "terraform/", "tf/",
        "pulumi/",
        "ansible/", "salt/",
        "cloudformation/",
        "docker/",
        "k8s/", "kubernetes/",
        "helm/", "charts/",
        "kustomize/",
        "deploy/", "deployment/", "deployments/",
        "ops/", "devops/", "platform/",
        "provisioning/",
        "config/", "configs/", "configuration/",
    ],
}

# Flattened set for sparse path derivation
SCOPE_INDICATOR_DIRS = frozenset(
    d.rstrip("/") for dirs in SCOPE_INDICATORS.values() for d in dirs
)


class ScopeBoundary(BaseModel):
    """LLM output: a scope boundary decision (no file listing)."""

    subroot: str = Field(description="Path relative to repo root (e.g., 'packages/auth' or '.' for root)")
    scope_type: ScopeType = Field(description="Type of scope")
    reason: str = Field(description="Brief explanation for this boundary")
    description: str = Field(
        default="",
        description="Brief description (max 500 chars) of what this scope/folder contains and its role in the project"
    )


class ScopeRefinementSignature(dspy.Signature):
    """Refine and finalize scope boundaries for a pull request.

    You receive deterministic scope candidates (heuristic proposals), unassigned files,
    and project instructions that describe the repository structure and conventions.
    You have tools to explore the repository filesystem and search code.

    TOOLS AVAILABLE:
    - list_directory: see directory contents
    - get_tree: get subtree structure (use sparingly, targeted)
    - read_file: read manifest files to understand package boundaries
    - search_literal: find patterns across the codebase
    - find_imports_of: understand dependencies between directories

    YOUR ROLE:
    Produce the MINIMAL correct set of scope boundaries. The deterministic heuristics provide
    a starting point — validate, merge, or reclassify as needed.

    Files are assigned automatically to the deepest matching boundary by path prefix.
    You only need to decide WHERE boundaries are, not which files go where.

    REFINEMENT OPERATIONS:
    1. MERGE: Combine candidates that share a deployment/release boundary
    2. RECLASSIFY: Change scope_type if the heuristic got it wrong
    3. CREATE: New boundary only for clearly separate units (especially orphans)
    4. DROP: Remove candidates with no structural value

    WHEN TO USE TOOLS:
    - Use list_directory or get_tree to verify a directory boundary exists
    - Use read_file on manifest files to check if two directories share a package
    - Use find_imports_of to check if directories are coupled (merge signal)
    - Do NOT explore exhaustively — only when a decision requires verification

    CRITICAL RULES:
    1. Output only scope boundaries (subroots). Files are assigned automatically by path
       prefix to the deepest matching scope.
    2. No overlapping scopes (parent contains child)
    3. Manifest-backed candidates (marked 'manifest=...') are authoritative. Keep them
       unless merging multiple manifests into one.
    4. Prefer FEWER scopes. 1-3 scopes is typical for most PRs.
    5. When in doubt, MERGE into fewer scopes rather than split.

    OUTPUT: Final refined scope boundaries. Files are assigned automatically.

    For each scope boundary, include a `description` (max 615 tokens) summarizing
    what the folder contains and its role in the project. For example:
    - "Auth library handling JWT issuance and session management"
    - "API gateway routing to downstream services"
    """

    candidates: str = dspy.InputField(
        desc="Deterministic scope candidates with manifest info and file lists. Subject to refinement."
    )
    orphan_files: list[str] = dspy.InputField(
        desc="Changed files not assigned to any candidate (may be empty list)"
    )
    mr_title: str = dspy.InputField(desc="PR title for intent context")
    mr_description: str = dspy.InputField(desc="PR description for intent context")
    project_instructions: str = dspy.InputField(
        desc="Project coding guidelines and structure context from config files (AGENTS.md, .kilo/, etc.). May be empty."
    )

    scopes: list[ScopeBoundary] = dspy.OutputField(
        desc="Scope boundaries (subroots) with descriptions. Files are assigned automatically — do NOT list files. "
        "For each scope boundary, include a `description` (max 500 characters) summarizing what the folder contains and its role in the project."
    )


def derive_sparse_paths(changed_files: list[str]) -> list[str]:
    """Derive minimal sparse checkout paths from changed files.

    Args:
        changed_files: List of changed file paths

    Returns:
        List of sparse paths for git sparse-checkout
    """
    scope_roots: set[str] = set()

    for filepath in changed_files:
        parts = filepath.split("/")
        if len(parts) <= 1:
            continue  # Root-level file, handled by "/*" below

        # Strategy A: Find scope indicator and take the next directory
        found_indicator = False
        for i, part in enumerate(parts[:-1]):  # Skip filename
            if part.lower() in SCOPE_INDICATOR_DIRS and i + 1 < len(parts) - 1:
                scope_root = "/".join(parts[:i + 2]) + "/"
                scope_roots.add(scope_root)
                found_indicator = True
                break

        # Strategy B: No indicator found -- use depth-2 prefix
        if not found_indicator:
            depth = min(2, len(parts) - 1)
            scope_roots.add("/".join(parts[:depth]) + "/")

    # Always include root-level files for root manifests
    paths = sorted(scope_roots)
    paths.append("/*")

    # Explicitly add root manifest files to ensure they are checked out
    # in sparse/treeless clones (/* pattern doesn't always work reliably)
    for manifest in MANIFEST_FILES:
        paths.append(manifest)
    for manifest_pattern in MANIFEST_GLOBS:
        # For glob patterns like *.csproj, we need to add the pattern itself
        paths.append(manifest_pattern)

    # Agent config directories — project instructions for ReAct agents
    paths.extend([
        ".claude/",
        ".kilo/",
        ".agent/",
        ".ai/",
        ".cursor/",
        ".codex/",
        "AGENTS.md",
        "CLAUDE.md",
        "SKILL.md",
    ])

    return paths


class ScopeResolver(dspy.Module):
    """Deterministic scope resolver with LLM fallback for ambiguous cases."""

    def __init__(self) -> None:
        """Initialize the scope resolver."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    async def _create_tools(self, repo_path: Path) -> tuple[list[Any], list[Any]]:
        """Create tools for the scope agent: filesystem + ripgrep.

        Args:
            repo_path: Path to the repository root (not scope-restricted)

        Returns:
            Tuple of (tools list, contexts list for cleanup)
        """
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        repo_path_str = str(repo_path)
        caller = "scope_resolver"

        tools.extend(await connect_mcp_server(
            tools_dir / "storage" / "filesystem" / "server.py",
            [repo_path_str], contexts, caller,
        ))
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "ripgrep" / "server.py",
            [repo_path_str], contexts, caller,
        ))
        return tools, contexts

    async def _ensure_repo(
        self, mr: MergeRequest, repo_path: Path, is_local: bool
    ) -> None:
        """Clone repo programmatically if not already on disk.

        Args:
            mr: The merge request
            repo_path: Path where repo should be cloned
            is_local: If True, skip cloning (repo already on disk)
        """
        if is_local:
            logger.debug("Local review - skipping clone")
            return

        if repo_path.exists() and (repo_path / ".git").exists():
            from git import Repo
            logger.debug("Updating existing clone at %s", repo_path)
            changed_file_paths = [f.filename for f in mr.changed_files]
            sparse_paths = derive_sparse_paths(changed_file_paths)
            # Update sparse-checkout config
            sparse_file = repo_path / ".git" / "info" / "sparse-checkout"
            sparse_file.parent.mkdir(parents=True, exist_ok=True)
            sparse_file.write_text("\n".join(sparse_paths) + "\n")
            # Fetch and checkout correct ref
            repo = Repo(repo_path)
            repo.git.fetch("origin", mr.head_sha, "--depth", "1")
            repo.git.checkout(mr.head_sha)
            # Ensure manifests at root + parent dirs
            await self._ensure_manifests(repo_path, changed_file_paths)
            return

        changed_file_paths = [f.filename for f in mr.changed_files]
        sparse_paths = derive_sparse_paths(changed_file_paths)
        logger.info("Sparse checkout paths: %s", sparse_paths)

        # Build a dummy URL to get the right client
        if mr.platform == "gitlab":
            gitlab_url = self._settings.gitlab_url.rstrip("/")
            dummy_url = f"{gitlab_url}/{mr.repo_owner}/{mr.repo_name}/-/merge_requests/1"
        else:
            dummy_url = f"https://github.com/{mr.repo_owner}/{mr.repo_name}/pull/1"

        client = get_client(dummy_url, self._settings)
        logger.info(
            "Cloning %s/%s@%s...", mr.repo_owner, mr.repo_name, mr.head_sha[:8]
        )

        client.clone_repository(
            owner=mr.repo_owner,
            repo_name=mr.repo_name,
            ref=mr.head_sha,
            target_path=repo_path,
            depth=1,
            sparse_paths=sparse_paths,
        )
        logger.info("Clone complete: %s", repo_path)

        # Ensure manifest files are present at root and parent directories
        await self._ensure_manifests(repo_path, changed_file_paths)

    async def _ensure_manifests(self, repo_path: Path, changed_files: list[str]) -> None:
        """Ensure manifest files at root and parent directories are checked out.

        Sparse/treeless clones may not materialize manifests at ancestor directories.
        This explicitly checks out known manifest files at:
        - Repository root
        - Every ancestor directory of every changed file path

        Args:
            repo_path: Path to the repository root
            changed_files: List of changed file paths
        """
        from git import Repo

        # Collect all ancestor directories of changed files
        parent_dirs: set[str] = set()
        for filepath in changed_files:
            parts = filepath.split("/")
            for depth in range(1, len(parts)):  # skip filename, collect dirs
                parent_dirs.add("/".join(parts[:depth]))

        # Build list of manifest paths to check
        manifest_paths: list[str] = []

        # Root manifests
        for manifest in MANIFEST_FILES:
            manifest_paths.append(manifest)

        # Parent manifests
        for parent in parent_dirs:
            for manifest in MANIFEST_FILES:
                manifest_paths.append(f"{parent}/{manifest}")

        # Checkout missing manifests
        try:
            from git import Repo
            from git.exc import GitCommandError

            repo = Repo(repo_path)
            for path in manifest_paths:
                if not (repo_path / path).exists():
                    try:
                        repo.git.checkout("HEAD", "--", path)
                        logger.debug("Checked out manifest: %s", path)
                    except GitCommandError:
                        pass  # File doesn't exist in repo — expected
                    except Exception as e:
                        logger.warning("Unexpected error checking out manifest %s: %s", path, e)
        except Exception as e:
            logger.warning("Failed to ensure manifests: %s", e)

    def _resolve(
        self, repo_path: Path, changed_files: list[ChangedFile], repo: str
    ) -> tuple[list[ScopeResult], list[ChangedFile]]:
        """Run deterministic scope resolution.

        Args:
            repo_path: Path to the cloned repository
            changed_files: List of changed files
            repo: Repo identifier

        Returns:
            Tuple of (active scopes, orphan files)
        """
        excluded_dirs = self._settings.excluded_directories
        manifests = self._discover_manifests(repo_path, excluded_dirs)
        logger.info(
            "Manifest discovery at %s found %d manifest(s): %s",
            repo_path,
            len(manifests),
            {str(k): v[1] for k, v in manifests.items()},
        )

        # Build ScopeResult per manifest
        scopes: dict[str, ScopeResult] = {}
        for manifest_dir, (pkg_mgr, manifest_filename) in manifests.items():
            subroot = str(manifest_dir) if str(manifest_dir) != "." else "."
            lock_file = self._find_lock_file(repo_path, manifest_dir, manifest_filename)
            scope_type = self._classify_scope_type(subroot, manifest_filename)
            changed_paths = {f.filename for f in changed_files}
            deps_changed = self._dependencies_changed(manifest_dir, manifest_filename, lock_file, changed_paths)
            manifest_path = str(manifest_dir / manifest_filename) if manifest_dir != Path(".") else manifest_filename

            # Extract package name from manifest
            package_name = extract_package_name(manifest_path, repo_path)

            # Build deterministic description
            description = f"{pkg_mgr} package at {subroot}" if subroot != "." else f"{pkg_mgr} package (root)"

            scopes[subroot] = ScopeResult(
                repo=repo,
                subroot=subroot,
                scope_type=scope_type,
                package_manifest=PackageManifest(
                    manifest_path=manifest_path,
                    lock_file_path=str(lock_file) if lock_file else None,
                    package_manager=pkg_mgr,
                    dependencies_changed=deps_changed,
                    package_name=package_name,
                ),
                reason=f"manifest {manifest_filename} at {subroot}/",
                description=description,
            )

        has_nested_manifests = any(subroot != "." for subroot in scopes)
        root_is_sole_manifest = "." in scopes and not has_nested_manifests

        for file in changed_files:
            covered_by_nested = any(
                subroot != "." and file.filename.startswith(subroot + "/")
                for subroot in scopes
            )
            if covered_by_nested or root_is_sole_manifest:
                continue

            indicator_type, indicator_path = self._find_scope_indicator(file.filename)
            if indicator_path and indicator_type and indicator_path not in scopes:
                # Build deterministic description for indicator-based scope
                description = f"{indicator_type.value} scope at {indicator_path}"
                scopes[indicator_path] = ScopeResult(
                    repo=repo,
                    subroot=indicator_path,
                    scope_type=indicator_type,
                    reason="scope indicator in path (no parent manifest)",
                    description=description,
                )

        # Assign files to deepest matching scope
        orphans = self._assign_files(list(scopes.values()), changed_files)

        # Return only scopes that have files
        active_scopes = [s for s in scopes.values() if s.changed_files]
        return active_scopes, orphans

    def _discover_manifests(
        self, repo_path: Path, excluded_dirs: list[str]
    ) -> dict[Path, tuple[str, str]]:
        """Discover all package manifests in the repo.

        Args:
            repo_path: Path to the repository root
            excluded_dirs: List of directory names to exclude from scanning

        Returns:
            Dict mapping manifest directory -> (package manager, filename)
        """
        logger.debug("Walking %s for manifests (excluded: %s)", repo_path, excluded_dirs)
        manifests: dict[Path, tuple[str, str]] = {}
        excluded_set = set(excluded_dirs)

        for root, dirs, files in os.walk(repo_path):
            # Skip excluded and hidden directories
            dirs[:] = [d for d in dirs if d not in excluded_set and not d.startswith(".")]

            for filename in files:
                # Check exact matches
                if filename in MANIFEST_FILES:
                    manifest_path = Path(root) / filename
                    rel_path = manifest_path.relative_to(repo_path)
                    manifests[rel_path.parent] = (MANIFEST_FILES[filename], filename)
                    continue

                # Check glob patterns
                for pattern, pkg_mgr in MANIFEST_GLOBS.items():
                    if fnmatch.fnmatch(filename, pattern):
                        manifest_path = Path(root) / filename
                        rel_path = manifest_path.relative_to(repo_path)
                        manifests[rel_path.parent] = (pkg_mgr, filename)
                        break

        return manifests

    def _classify_scope_type(self, subroot: str, manifest_filename: str | None) -> ScopeType:
        """Classify scope type from path indicators.

        Args:
            subroot: Scope root path
            manifest_filename: Manifest filename (optional)

        Returns:
            ScopeType classification
        """
        subroot_lower = subroot.lower()

        # Check for script indicators first (most specific)
        for indicator in SCOPE_INDICATORS[ScopeType.SCRIPT]:
            if indicator.rstrip("/") in subroot_lower.split("/"):
                return ScopeType.SCRIPT

        # Check for service indicators
        for indicator in SCOPE_INDICATORS[ScopeType.SERVICE]:
            if indicator.rstrip("/") in subroot_lower.split("/"):
                return ScopeType.SERVICE

        # Check for application indicators
        for indicator in SCOPE_INDICATORS[ScopeType.APPLICATION]:
            if indicator.rstrip("/") in subroot_lower.split("/"):
                return ScopeType.APPLICATION

        # Check for library indicators
        for indicator in SCOPE_INDICATORS[ScopeType.LIBRARY]:
            if indicator.rstrip("/") in subroot_lower.split("/"):
                return ScopeType.LIBRARY

        # Default based on manifest type
        if manifest_filename:
            if manifest_filename == "Chart.yaml":
                return ScopeType.SERVICE  # Helm charts are deployable
            if manifest_filename in ("Dockerfile", "docker-compose.yml"):
                return ScopeType.SCRIPT

        # Fallback: library if it has a manifest, application otherwise
        return ScopeType.LIBRARY if manifest_filename else ScopeType.APPLICATION

    def _find_lock_file(
        self, repo_path: Path, manifest_dir: Path, manifest_filename: str
    ) -> Path | None:
        """Find the lock file corresponding to a manifest.

        Args:
            repo_path: Path to the repository root
            manifest_dir: Directory containing manifest
            manifest_filename: Name of manifest file

        Returns:
            Path to lock file or None
        """
        search_dir = repo_path / manifest_dir
        if not search_dir.exists():
            return None

        for lock_file, manifest_pattern in LOCK_TO_MANIFEST.items():
            if fnmatch.fnmatch(manifest_filename, manifest_pattern):
                lock_path = search_dir / lock_file
                if lock_path.exists():
                    return lock_path.relative_to(repo_path)

        return None

    def _dependencies_changed(
        self,
        manifest_dir: Path,
        manifest_filename: str,
        lock_file: Path | None,
        changed_paths: set[str],
    ) -> bool:
        """Check if manifest or lock file was changed.

        Args:
            manifest_dir: Directory containing manifest
            manifest_filename: Name of manifest file
            lock_file: Path to lock file (optional)
            changed_paths: Set of changed file paths

        Returns:
            True if dependencies changed
        """
        manifest_path = str(manifest_dir / manifest_filename) if manifest_dir != Path(".") else manifest_filename
        if manifest_path in changed_paths:
            return True
        return bool(lock_file and str(lock_file) in changed_paths)

    def _assign_files(
        self, scopes: list[ScopeResult], changed_files: list[ChangedFile]
    ) -> list[ChangedFile]:
        """Assign each changed file to its deepest matching scope.

        Args:
            scopes: List of scope results
            changed_files: Changed files to assign

        Returns:
            List of orphan files that couldn't be assigned
        """
        # Sort by depth (deepest first) for greedy assignment
        sorted_scopes = sorted(scopes, key=lambda s: (-s.subroot.count("/"), s.subroot))
        orphans: list[ChangedFile] = []

        for file in changed_files:
            assigned = False
            for scope in sorted_scopes:
                prefix = scope.subroot + "/" if scope.subroot != "." else ""
                if file.filename.startswith(prefix) or (scope.subroot == "." and "/" not in file.filename):
                    scope.changed_files.append(file)
                    scope.has_changes = True
                    assigned = True
                    break
            if not assigned:
                orphans.append(file)

        return orphans

    def _find_scope_indicator(self, filepath: str) -> tuple[ScopeType | None, str]:
        """Find scope indicator in file path.

        Args:
            filepath: File path to analyze

        Returns:
            Tuple of (scope type, scope root path) or (None, "")
        """
        parts = filepath.split("/")

        for i, part in enumerate(parts[:-1]):  # Skip filename
            part_lower = part.lower()

            # Check each scope type
            for scope_type, indicators in SCOPE_INDICATORS.items():
                for indicator in indicators:
                    indicator_name = indicator.rstrip("/")
                    if part_lower == indicator_name:
                        # Scope root is one level past the indicator
                        if i + 1 < len(parts) - 1:
                            scope_root = "/".join(parts[:i + 2])
                            return scope_type, scope_root

        return None, ""

    def _format_candidate(self, s: ScopeResult) -> str:
        """Format a scope candidate for the LLM.

        Args:
            s: Scope result to format

        Returns:
            Formatted candidate string
        """
        manifest_info = ""
        if s.package_manifest:
            manifest_info = f", manifest={s.package_manifest.manifest_path}"
        files = ", ".join(f.filename for f in s.changed_files)
        return f"- {s.subroot} ({s.scope_type.value}{manifest_info}): files=[{files}]"

    def _apply_boundaries(
        self,
        boundaries: list[ScopeBoundary],
        all_changed_files: list[ChangedFile],
        deterministic_scopes: list[ScopeResult],
        repo: str,
    ) -> list[ScopeResult]:
        """Apply LLM boundaries + manifest guardrail + deterministic file assignment.

        Args:
            boundaries: Scope boundaries from LLM
            all_changed_files: All changed files to assign
            deterministic_scopes: Original deterministic scopes (for manifest info)
            repo: Repo identifier

        Returns:
            List of ScopeResult with files assigned
        """
        # Index manifest-backed scopes from deterministic layer
        manifest_scopes = {
            s.subroot: s for s in deterministic_scopes if s.package_manifest
        }

        # Build final boundary set
        final_boundaries: dict[str, ScopeResult] = {}

        # 1. Always include manifest-backed scopes (immutable baseline)
        for subroot, det_scope in manifest_scopes.items():
            final_boundaries[subroot] = ScopeResult(
                repo=repo,
                subroot=subroot,
                scope_type=det_scope.scope_type,
                package_manifest=det_scope.package_manifest,
                reason=det_scope.reason,
            )

        # 2. Add LLM boundaries — but discard if child of a manifest scope
        for boundary in boundaries:
            # Check if this boundary is inside a manifest-backed scope
            inside_manifest = any(
                boundary.subroot.startswith(ms + "/") or ms == "."
                for ms in manifest_scopes
                if ms != boundary.subroot
            )
            if inside_manifest:
                logger.debug(
                    "Discarding LLM boundary '%s' — inside manifest scope", boundary.subroot
                )
                continue

            # LLM can override scope_type of manifest scopes
            if boundary.subroot in final_boundaries:
                final_boundaries[boundary.subroot].scope_type = boundary.scope_type
            else:
                final_boundaries[boundary.subroot] = ScopeResult(
                    repo=repo,
                    subroot=boundary.subroot,
                    scope_type=boundary.scope_type,
                    reason=boundary.reason,
                )

        # 3. Deterministic file assignment to deepest matching boundary
        orphans = self._assign_files(list(final_boundaries.values()), all_changed_files)

        # 4. Stragglers → root scope
        if orphans:
            if "." in final_boundaries:
                for f in orphans:
                    final_boundaries["."].changed_files.append(f)
                    final_boundaries["."].has_changes = True
            else:
                final_boundaries["."] = ScopeResult(
                    repo=repo, subroot=".", scope_type=ScopeType.APPLICATION,
                    has_changes=True, changed_files=orphans,
                    reason="catch-all for unmatched files",
                )

        return [s for s in final_boundaries.values() if s.changed_files]

    async def _refine_scopes(
        self,
        scopes: list[ScopeResult],
        orphans: list[ChangedFile],
        review_context: ReviewContext,
    ) -> tuple[list[ScopeResult], "ContextMemory | None"]:
        """Use ReAct agent to refine scope assignments from deterministic candidates.

        Args:
            scopes: Already-resolved scope results
            orphans: Orphan files that couldn't be assigned
            review_context: Review context with memory and metadata

        Returns:
            Tuple of (list of ScopeResult with agent-resolved assignments,
                      ContextMemory with topics and items from Hippocampus)
        """
        from codespy.agents.memory.hippocampus import (
            ContextMemory, Topic, compute_common_ancestor_topic_id, make_topic_id,
        )

        # Local bindings from review_context metadata
        mr = review_context.metadata.mr
        repo_path = review_context.metadata.repo_path
        run_id = review_context.metadata.run_id

        # Build candidates string from already-resolved scopes
        candidates_str = "\n".join(self._format_candidate(s) for s in scopes)

        # Read root-level project instructions for the scope agent
        project_instructions = collect_skills(repo_path, ".") or ""

        max_iters = self._settings.get_max_iters("scope")
        tools, contexts = await self._create_tools(repo_path)
        try:
            agent = ContextSafe(
                dspy.ReAct(
                    signature=ScopeRefinementSignature,
                    tools=tools,
                    max_iters=max_iters,
                ),
                ScopeRefinementSignature,
                tools=tools,
                name="scope",
            )
            mem: Hippocampus | None = None

            async with SignatureContext("scope", self._cost_tracker):
                if self._settings.get_memory_enabled("scope"):
                    question = (
                        f"refine scopes of {review_context.pr_context.repo_slug}: "
                        f"PR #{review_context.pr_context.mr_number} "
                        f"{review_context.pr_context.mr_title}: "
                        f"{review_context.pr_context.summary}"
                    )
                    mem = Hippocampus(
                        agent,
                        budget=self._settings.get_memory_budget("scope"),
                        max_reflects=self._settings.get_memory_max_reflects("scope"),
                        question=question,
                        task_name="scope",
                        run_id=run_id,
                        initial_memory=review_context.memory,
                    )
                    result = await mem.aforward(
                        candidates=candidates_str,
                        orphan_files=[f.filename for f in orphans],
                        mr_title=mr.title or "No title",
                        mr_description=mr.body or "No description",
                        project_instructions=project_instructions,
                    )
                else:
                    result = await agent.acall(
                        candidates=candidates_str,
                        orphan_files=[f.filename for f in orphans],
                        mr_title=mr.title or "No title",
                        mr_description=mr.body or "No description",
                        project_instructions=project_instructions,
                    )

            # Collect all changed files (from scopes + orphans)
            all_files = [f for s in scopes for f in s.changed_files] + orphans

            # Apply LLM boundaries with manifest guardrail and deterministic file assignment
            final_scopes = self._apply_boundaries(
                result.scopes, all_files, scopes, mr.repo_slug
            )

            # Copy ScopeBoundary.description to ScopeResult.description (overrides deterministic fallback)
            boundary_descriptions: dict[str, str] = {b.subroot: b.description for b in result.scopes}
            for scope in final_scopes:
                if scope.subroot in boundary_descriptions:
                    scope.description = boundary_descriptions[scope.subroot]

            # Build topic IDs + internal lookup
            internal_packages: dict[str, str] = {}
            scope_topic_ids: dict[str, str] = {}
            for scope in final_scopes:
                pkg_name = scope.package_manifest.package_name if scope.package_manifest else None
                tid = make_topic_id(mr.repo_full_name, scope.subroot, pkg_name)
                scope_topic_ids[scope.subroot] = tid
                if pkg_name:
                    internal_packages[pkg_name] = tid

            # Build Topics with resolved dependencies
            scope_topics: list[Topic] = []
            for scope in final_scopes:
                dep_topic_ids: list[str] = []
                if scope.package_manifest:
                    dep_names, dep_repos = extract_dependencies(
                        scope.package_manifest.manifest_path, repo_path
                    )
                    ecosystem = PACKAGE_MANAGER_TO_ECOSYSTEM.get(
                        scope.package_manifest.package_manager,
                        scope.package_manifest.package_manager,
                    )
                    for name in dep_names:
                        if name in internal_packages:
                            # Rule 1: internal scope match
                            dep_topic_ids.append(internal_packages[name])
                        elif name in dep_repos:
                            # Rule 2: repo identifiable from source metadata
                            dep_topic_ids.append(make_topic_id(dep_repos[name], "", name))
                        elif infer_repo_from_name(name):
                            # Rule 2: repo identifiable from dep name (Go modules)
                            dep_topic_ids.append(make_topic_id(infer_repo_from_name(name), "", name))
                        else:
                            # Rule 3: external
                            dep_topic_ids.append(f"{ecosystem}/{name}")

                scope_topics.append(Topic(
                    id=scope_topic_ids[scope.subroot],
                    description=scope.description,
                    dependencies=dep_topic_ids,
                ))

            # Compute common ancestor topic if >1 scope
            common_ancestor_topic_id = compute_common_ancestor_topic_id(
                mr.repo_full_name, [s.subroot for s in final_scopes]
            )
            if common_ancestor_topic_id:
                # Build description: "Common context for scopes: subroot1, subroot2, ..."
                subroot_list = ", ".join(s.subroot for s in final_scopes)
                common_desc = f"Common context for scopes: {subroot_list}"
                scope_topics.append(Topic(id=common_ancestor_topic_id, description=common_desc))
                stamp_topic_ids = [common_ancestor_topic_id]
            elif scope_topics:
                # Single scope: stamp with its topic ID
                stamp_topic_ids = [scope_topics[0].id]
            else:
                stamp_topic_ids = []

            # Attach hierarchical skills to each produced scope
            for scope in final_scopes:
                scope.skills = collect_skills(repo_path, scope.subroot)

            # Bind topics to hippocampus cmem BEFORE building context_memory.
            # This ensures: (a) persisted episode includes topics, (b) items
            # copied into context_memory are pre-stamped with topic_ids,
            # (c) any new items from consolidation also get topic_ids via _topic_ids.
            if mem is not None and stamp_topic_ids:
                mem._topic_ids = stamp_topic_ids
                mem.cmem.bind_topics(scope_topics, stamp_topic_ids)

            # Build ContextMemory from Hippocampus cmem (items already stamped)
            context_memory = ContextMemory(
                topics=scope_topics,
                context_roadmap=mem.cmem.context_roadmap.copy() if mem else [],
                context_understanding=mem.cmem.context_understanding.copy() if mem else [],
                domain_constants=mem.cmem.domain_constants.copy() if mem else [],
                parsing_schema=mem.cmem.parsing_schema.copy() if mem else [],
                reusable_results=mem.cmem.reusable_results.copy() if mem else [],
            )

            # Persist episode at deepest common folder when memory is enabled
            if mem is not None:
                common_dir = _deepest_common_folder(final_scopes, mr.repo_slug)
                scope_desc = "\n".join(
                    f"- {s.subroot} ({s.scope_type.value}): {len(s.changed_files)} files"
                    for s in final_scopes
                )
                await mem.aend_episode(
                    get_memory_store(self._settings),
                    common_dir,
                    artifacts={"scopes": scope_desc},
                )
                return final_scopes, context_memory

            # No memory enabled: return scopes with ContextMemory containing topics only
            if scope_topics:
                return final_scopes, context_memory
            return final_scopes, None

        finally:
            await cleanup_mcp_contexts(contexts)

    async def aforward(
        self,
        review_context: ReviewContext,
    ) -> tuple[list[ScopeResult], "ContextMemory | None"]:
        """Resolve scopes in the repository for the given MR.

        Args:
            review_context: Review context with inherited memory and metadata

        Returns:
            Tuple of (list of ScopeResult, final context memory or None)
        """
        # Local bindings from review_context metadata
        mr = review_context.metadata.mr
        repo_path = review_context.metadata.repo_path
        is_local = review_context.metadata.is_local
        run_id = review_context.metadata.run_id

        excluded_dirs = self._settings.excluded_directories
        reviewable_files = [f for f in mr.changed_files if should_review_file(f, excluded_dirs)]
        if not reviewable_files:
            return [], review_context.memory
        repo = mr.repo_slug
        if not self._settings.is_signature_enabled("scope"):
            fallback = ScopeResult(
                repo=repo, subroot=".", scope_type=ScopeType.APPLICATION,
                has_changes=True, changed_files=reviewable_files,
                reason="Scope identification disabled",
                description="Repository root",
            )
            fallback.skills = collect_skills(repo_path, ".")
            root_topic = fallback.topic(mr.repo_full_name)
            return [fallback], ContextMemory(topics=[root_topic])

        try:
            await self._ensure_repo(mr, repo_path, is_local)
            scopes, orphans = self._resolve(repo_path, reviewable_files, repo)
            # Log deterministic scopes before LLM refinement
            if scopes:
                det_summary = "\n".join(
                    f"  - {s.subroot} ({s.scope_type.value})"
                    f"{f', manifest={s.package_manifest.manifest_path}' if s.package_manifest else ''}"
                    f": {len(s.changed_files)} files"
                    for s in scopes
                )
                logger.info(
                    "Deterministic scope identification found %d scope(s):\n%s",
                    len(scopes), det_summary
                )
            if orphans:
                logger.info("Deterministic identification produced %d orphan(s)", len(orphans))
            # Load prior scope memory from the deepest common folder
            loaded_memory: ContextMemory | None = None
            if self._settings.get_memory_enabled("scope"):
                from codespy.agents.memory.hippocampus.episode import find_latest_episode
                store = get_memory_store(self._settings)
                common_dir = _deepest_common_folder(scopes, mr.repo_slug) if scopes else f"/{mr.repo_slug}/"
                prior_episode = find_latest_episode(store, common_dir, task="scope", exclude_run_id=run_id)
                # Fallback: prior run may have persisted at repo root if scopes differed
                if prior_episode is None and common_dir != f"/{mr.repo_slug}/":
                    prior_episode = find_latest_episode(
                        store, f"/{mr.repo_slug}/", task="scope", exclude_run_id=run_id
                    )
                if prior_episode is not None:
                    loaded_memory = prior_episode.context_memory
                    # Strip prior-run topics: current run builds authoritative topics
                    # via bind_topics after refinement. Clearing item topic_ids ensures
                    # bind_topics can re-bind them to the current run's stamp_topic_ids.
                    loaded_memory.topics = []
                    for item in loaded_memory.all_items():
                        item.topic_ids = []
                    logger.info(
                        "Loaded prior scope memory (run=%s, items=%d)",
                        prior_episode.run_id[:8], len(loaded_memory.all_items()),
                    )
                else:
                    logger.debug("No prior scope episode found at %s", common_dir)
            # Inject loaded memory into review_context for _refine_scopes
            if loaded_memory is not None:
                merged_memory = (
                    ContextMemory.merge(loaded_memory, review_context.memory)
                    if review_context.memory
                    else loaded_memory
                )
                review_context = ReviewContext(
                    pr_context=review_context.pr_context,
                    memory=merged_memory,
                    metadata=review_context.metadata,
                )
            scopes, context_memory = await self._refine_scopes(
                scopes, orphans, review_context
            )
            # Log final scopes for visibility
            scope_summary = "\n".join(
                f"  - {s.subroot} ({s.scope_type.value}): {len(s.changed_files)} files"
                for s in scopes
            )
            logger.info("Resolved %d scope(s) for %s:\n%s", len(scopes), mr.repo_slug, scope_summary)
            return scopes, context_memory

        except Exception as e:
            logger.error("Scope resolution failed: %s", e, exc_info=True)
            fallback = ScopeResult(
                repo=repo, subroot=".", scope_type=ScopeType.APPLICATION,
                has_changes=True, changed_files=reviewable_files,
                reason=f"Fallback due to error: {e}",
                description="Repository root",
            )
            root_topic = fallback.topic(mr.repo_full_name)
            return [fallback], ContextMemory(topics=[root_topic])

    def forward(
        self,
        review_context: ReviewContext,
    ) -> tuple[list[ScopeResult], ContextMemory | None]:
        """Resolve scopes (sync wrapper)."""
        return asyncio.run(self.aforward(review_context))
