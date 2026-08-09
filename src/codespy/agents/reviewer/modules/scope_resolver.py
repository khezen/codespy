"""Scope resolver module - merged deterministic analysis + LLM fallback.

This module combines deterministic scope identification with LLM fallback
for ambiguous cases, replacing the previous split between scope_analyzer
and scope_identifier.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import dspy  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.memory.hippocampus import ContextMap, Hippocampus
from codespy.agents.reviewer.models import (
    PackageManifest,
    ReviewContext,
    ScopeResult,
    ScopeType,
)
from codespy.config import get_settings
from codespy.tools.git.client import get_client
from codespy.tools.git.models import ChangedFile, MergeRequest, should_review_file

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

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
        "middleware/",
        "framework/",
        "utils/", "utilities/", "helpers/",
        "internal/",
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
        "handlers/", "endpoints/",
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


class ScopeAssignment(BaseModel):
    """LLM-friendly scope assignment with string file paths.

    Used for LLM output in the fallback path.
    """

    subroot: str = Field(description="Path relative to repo root (e.g., packages/auth)")
    scope_type: ScopeType = Field(description="Type of scope (library, service, etc.)")
    has_changes: bool = Field(
        default=False, description="Whether this scope has changed files from PR"
    )
    is_dependency: bool = Field(
        default=False, description="Whether this scope depends on a changed scope"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence score for scope identification"
    )
    language: str | None = Field(default=None, description="Primary language detected")
    package_manifest: PackageManifest | None = Field(
        default=None, description="Package manifest info if present"
    )
    changed_files: list[str] = Field(
        default_factory=list, description="Changed file paths belonging to this scope"
    )
    reason: str = Field(description="Explanation for why this scope was identified")


class ScopeClassifierSignature(dspy.Signature):
    """Assign orphan files to the most appropriate scope given pre-computed candidates and repo structure.

    You are given:
    - Pre-identified scope candidates (from deterministic manifest/indicator analysis)
    - Orphan files that could not be assigned deterministically
    - A directory tree of the repository

    SCOPE IDENTIFICATION FROM FILE PATHS:
    Analyze orphan file paths to find the best matching scope:
    1. Extract common directory prefixes from orphan files to find candidate scopes
    2. Look for scope indicator patterns at ANY DEPTH in the path:
       - svc/, services/, microservices/ -> service scope
       - libs/, packages/, shared/, common/, core/ -> library scope
       - apps/, web/, frontend/, mobile/ -> application scope
       - scripts/, bin/, tools/, hack/, ci/, .github/, .gitlab/ -> script scope
    3. Examples of nested scope detection:
       - File: mono/svc/my-service-v1/internal/handler.go
         -> Scope: mono/svc/my-service-v1 ("svc/" indicates service)
       - File: platform/packages/auth/src/index.ts
         -> Scope: platform/packages/auth ("packages/" indicates library)
    4. Group files by longest common directory prefix containing a scope indicator

    SCOPE TYPE CLASSIFICATION:
    These patterns can appear at ANY nesting depth:
    - library: Shared code that others import
      * Patterns: */libs/*, */packages/*, */shared/*, */common/*, */core/*, */sdk/*
    - service: Isolated microservice with APIs
      * Patterns: */services/*, */microservices/*, */svc/*, */cmd/*
    - application: Standalone app or frontend
      * Patterns: */apps/*, */web/*, */frontend/*, */mobile/*
    - script: Build/deployment scripts, tooling, infrastructure
      * Patterns: */scripts/*, */bin/*, */tools/*, */ci/*, */.github/*, */infra/*

    MONO-REPO AWARENESS:
    - Scope indicator directories (packages/, services/, apps/, svc/) can appear at any level
    - Prefer the deepest directory that forms a logical boundary
    - Use repo_tree to verify directory structure exists

    CRITICAL RULES:
    1. EVERY orphan file must be assigned to exactly ONE scope
    2. Don't create overlapping scopes (parent contains child)
    3. Prefer the most specific scope -- deepest directory that forms a logical boundary
    4. Use "." as scope ONLY when files are truly root-level with no nested structure
    5. Assign orphans to existing candidates when paths are compatible (file is under candidate subroot)
    6. Create new scopes only when orphan files clearly belong to an undiscovered boundary

    OUTPUT: Include ALL files (from candidates AND orphans) in the final scope assignments.
    Group files by common directory prefix. Keep reasoning concise.
    """

    candidates: str = dspy.InputField(
        desc="Pre-identified scope candidates with files already assigned (one per line)"
    )
    orphan_files: list[str] = dspy.InputField(
        desc="File paths that could not be assigned to any candidate scope"
    )
    repo_tree: str = dspy.InputField(desc="Directory tree of repo (depth=6)")
    mr_title: str = dspy.InputField(desc="PR title for context")
    mr_description: str = dspy.InputField(desc="PR description for context")

    scopes: list[ScopeAssignment] = dspy.OutputField(
        desc="Final scope assignments including all files from candidates and resolved orphans"
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

    return paths


class ScopeResolver(dspy.Module):
    """Deterministic scope resolver with LLM fallback for ambiguous cases."""

    def __init__(self) -> None:
        """Initialize the scope resolver."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

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
            logger.debug("Repo already cloned at %s", repo_path)
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

        # Build ScopeResult per manifest
        scopes: dict[str, ScopeResult] = {}
        for manifest_dir, (pkg_mgr, manifest_filename) in manifests.items():
            subroot = str(manifest_dir) if str(manifest_dir) != "." else "."
            lock_file = self._find_lock_file(repo_path, manifest_dir, manifest_filename)
            scope_type = self._classify_scope_type(subroot, manifest_filename)
            changed_paths = {f.filename for f in changed_files}
            deps_changed = self._dependencies_changed(manifest_dir, manifest_filename, lock_file, changed_paths)
            manifest_path = str(manifest_dir / manifest_filename) if manifest_dir != Path(".") else manifest_filename

            scopes[subroot] = ScopeResult(
                repo=repo,
                subroot=subroot,
                scope_type=scope_type,
                confidence=0.9,
                package_manifest=PackageManifest(
                    manifest_path=manifest_path,
                    lock_file_path=str(lock_file) if lock_file else None,
                    package_manager=pkg_mgr,
                    dependencies_changed=deps_changed,
                ),
                reason=f"manifest {manifest_filename} at {subroot}/",
            )

        # Add scope-indicator-based scopes for uncovered paths
        for file in changed_files:
            indicator_type, indicator_path = self._find_scope_indicator(file.filename)
            if indicator_path and indicator_type and indicator_path not in scopes:
                scopes[indicator_path] = ScopeResult(
                    repo=repo,
                    subroot=indicator_path,
                    scope_type=indicator_type,
                    confidence=0.9,
                    reason="scope indicator in path",
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
        if lock_file and str(lock_file) in changed_paths:
            return True
        return False

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
        sorted_scopes = sorted(scopes, key=lambda s: s.subroot.count("/"), reverse=True)
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

    def _build_repo_tree(
        self, repo_path: Path, max_depth: int = 6, max_lines: int = 200
    ) -> str:
        """Build a string representation of the repo tree.

        Args:
            repo_path: Path to the repository root
            max_depth: Maximum depth to traverse
            max_lines: Maximum lines to return

        Returns:
            Tree string for LLM context
        """
        lines: list[str] = []
        excluded_dirs = self._settings.excluded_directories

        for root, dirs, files in os.walk(repo_path):
            rel_root = Path(root).relative_to(repo_path)
            depth = len(rel_root.parts) if str(rel_root) != "." else 0

            if depth > max_depth:
                dirs[:] = []
                continue

            dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith(".")]

            indent = "  " * depth
            dir_name = Path(root).name if depth > 0 else "."

            manifest_markers = []
            for f in files:
                if f in MANIFEST_FILES:
                    manifest_markers.append(f)
                for pattern in MANIFEST_GLOBS:
                    if fnmatch.fnmatch(f, pattern):
                        manifest_markers.append(f)

            marker_str = f" ({', '.join(manifest_markers)})" if manifest_markers else ""
            lines.append(f"{indent}{dir_name}/{marker_str}")

        return "\n".join(lines[:max_lines])

    async def _resolve_orphans(
        self,
        scopes: list[ScopeResult],
        orphans: list[ChangedFile],
        repo_tree: str,
        mr: MergeRequest,
        review_context: ReviewContext | None,
        run_id: str | None,
    ) -> list[ScopeResult]:
        """Use LLM to resolve ambiguous scope assignments.

        Args:
            scopes: Already-resolved scope results
            orphans: Orphan files that couldn't be assigned
            repo_tree: Directory tree for context
            mr: The merge request
            review_context: Optional review context with memory
            run_id: Pipeline run identifier

        Returns:
            List of ScopeResult with LLM-resolved assignments
        """
        # Build candidates string from already-resolved scopes
        candidates_str = "\n".join(
            f"- {s.subroot} ({s.scope_type.value}): "
            f"files=[{', '.join(f.filename for f in s.changed_files)}]"
            for s in scopes
        )

        predictor = dspy.Predict(ScopeClassifierSignature)
        mem: Hippocampus | None = None

        async with SignatureContext("scope", self._cost_tracker):
            if self._settings.get_memory_enabled("scope") and review_context:
                question = (
                    f"classify scopes of {review_context.pr_context.repo_slug}: "
                    f"pull request {review_context.pr_context.mr_number} "
                    f"{review_context.pr_context.mr_title}: {review_context.pr_context.summary}"
                )
                mem = Hippocampus(
                    predictor,
                    budget=self._settings.get_memory_budget("scope"),
                    max_reflects=self._settings.get_memory_max_reflects("scope"),
                    question=question,
                    task_name="scope",
                    run_id=run_id,
                    initial_memory=review_context.memory if review_context else None,
                )
                result = await mem.aforward(
                    candidates=candidates_str,
                    orphan_files=[f.filename for f in orphans],
                    repo_tree=repo_tree,
                    mr_title=mr.title or "No title",
                    mr_description=mr.body or "No description",
                )
            else:
                result = await predictor.acall(
                    candidates=candidates_str,
                    orphan_files=[f.filename for f in orphans],
                    repo_tree=repo_tree,
                    mr_title=mr.title or "No title",
                    mr_description=mr.body or "No description",
                )

        # Build file map from all known files
        all_files = {f.filename: f for s in scopes for f in s.changed_files}
        all_files.update({f.filename: f for f in orphans})

        return self._convert_assignments(result.scopes, all_files, mr.repo_slug)

    def _convert_assignments(
        self,
        assignments: list[ScopeAssignment],
        changed_files_map: dict[str, ChangedFile],
        repo: str,
    ) -> list[ScopeResult]:
        """Convert LLM scope assignments to ScopeResults.

        Args:
            assignments: Scope assignments from LLM
            changed_files_map: Map from filename to ChangedFile
            repo: Repo identifier

        Returns:
            List of ScopeResult
        """
        results: list[ScopeResult] = []
        for assignment in assignments:
            changed_files: list[ChangedFile] = []
            for filepath in assignment.changed_files:
                if filepath in changed_files_map:
                    changed_files.append(changed_files_map[filepath])
                else:
                    logger.warning(
                        "File '%s' from scope assignment not found in PR", filepath
                    )
            results.append(
                ScopeResult(
                    repo=repo,
                    subroot=assignment.subroot,
                    scope_type=assignment.scope_type,
                    has_changes=assignment.has_changes,
                    is_dependency=assignment.is_dependency,
                    confidence=assignment.confidence,
                    language=assignment.language,
                    package_manifest=assignment.package_manifest,
                    changed_files=changed_files,
                    reason=assignment.reason,
                )
            )
        return results

    async def aforward(
        self,
        mr: MergeRequest,
        repo_path: Path,
        is_local: bool = False,
        run_id: str | None = None,
        review_context: ReviewContext | None = None,
    ) -> tuple[list[ScopeResult], ContextMap | None]:
        """Resolve scopes in the repository for the given MR.

        Args:
            mr: The merge request to analyze
            repo_path: Path to the repository root
            is_local: If True, repo is already on disk
            run_id: Pipeline run identifier
            review_context: Review context with inherited memory

        Returns:
            Tuple of (list of ScopeResult, final context map or None)
        """
        excluded_dirs = self._settings.excluded_directories
        reviewable_files = [f for f in mr.changed_files if should_review_file(f, excluded_dirs)]

        if not reviewable_files:
            return [], review_context.memory if review_context else None

        repo = mr.repo_slug

        if not self._settings.is_signature_enabled("scope"):
            return [ScopeResult(
                repo=repo, subroot=".", scope_type=ScopeType.APPLICATION,
                has_changes=True, confidence=0.5, changed_files=reviewable_files,
                reason="Scope identification disabled",
            )], review_context.memory if review_context else None

        try:
            await self._ensure_repo(mr, repo_path, is_local)
            scopes, orphans = self._resolve(repo_path, reviewable_files, repo)

            if not orphans:
                return scopes, review_context.memory if review_context else None

            # LLM fallback for orphans
            repo_tree = self._build_repo_tree(repo_path)
            scopes = await self._resolve_orphans(scopes, orphans, repo_tree, mr, review_context, run_id)
            return scopes, review_context.memory if review_context else None

        except Exception as e:
            logger.error("Scope resolution failed: %s", e, exc_info=True)
            return [ScopeResult(
                repo=repo, subroot=".", scope_type=ScopeType.APPLICATION,
                has_changes=True, confidence=0.5, changed_files=reviewable_files,
                reason=f"Fallback due to error: {e}",
            )], review_context.memory if review_context else None

    def forward(
        self,
        mr: MergeRequest,
        repo_path: Path,
        is_local: bool = False,
        run_id: str | None = None,
        review_context: ReviewContext | None = None,
    ) -> tuple[list[ScopeResult], ContextMap | None]:
        """Resolve scopes (sync wrapper)."""
        return asyncio.run(
            self.aforward(
                mr, repo_path, is_local=is_local, run_id=run_id, review_context=review_context
            )
        )
