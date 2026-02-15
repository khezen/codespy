"""Helper functions for DSPy review modules."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from codespy.tools.git.models import ChangedFile
from codespy.agents.reviewer.models import Issue

if TYPE_CHECKING:
    from codespy.agents.reviewer.models import ScopeResult

logger = logging.getLogger(__name__)


# Language detection based on file extension
EXTENSION_TO_LANGUAGE = {
    "py": "Python",
    "js": "JavaScript",
    "ts": "TypeScript",
    "jsx": "JavaScript (React)",
    "tsx": "TypeScript (React)",
    "go": "Go",
    "rs": "Rust",
    "java": "Java",
    "kt": "Kotlin",
    "c": "C",
    "cpp": "C++",
    "h": "C/C++ Header",
    "hpp": "C++ Header",
    "cs": "C#",
    "rb": "Ruby",
    "php": "PHP",
    "swift": "Swift",
    "scala": "Scala",
    "sh": "Shell",
    "bash": "Bash",
    "sql": "SQL",
    "vue": "Vue",
    "svelte": "Svelte",
}

# Minimum confidence threshold
MIN_CONFIDENCE = 0.5

# Markdown file extensions to review
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx", ".rst", ".txt"}


def is_markdown_file(filename: str) -> bool:
    """Check if the file is a markdown documentation file."""
    _, ext = os.path.splitext(filename.lower())
    return ext in MARKDOWN_EXTENSIONS


def get_language(file: ChangedFile) -> str:
    """Get the programming language for a file based on extension.
    
    Args:
        file: The changed file
        
    Returns:
        Language name or "Unknown"
    """
    return EXTENSION_TO_LANGUAGE.get(file.extension, "Unknown")


def strip_prefix(path: str, prefix: str) -> str:
    """Strip a directory prefix from a file path.

    Args:
        path: Original file path (e.g., "packages/auth/src/index.ts")
        prefix: Prefix to strip (e.g., "packages/auth")

    Returns:
        Relative path (e.g., "src/index.ts"). Returns path unchanged
        if prefix is "." or path doesn't start with the prefix.
    """
    if prefix == "." or not prefix:
        return path
    # Normalize: ensure prefix ends with "/" for clean stripping
    normalized = prefix.rstrip("/") + "/"
    if path.startswith(normalized):
        return path[len(normalized):]
    # Exact match (file at scope root)
    if path == prefix:
        return os.path.basename(path)
    return path


def resolve_scope_root(repo_path: Path, subroot: str) -> Path:
    """Resolve the absolute path for a scope root directory.

    Args:
        repo_path: Path to the repository root
        subroot: Scope subroot relative to repo root (e.g., "packages/auth" or ".")

    Returns:
        repo_path unchanged if subroot is ".", otherwise repo_path / subroot
    """
    return repo_path if subroot == "." else repo_path / subroot


def make_scope_relative(scope: ScopeResult) -> ScopeResult:
    """Create a copy of a ScopeResult with file paths relative to scope.subroot.

    When MCP tools are rooted at repo_path/scope.subroot, the agent needs file
    paths relative to the scope root (not the repo root). This function creates
    a shallow copy of the scope with adjusted filenames.

    Args:
        scope: Original scope with repo-root-relative file paths

    Returns:
        New ScopeResult with scope-relative file paths in changed_files.
        The subroot is set to "." since paths are now relative to it.
    """
    from codespy.agents.reviewer.models import PackageManifest, ScopeResult as SR

    if scope.subroot == ".":
        # Still need to annotate patches even at repo root
        annotated_files = [
            ChangedFile(
                filename=f.filename,
                status=f.status,
                additions=f.additions,
                deletions=f.deletions,
                patch=f.annotated_patch if f.annotated_patch is not None else f.patch,
                previous_filename=f.previous_filename,
            )
            for f in scope.changed_files
        ]
        return SR(
            subroot=".",
            scope_type=scope.scope_type,
            has_changes=scope.has_changes,
            is_dependency=scope.is_dependency,
            confidence=scope.confidence,
            language=scope.language,
            package_manifest=scope.package_manifest,
            changed_files=annotated_files,
            reason=scope.reason,
        )
    relative_files = [
        ChangedFile(
            filename=strip_prefix(f.filename, scope.subroot),
            status=f.status,
            additions=f.additions,
            deletions=f.deletions,
            patch=f.annotated_patch if f.annotated_patch is not None else f.patch,
            previous_filename=(
                strip_prefix(f.previous_filename, scope.subroot)
                if f.previous_filename
                else None
            ),
        )
        for f in scope.changed_files
    ]
    # Adjust manifest paths too
    manifest = None
    if scope.package_manifest:
        manifest = PackageManifest(
            manifest_path=strip_prefix(scope.package_manifest.manifest_path, scope.subroot),
            lock_file_path=(
                strip_prefix(scope.package_manifest.lock_file_path, scope.subroot)
                if scope.package_manifest.lock_file_path
                else None
            ),
            package_manager=scope.package_manifest.package_manager,
            dependencies_changed=scope.package_manifest.dependencies_changed,
        )
    return SR(
        subroot=".",
        scope_type=scope.scope_type,
        has_changes=scope.has_changes,
        is_dependency=scope.is_dependency,
        confidence=scope.confidence,
        language=scope.language,
        package_manifest=manifest,
        changed_files=relative_files,
        reason=scope.reason,
    )


# Maximum distance (in lines) to snap an invalid line to a valid diff line.
# Beyond this threshold the line reference is dropped entirely.
_MAX_SNAP_DISTANCE = 5


def _nearest_valid_line(line: int, valid_lines: set[int]) -> int | None:
    """Return the nearest valid diff line within ``_MAX_SNAP_DISTANCE``, or None."""
    if not valid_lines:
        return None
    sorted_lines = sorted(valid_lines)
    best: int | None = None
    best_dist = _MAX_SNAP_DISTANCE + 1
    for vl in sorted_lines:
        dist = abs(vl - line)
        if dist < best_dist:
            best = vl
            best_dist = dist
        # Once we've passed the target, remaining lines are farther away
        if vl > line and dist > best_dist:
            break
    return best if best_dist <= _MAX_SNAP_DISTANCE else None


def validate_issue_lines(
    issues: list[Issue],
    changed_files: list[ChangedFile],
) -> None:
    """Validate and fix ``line_start`` / ``line_end`` on each issue (in-place).

    For every issue that carries line numbers, this function checks whether those
    lines actually appear in the diff of the corresponding ``ChangedFile``.  If a
    line is outside the diff it is *snapped* to the nearest valid diff line (within
    ``_MAX_SNAP_DISTANCE``).  If no close match exists the line reference is cleared
    to ``None`` so the comment falls back to the review body instead of landing on
    an unrelated line.

    Args:
        issues: Issues to validate (modified in-place).
        changed_files: The original (non-annotated) changed files whose
            ``valid_new_line_numbers`` represent the diff.
    """
    file_map: dict[str, ChangedFile] = {f.filename: f for f in changed_files}

    for issue in issues:
        cf = file_map.get(issue.filename)
        if cf is None:
            # File not in diff — clear lines
            issue.line_start = None
            issue.line_end = None
            continue

        valid = cf.valid_new_line_numbers
        if not valid:
            issue.line_start = None
            issue.line_end = None
            continue

        # --- line_start ---
        if issue.line_start is not None:
            if issue.line_start not in valid:
                snapped = _nearest_valid_line(issue.line_start, valid)
                if snapped is None:
                    logger.debug(
                        f"Clearing line_start={issue.line_start} for "
                        f"{issue.filename}: no valid diff line nearby"
                    )
                    issue.line_start = None
                    issue.line_end = None
                    continue
                else:
                    logger.debug(
                        f"Snapping line_start {issue.line_start}→{snapped} "
                        f"for {issue.filename}"
                    )
                    issue.line_start = snapped

        # --- line_end ---
        if issue.line_end is not None and issue.line_start is not None:
            if issue.line_end not in valid:
                snapped = _nearest_valid_line(issue.line_end, valid)
                if snapped is None or snapped < issue.line_start:
                    # Collapse to single-line
                    issue.line_end = issue.line_start
                else:
                    logger.debug(
                        f"Snapping line_end {issue.line_end}→{snapped} "
                        f"for {issue.filename}"
                    )
                    issue.line_end = snapped
        elif issue.line_end is not None and issue.line_start is None:
            issue.line_end = None


def restore_repo_paths(issues: list[Issue], subroot: str) -> None:
    """Restore repo-root-relative paths in issue filenames (in-place).

    After the agent reports issues with scope-relative paths, this function
    prepends the scope subroot so filenames are repo-root-relative again.

    Args:
        issues: List of issues to modify in-place
        subroot: Scope subroot to prepend (e.g., "packages/auth")
    """
    if subroot == "." or not subroot:
        return
    prefix = subroot.rstrip("/") + "/"
    for issue in issues:
        if issue.filename and not issue.filename.startswith(prefix):
            issue.filename = prefix + issue.filename
