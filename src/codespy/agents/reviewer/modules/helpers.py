"""Helper functions for DSPy review modules."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from codespy.tools.git.models import ChangedFile
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server
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
        return scope  # Already at repo root, no transformation needed

    relative_files = [
        ChangedFile(
            filename=strip_prefix(f.filename, scope.subroot),
            status=f.status,
            additions=f.additions,
            deletions=f.deletions,
            patch=f.patch,
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


async def create_mcp_tools(scope_root: Path, caller: str) -> tuple[list[Any], list[Any]]:
    """Create DSPy tools from MCP servers, rooted at scope directory.

    Args:
        scope_root: Path to the scope root directory (repo_path / scope.subroot)
        caller: Identifier for the calling module (for logging)

    Returns:
        Tuple of (tools list, contexts list for cleanup)
    """
    tools: list[Any] = []
    contexts: list[Any] = []
    tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
    scope_root_str = str(scope_root)
    # Filesystem tools: read_file, list_directory, get_tree, file_exists, get_file_info
    tools.extend(
        await connect_mcp_server(
            tools_dir / "filesystem" / "server.py", [scope_root_str], contexts, caller
        )
    )
    # Ripgrep tools: search_literal, find_function_usages, find_type_usages, etc.
    tools.extend(
        await connect_mcp_server(
            tools_dir / "parsers" / "ripgrep" / "server.py", [scope_root_str], contexts, caller
        )
    )
    # Treesitter tools: find_function_definitions, find_function_calls, etc.
    tools.extend(
        await connect_mcp_server(
            tools_dir / "parsers" / "treesitter" / "server.py", [scope_root_str], contexts, caller
        )
    )
    return tools, contexts


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




