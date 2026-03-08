"""Deterministic agentic context detector — finds AI agent prompts, instructions, and configs."""

import logging
from pathlib import Path

from codespy.tools.filesystem.client import FileSystem
from codespy.tools.filesystem.models import EntryType, TreeNode

logger = logging.getLogger(__name__)

# Single files at any depth that indicate agentic contexts (case-insensitive match).
_AGENTIC_SINGLE_FILES: set[str] = {
    "claude.md",
    "prompt.txt",
    "system_prompt.txt",
    "instructions.md",
    "babyagi.md",
    "agent_prompt.md",
    "agent_instructions.md",
    "task.md",
    "memory.md",
    "constraints.md",
    "ai_settings.json",
    "agent_config.yaml",
}

# Folder-based patterns: directory name → set of allowed extensions.
_AGENTIC_FOLDER_PATTERNS: dict[str, set[str]] = {
    "prompts": {".md"},
    "instructions": {".md"},
    "tools": {".md"},
    ".clinerules": {".md"},
    ".rules": {".md"},
    "config": {".json", ".yaml"},
}


def _matches_single_file(name: str) -> bool:
    """Check if a filename matches a known agentic single-file pattern."""
    return name.lower() in _AGENTIC_SINGLE_FILES


def _is_agentic_folder(dir_name: str) -> set[str] | None:
    """Return allowed extensions if dir_name is a known agentic folder, else None."""
    return _AGENTIC_FOLDER_PATTERNS.get(dir_name.lower())


def _collect_folder_files(node: TreeNode, prefix: str, allowed_exts: set[str]) -> list[str]:
    """Recursively collect files from an agentic folder that match allowed extensions."""
    paths: list[str] = []
    for child in node.children:
        if child.entry_type == EntryType.FILE:
            suffix = Path(child.name).suffix.lower()
            if suffix in allowed_exts:
                paths.append(f"{prefix}{child.name}")
        elif child.entry_type == EntryType.DIRECTORY:
            # Recurse into subdirectories within the agentic folder
            paths.extend(_collect_folder_files(child, f"{prefix}{child.name}/", allowed_exts))
    return paths


def _scan_tree(node: TreeNode, prefix: str = "") -> list[str]:
    """Recursively scan a tree for agentic context files.

    Detects:
    - Single files matching _AGENTIC_SINGLE_FILES at any depth
    - Files inside known agentic folders matching _AGENTIC_FOLDER_PATTERNS
    """
    paths: list[str] = []
    for child in node.children:
        rel = f"{prefix}{child.name}" if prefix else child.name
        if child.entry_type == EntryType.DIRECTORY:
            allowed_exts = _is_agentic_folder(child.name)
            if allowed_exts is not None:
                # Collect matching files directly inside this folder
                paths.extend(_collect_folder_files(child, f"{rel}/", allowed_exts))
            else:
                # Recurse into non-agentic directories
                paths.extend(_scan_tree(child, f"{rel}/"))
        elif _matches_single_file(child.name):
            paths.append(rel)
    return paths


def detect_agentic_contexts(scope_root: Path) -> list[str]:
    """Detect agentic context files in a scope directory.

    Single tree scan at depth 3 to find AI agent prompts, instructions,
    and configuration files.

    Args:
        scope_root: Absolute path to the scope root directory.

    Returns:
        List of relative file paths for detected agentic contexts.
    """
    try:
        fs = FileSystem(scope_root, create_if_missing=False)
    except Exception:
        logger.debug(f"Cannot access scope root for agentic detection: {scope_root}")
        return []

    tree = fs.get_tree(max_depth=3, include_hidden=True)
    agentic_files = _scan_tree(tree)

    if agentic_files:
        logger.info(f"Detected {len(agentic_files)} agentic context(s) in {scope_root}: {agentic_files}")

    return sorted(agentic_files)


def extract_agentic_content(scope_root: Path, context_paths: list[str]) -> str:
    """Read and concatenate agentic context file contents.

    Args:
        scope_root: Absolute path to the scope root directory.
        context_paths: List of relative paths to agentic context files.

    Returns:
        Concatenated content with ``=== filename ===`` headers,
        or empty string if no files or all reads fail.
    """
    if not context_paths:
        return ""

    try:
        fs = FileSystem(scope_root, create_if_missing=False)
    except Exception:
        logger.debug(f"Cannot access scope root for agentic extraction: {scope_root}")
        return ""

    parts: list[str] = []
    for path in context_paths:
        try:
            content = fs.read_file(path)
            parts.append(f"=== {path} ===\n{content.content}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not read agentic context {path}: {e}")

    return "\n\n".join(parts)
