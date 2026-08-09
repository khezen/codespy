"""Deterministic documentation extractor — single tree scan, no LLM."""

import logging
import re
from pathlib import Path

from codespy.tools.storage import EntryType, FileSystem, TreeNode

logger = logging.getLogger(__name__)

# Filename patterns recognised as documentation (case-insensitive).
_DOC_FILE_RE = re.compile(
    r"^(readme|changelog|contributing|\.env\.example)",
    re.IGNORECASE,
)
# Top-level directory names whose *entire* contents count as docs.
_DOC_DIRS = {"docs", "documentation"}


def _collect_all_files(node: TreeNode, prefix: str) -> list[str]:
    """Recursively collect all file paths under a tree node."""
    paths: list[str] = []
    for child in node.children:
        rel = f"{prefix}{child.name}"
        if child.entry_type == EntryType.DIRECTORY:
            paths.extend(_collect_all_files(child, f"{rel}/"))
        else:
            paths.append(rel)
    return paths


def _collect_doc_paths(node: TreeNode, prefix: str = "") -> list[str]:
    """Walk a shallow tree and return relative paths of doc files.

    Matches:
    - Root-level files: README*, CHANGELOG*, CONTRIBUTING*, .env.example
    - Everything under docs/ or documentation/ directories
    """
    paths: list[str] = []
    for child in node.children:
        rel = f"{prefix}{child.name}" if prefix else child.name
        if child.entry_type == EntryType.DIRECTORY:
            if child.name.lower() in _DOC_DIRS:
                paths.extend(_collect_all_files(child, f"{rel}/"))
            else:
                # Recurse into non-doc subdirs (depth-2 tree already limits this).
                paths.extend(_collect_doc_paths(child, f"{rel}/"))
        elif _DOC_FILE_RE.match(child.name):
            paths.append(rel)
    return paths


def extract_documentation(scope_root: Path) -> str:
    """Extract documentation content from a scope directory.

    Single tree scan at depth 2, fast-fail if no doc files found.

    Returns:
        Concatenated documentation with ``=== filename ===`` headers,
        or empty string if no documentation exists.
    """
    if not scope_root.exists():
        logger.debug("Scope root does not exist, skipping: %s", scope_root)
        return ""
    fs = FileSystem(scope_root, create_if_missing=False)

    # One tree scan — depth 2 covers root files + immediate subdirs.
    tree = fs.get_tree(max_depth=2)
    doc_paths = _collect_doc_paths(tree)

    if not doc_paths:
        return ""

    parts: list[str] = []
    for path in doc_paths:
        try:
            content = fs.read_file(path)
        except Exception as e:  # noqa: BLE001
            # read_file returns an error Content rather than raising for missing
            # or unreadable files, but path resolution and stat() can still raise.
            logger.warning(f"Could not read doc file {path}: {e}")
            continue

        # read_file signals failure via Content.error, leaving content empty. An
        # unchecked append would emit a header with a blank body, which reads to
        # the LLM as "this doc exists and is empty" rather than "not available"
        # — prompting false "add missing docs" findings.
        if not content.success:
            logger.debug(f"Skipping unreadable doc file {path}: {content.error}")
            continue

        if not content.content.strip():
            logger.debug(f"Skipping empty doc file {path}")
            continue

        parts.append(f"=== {path} ===\n{content.content}")

    return "\n\n".join(parts)
