"""Patch compaction utilities to expand diff context to function boundaries."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codespy.agents.reviewer.models import ScopeResult

from codespy.tools.git.models import ChangedFile, FileStatus
from codespy.tools.parsers.treesitter import FunctionInfo, TreeSitterParser

logger = logging.getLogger(__name__)

# Non-code file extensions that should not be processed
NON_CODE_EXTENSIONS = {
    "md",
    "txt",
    "rst",
    "yaml",
    "yml",
    "json",
    "toml",
    "ini",
    "cfg",
    "conf",
    "xml",
    "html",
    "htm",
    "css",
    "scss",
    "sass",
    "less",
    "csv",
    "tsv",
}

# Compiled regex for hunk header parsing
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def compact_patches(scopes: list[ScopeResult], repo_path: Path) -> None:
    """Compact patches on all changed files across scopes (mutates in place).

    For each code file with a patch, attempts compaction. Non-code files
    and files where compaction fails keep their original patch.

    Args:
        scopes: List of scope results containing changed files
        repo_path: Path to the cloned repository
    """
    parser = TreeSitterParser(repo_path)

    for scope in scopes:
        for file in scope.changed_files:
            if not _should_compact_file(file):
                continue

            try:
                if file.patch is None:
                    continue
                compacted = compact_patch(file.patch, Path(file.filename), repo_path, parser)
                if compacted != file.patch:
                    logger.debug(f"Compacted patch for {file.filename}")
                    file.patch = compacted
            except Exception as e:
                logger.debug(f"Failed to compact patch for {file.filename}: {e}")
                # Keep original patch on failure


def _should_compact_file(file: ChangedFile) -> bool:
    """Check if a file should be compacted.

    Args:
        file: The ChangedFile to check

    Returns:
        True if the file should be compacted
    """
    # Skip files without patches
    if not file.patch:
        return False

    # Skip deleted files (no source to read)
    if file.status == FileStatus.REMOVED:
        return False

    # Skip non-code files
    if file.extension in NON_CODE_EXTENSIONS:
        return False

    # Skip binary and lock files
    if file.is_binary or file.is_lock_file:
        return False

    return True


def compact_patch(
    raw_patch: str,
    file_path: Path,
    repo_path: Path,
    parser: TreeSitterParser | None = None,
) -> str:
    """Expand diff context to full function bodies using TreeSitter.

    Returns compacted patch string, or raw_patch unchanged if compaction
    isn't possible (no TreeSitter, not a code file, deleted file, etc.).

    Args:
        raw_patch: The original unified diff patch
        file_path: Path to the file (relative to repo root)
        repo_path: Path to the repository root
        parser: Optional TreeSitterParser instance (created if not provided)

    Returns:
        Compacted patch or original patch if compaction fails
    """
    # Skip empty patches
    if not raw_patch:
        return raw_patch

    # Create parser if not provided
    if parser is None:
        parser = TreeSitterParser(repo_path)

    # Check if TreeSitter is available
    if not parser.available:
        logger.debug(f"TreeSitter not available, skipping compaction for {file_path}")
        return raw_patch

    # Check if file is a code file we can parse
    extension = file_path.suffix.lstrip(".") if file_path.suffix else ""
    if not extension or extension in NON_CODE_EXTENSIONS:
        return raw_patch

    # Get absolute file path
    abs_file_path = repo_path / file_path
    if not abs_file_path.exists():
        logger.warning(f"Source file not found: {abs_file_path}")
        return raw_patch

    # Find function definitions in the file
    try:
        functions = parser.find_function_definitions(abs_file_path)
    except Exception as e:
        logger.debug(f"Failed to find functions in {file_path}: {e}")
        return raw_patch

    if not functions:
        # No functions found, nothing to expand
        return raw_patch

    # Parse hunks from patch
    hunks = _parse_hunks(raw_patch)
    if not hunks:
        return raw_patch

    # Read source file lines
    try:
        source_lines = abs_file_path.read_text().splitlines()
    except Exception as e:
        logger.debug(f"Failed to read source file {file_path}: {e}")
        return raw_patch

    # Expand each hunk to function boundaries
    expanded_hunks: list[dict[str, Any]] = []
    for hunk in hunks:
        expanded = _expand_hunk_to_functions(hunk, functions, source_lines)
        if expanded:
            expanded_hunks.append(expanded)

    if not expanded_hunks:
        return raw_patch

    # Merge overlapping hunks
    merged_hunks = _merge_hunks(expanded_hunks)

    # Rebuild the compacted patch
    compacted = _rebuild_patch(raw_patch, merged_hunks, source_lines)

    return compacted if compacted else raw_patch


def _parse_hunks(patch: str) -> list[dict[str, Any]]:
    """Parse hunks from a unified diff patch.

    Args:
        patch: The unified diff patch string

    Returns:
        List of hunk dictionaries with parsed info
    """
    hunks = []
    lines = patch.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for hunk header
        match = HUNK_HEADER_RE.match(line)
        if match:
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) else 1
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) else 1

            # Collect hunk lines
            hunk_lines = []
            i += 1
            changed_new_lines = []
            current_new_line = new_start

            while i < len(lines):
                hunk_line = lines[i]

                # Stop at next hunk header or empty line that ends the hunk
                if hunk_line.startswith("@@"):
                    break

                hunk_lines.append(hunk_line)

                # Track changed lines in the new file
                if hunk_line.startswith("+"):
                    changed_new_lines.append(current_new_line)
                    current_new_line += 1
                elif hunk_line.startswith(" "):
                    current_new_line += 1
                elif hunk_line.startswith("-"):
                    pass  # Deleted line, not in new file
                elif hunk_line.startswith("\\"):
                    # "\ No newline at end of file" marker
                    pass

                i += 1

            hunks.append({
                "header": line,
                "old_start": old_start,
                "old_count": old_count,
                "new_start": new_start,
                "new_count": new_count,
                "lines": hunk_lines,
                "changed_new_lines": changed_new_lines,
            })
        else:
            i += 1

    return hunks


def _expand_hunk_to_functions(
    hunk: dict[str, Any],
    functions: list[FunctionInfo],
    source_lines: list[str],
) -> dict[str, Any] | None:
    """Expand a hunk to cover enclosing function boundaries.

    Args:
        hunk: The hunk dictionary
        functions: List of function definitions
        source_lines: Source file lines

    Returns:
        Expanded hunk dictionary or None if no expansion needed
    """
    changed_lines = hunk["changed_new_lines"]
    if not changed_lines:
        return hunk

    # Find enclosing functions for changed lines
    enclosing_functions: list[FunctionInfo] = []
    for line_num in changed_lines:
        # Find innermost enclosing function
        best_match: FunctionInfo | None = None
        for func in functions:
            if func.line_start <= line_num <= func.line_end:
                if best_match is None or (
                    func.line_start >= best_match.line_start
                    and func.line_end <= best_match.line_end
                ):
                    best_match = func

        if best_match and best_match not in enclosing_functions:
            enclosing_functions.append(best_match)

    if not enclosing_functions:
        # No enclosing functions, return hunk unchanged
        return hunk

    # Calculate expansion range
    min_func_start = min(f.line_start for f in enclosing_functions)
    max_func_end = max(f.line_end for f in enclosing_functions)

    # Determine hunk boundaries in new file
    hunk_start_new = hunk["new_start"]
    hunk_end_new = hunk_start_new + hunk["new_count"] - 1
    # Account for lines that don't end with newline
    if hunk["lines"] and hunk["lines"][-1].startswith("\\"):
        hunk_end_new = hunk_start_new + hunk["new_count"]

    # Don't expand if already covering the entire function(s)
    if min_func_start >= hunk_start_new and max_func_end <= hunk_end_new:
        return hunk

    # Calculate expansion
    expansion_start = min(hunk_start_new, min_func_start)
    expansion_end = max(hunk_end_new, max_func_end)

    return {
        "original_hunk": hunk,
        "expansion_start": expansion_start,
        "expansion_end": expansion_end,
        "hunk_start_new": hunk_start_new,
        "hunk_end_new": hunk_end_new,
        "enclosing_functions": enclosing_functions,
    }


def _merge_hunks(expanded_hunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping or adjacent expanded hunks.

    Args:
        expanded_hunks: List of expanded hunk dictionaries

    Returns:
        List of merged hunks
    """
    if len(expanded_hunks) <= 1:
        return expanded_hunks

    # Sort by expansion start
    sorted_hunks = sorted(expanded_hunks, key=lambda h: h.get("expansion_start", 0))

    merged = [sorted_hunks[0]]

    for hunk in sorted_hunks[1:]:
        last = merged[-1]

        # Check if this hunk overlaps or is adjacent to the last merged hunk
        last_end = last.get("expansion_end", 0)
        current_start = hunk.get("expansion_start", 0)

        if current_start <= last_end + 1:  # Overlapping or adjacent
            # Merge: extend the end if needed
            last["expansion_end"] = max(last_end, hunk.get("expansion_end", 0))
            # Keep track of original hunks for reconstruction
            if "original_hunk" in last and "original_hunk" in hunk:
                if "merged_hunks" not in last:
                    last["merged_hunks"] = [last["original_hunk"]]
                last["merged_hunks"].append(hunk["original_hunk"])
        else:
            merged.append(hunk)

    return merged


def _rebuild_patch(
    raw_patch: str,
    merged_hunks: list[dict[str, Any]],
    source_lines: list[str],
) -> str | None:
    """Rebuild the patch with expanded context.

    Args:
        raw_patch: The original patch
        merged_hunks: List of merged expanded hunks
        source_lines: Source file lines

    Returns:
        Rebuilt patch string or None if rebuild fails
    """
    if not merged_hunks:
        return None

    # Split original patch to get header lines (before first hunk)
    lines = raw_patch.split("\n")
    header_lines = []
    for line in lines:
        if line.startswith("@@"):
            break
        header_lines.append(line)

    result_lines = list(header_lines)

    for merged_hunk in merged_hunks:
        expansion_start = merged_hunk.get("expansion_start")
        expansion_end = merged_hunk.get("expansion_end")
        original_hunk = merged_hunk.get("original_hunk", merged_hunk)

        if expansion_start is None or expansion_end is None:
            # No expansion, keep original hunk
            result_lines.append(original_hunk["header"])
            result_lines.extend(original_hunk["lines"])
            continue

        # Get the hunk boundaries
        hunk_start_new = merged_hunk.get("hunk_start_new", expansion_start)
        hunk_end_new = merged_hunk.get("hunk_end_new", expansion_end)

        # Build new hunk lines
        new_hunk_lines = []

        # Add context lines before the original hunk (from expansion_start to hunk_start_new - 1)
        for line_num in range(expansion_start, hunk_start_new):
            if line_num <= len(source_lines):
                new_hunk_lines.append(f" {source_lines[line_num - 1]}")

        # Add all original hunk lines (context + changes).
        # No overlap with expansion context: pre-expansion ends at hunk_start_new,
        # post-expansion begins at hunk_end_new + 1, so interior context is unique.
        original_lines = original_hunk.get("lines", [])
        new_hunk_lines.extend(original_lines)

        # Add context lines after the original hunk (from hunk_end_new + 1 to expansion_end)
        for line_num in range(hunk_end_new + 1, expansion_end + 1):
            if line_num <= len(source_lines):
                new_hunk_lines.append(f" {source_lines[line_num - 1]}")

        # Calculate new hunk header counts
        # For the new file: count context lines and additions
        new_file_count = 0
        for line in new_hunk_lines:
            if line.startswith(" ") or line.startswith("+"):
                new_file_count += 1

        # For the old file: we approximate by using the ratio of original change
        # This is a simplification; the old file line numbers would need full reconstruction
        # We use the original old_count as a reasonable approximation
        old_count = original_hunk.get("old_count", new_file_count)

        # Build new header
        new_header = f"@@ -{expansion_start},{old_count} +{expansion_start},{new_file_count} @@"

        result_lines.append(new_header)
        result_lines.extend(new_hunk_lines)

    return "\n".join(result_lines)
