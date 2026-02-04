"""MCP server for filesystem operations."""

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from codespy.tools.filesystem.client import FileSystem

logger = logging.getLogger(__name__)

# Get caller module from environment (set by mcp_utils.py)
_caller_module = os.environ.get("MCP_CALLER_MODULE", "unknown")

mcp = FastMCP("filesystem")
_fs: FileSystem | None = None


def _get_fs() -> FileSystem:
    """Get the FileSystem instance, raising if not initialized."""
    if _fs is None:
        raise RuntimeError("FileSystem not initialized")
    return _fs


@lru_cache(maxsize=256)
def _read_file_cached(path: str, max_bytes: int, max_lines: int | None) -> tuple:
    """Cached version of read_file."""
    result = _get_fs().read_file(path, max_bytes, max_lines)
    return tuple(sorted(result.model_dump().items()))


@mcp.tool()
def read_file(path: str, max_bytes: int = 100_000, max_lines: int | None = None) -> dict:
    """Read contents of a file.

    Args:
        path: Relative path to file
        max_bytes: Maximum bytes to read (default 100KB)
        max_lines: Maximum lines to read (optional)

    Returns:
        Dict with path, content, size, lines, and truncated flag
    """
    fs = _get_fs()
    resolved = fs.root / path if path else fs.root
    logger.info(f"[FS] {_caller_module} -> read_file: {resolved}")
    return dict(_read_file_cached(path, max_bytes, max_lines))


@lru_cache(maxsize=256)
def _list_directory_cached(path: str, include_hidden: bool) -> tuple:
    """Cached version of list_directory."""
    result = _get_fs().list_directory(path, include_hidden)
    return tuple(sorted(result.model_dump().items()))


@mcp.tool()
def list_directory(path: str = "", include_hidden: bool = False) -> dict:
    """List contents of a directory.

    Args:
        path: Relative path to directory
        include_hidden: Whether to include hidden files

    Returns:
        Dict with path, entries, total_files, total_directories
    """
    fs = _get_fs()
    resolved = fs.root / path if path else fs.root
    logger.info(f"[FS] {_caller_module} -> list_directory: {resolved}")
    return dict(_list_directory_cached(path, include_hidden))


@lru_cache(maxsize=128)
def _get_tree_cached(path: str, max_depth: int, include_hidden: bool) -> str:
    """Cached version of get_tree."""
    return _get_fs().get_tree_string(path, max_depth, include_hidden)


@mcp.tool()
def get_tree(path: str = "", max_depth: int = 3, include_hidden: bool = False) -> str:
    """Get string representation of directory tree.

    Args:
        path: Relative path to directory
        max_depth: Maximum depth to traverse
        include_hidden: Whether to include hidden files

    Returns:
        String representation of the directory tree
    """
    fs = _get_fs()
    resolved = fs.root / path if path else fs.root
    logger.info(f"[FS] {_caller_module} -> get_tree: {resolved} (depth={max_depth})")
    return _get_tree_cached(path, max_depth, include_hidden)


@lru_cache(maxsize=512)
def _file_exists_cached(path: str) -> bool:
    """Cached version of file_exists."""
    return _get_fs().exists(path)


@mcp.tool()
def file_exists(path: str = "") -> bool:
    """Check if a path exists.

    Args:
        path: Relative path to check

    Returns:
        True if path exists
    """
    fs = _get_fs()
    resolved = fs.root / path if path else fs.root
    logger.info(f"[FS] {_caller_module} -> file_exists: {resolved}")
    return _file_exists_cached(path)


@lru_cache(maxsize=256)
def _get_file_info_cached(path: str) -> tuple:
    """Cached version of get_file_info."""
    result = _get_fs().get_info(path)
    return tuple(sorted(result.model_dump().items()))


@mcp.tool()
def get_file_info(path: str = "") -> dict:
    """Get information about a file or directory.

    Args:
        path: Relative path

    Returns:
        Dict with path, name, entry_type, size, modified_at, extension
    """
    fs = _get_fs()
    resolved = fs.root / path if path else fs.root
    logger.info(f"[FS] {_caller_module} -> get_file_info: {resolved}")
    return dict(_get_file_info_cached(path))


if __name__ == "__main__":
    # Suppress noisy MCP server "Processing request" logs
    logging.getLogger("mcp.server").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)
    
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    _fs = FileSystem(root)
    mcp.run()
