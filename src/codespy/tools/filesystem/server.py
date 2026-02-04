"""MCP server for filesystem operations."""

import logging
import os
import sys
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
    return fs.read_file(path, max_bytes, max_lines).model_dump()


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
    return fs.list_directory(path, include_hidden).model_dump()


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
    return fs.get_tree_string(path, max_depth, include_hidden)


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
    return fs.exists(path)


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
    return fs.get_info(path).model_dump()


if __name__ == "__main__":
    # Suppress noisy MCP server "Processing request" logs
    logging.getLogger("mcp.server").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)
    
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    _fs = FileSystem(root)
    mcp.run()
