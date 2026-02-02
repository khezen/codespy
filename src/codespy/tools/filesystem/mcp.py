"""MCP server for filesystem operations."""

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from codespy.tools.filesystem.client import FileSystem

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
    return _get_fs().read_file(path, max_bytes, max_lines).model_dump()


@mcp.tool()
def list_directory(path: str = "", include_hidden: bool = False) -> dict:
    """List contents of a directory.

    Args:
        path: Relative path to directory
        include_hidden: Whether to include hidden files

    Returns:
        Dict with path, entries, total_files, total_directories
    """
    return _get_fs().list_directory(path, include_hidden).model_dump()


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
    return _get_fs().get_tree_string(path, max_depth, include_hidden)


@mcp.tool()
def file_exists(path: str = "") -> bool:
    """Check if a path exists.

    Args:
        path: Relative path to check

    Returns:
        True if path exists
    """
    return _get_fs().exists(path)


@mcp.tool()
def get_file_info(path: str = "") -> dict:
    """Get information about a file or directory.

    Args:
        path: Relative path

    Returns:
        Dict with path, name, entry_type, size, modified_at, extension
    """
    return _get_fs().get_info(path).model_dump()


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    _fs = FileSystem(root)
    mcp.run()
