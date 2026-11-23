"""MCP server for filesystem operations."""

import functools
import logging
import os
from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from codespy.tools.storage.filesystem.client import FileSystem, InvalidPathError
from codespy.tools.storage.filesystem.protected import is_protected_path, normalize_rel_path

logger = logging.getLogger(__name__)

# Failures caused by the agent's arguments (missing path, file vs directory, a path
# that escapes the root or is invalid). Raised as ToolError so MCP returns the message
# to the agent instead of logging a crash traceback and hiding the reason. Other
# ValueErrors are bugs and stay crashes.
_EXPECTED_ERRORS = (FileNotFoundError, NotADirectoryError, IsADirectoryError,
                    PermissionError, InvalidPathError)

_F = TypeVar("_F", bound=Callable[..., Any])


def _agent_errors(fn: _F) -> _F:
    """Re-raise expected filesystem errors as ToolError (message kept)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except _EXPECTED_ERRORS as e:
            raise ToolError(str(e)) from e

    return wrapper  # type: ignore[return-value]

# Get caller module from environment (set by mcp_utils.py)
_caller_module = os.environ.get("MCP_CALLER_MODULE", "unknown")

mcp = MCPServer("filesystem")
_fs: FileSystem | None = None

# Manual cache for read_file to skip caching error results
_READ_FILE_CACHE_MAX = 512
_read_file_cache: OrderedDict[tuple[str, int, int | None], tuple] = OrderedDict()

# True once the write tools are registered (--writable)
_write_tools_registered = False


def _get_fs() -> FileSystem:
    """Get the FileSystem instance, raising if not initialized."""
    if _fs is None:
        raise RuntimeError("FileSystem not initialized")
    return _fs


def _read_file_cached(path: str, max_bytes: int, max_lines: int | None) -> tuple:
    """Cached version of read_file that doesn't cache error results."""
    key = (path, max_bytes, max_lines)
    if key in _read_file_cache:
        _read_file_cache.move_to_end(key)
        return _read_file_cache[key]
    result = _get_fs().read_file(path, max_bytes, max_lines)
    dumped = tuple(sorted(result.model_dump().items()))
    if not result.error:
        _read_file_cache[key] = dumped
        if len(_read_file_cache) > _READ_FILE_CACHE_MAX:
            _read_file_cache.popitem(last=False)
    return dumped


@mcp.tool()
@_agent_errors
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
@_agent_errors
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
@_agent_errors
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
@_agent_errors
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
@_agent_errors
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


# ------------------------------------------------------------------
# Write tools (registered only with --writable; see register_write_tools)
# ------------------------------------------------------------------


def _clear_caches() -> None:
    """Clear every read cache after a write, so later reads see the new content."""
    _read_file_cache.clear()
    _list_directory_cached.cache_clear()
    _get_tree_cached.cache_clear()
    _file_exists_cached.cache_clear()
    _get_file_info_cached.cache_clear()


def _check_writable(path: str, operation: str) -> Path:
    """Resolve ``path`` for a write, or raise ToolError.

    Refuses protected paths (``.git``, ``.github/workflows``, ``.codespy``),
    symlinks (the target or any parent directory under the root) and paths
    escaping the root.
    """
    fs = _get_fs()
    rel = normalize_rel_path(path)
    if is_protected_path(rel):
        raise ToolError(f"{operation} refused: protected path: {path!r}")
    current = fs.root
    for part in rel.split("/"):
        current = current / part
        if current.is_symlink():
            raise ToolError(f"{operation} refused: symlink in path: {path!r}")
    resolved = fs._resolve_path(rel)  # InvalidPathError (-> ToolError) when escaping the root
    if is_protected_path(str(resolved.relative_to(fs.root))):
        raise ToolError(f"{operation} refused: protected path: {path!r}")
    return resolved


def _result(result: Any) -> dict:
    """Return a successful OperationResult as a dict; raise ToolError on failure."""
    if not result.success:
        raise ToolError(result.error or "operation failed")
    _clear_caches()
    return dict(result.model_dump())


@_agent_errors
def write_file(path: str, content: str) -> dict:
    """Create or overwrite a text file (UTF-8). Parent directories are created.

    Args:
        path: Path relative to the repository root
        content: Full new file content

    Returns:
        Dict with success, path and message
    """
    resolved = _check_writable(path, "write")
    logger.info(f"[FS] {_caller_module} -> write_file: {resolved}")
    return _result(_get_fs().write_file(normalize_rel_path(path), content))


@_agent_errors
def replace_in_file(path: str, old: str, new: str) -> dict:
    """Replace one exact occurrence of ``old`` with ``new`` in a text file.

    Fails when ``old`` is found zero or several times: include more
    surrounding lines in ``old`` to make it unique.

    Args:
        path: Path relative to the repository root
        old: Exact text to replace (must occur exactly once)
        new: Replacement text

    Returns:
        Dict with success, path and message
    """
    resolved = _check_writable(path, "replace")
    logger.info(f"[FS] {_caller_module} -> replace_in_file: {resolved}")
    return _result(_get_fs().replace_in_file(normalize_rel_path(path), old, new))


@_agent_errors
def delete_file(path: str) -> dict:
    """Delete a file.

    Args:
        path: Path relative to the repository root

    Returns:
        Dict with success, path and message
    """
    resolved = _check_writable(path, "delete")
    logger.info(f"[FS] {_caller_module} -> delete_file: {resolved}")
    return _result(_get_fs().delete_file(normalize_rel_path(path)))


def register_write_tools() -> None:
    """Expose write_file / replace_in_file / delete_file (idempotent)."""
    global _write_tools_registered
    if _write_tools_registered:
        return
    for fn in (write_file, replace_in_file, delete_file):
        mcp.tool()(fn)
    _write_tools_registered = True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP filesystem server")
    parser.add_argument("root", nargs="?", default=".", help="Root directory")
    parser.add_argument(
        "--writable",
        action="store_true",
        help="Expose write tools (write_file, replace_in_file, delete_file)",
    )
    args = parser.parse_args()
    _fs = FileSystem(args.root)
    if args.writable:
        register_write_tools()
    mcp.run()
