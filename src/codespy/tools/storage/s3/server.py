"""MCP server for S3 filesystem-like operations."""

import logging
import os
import sys
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from codespy.tools.storage.s3.client import S3Client

logger = logging.getLogger(__name__)

# Get caller module from environment (set by mcp_utils.py)
_caller_module = os.environ.get("MCP_CALLER_MODULE", "unknown")

mcp = FastMCP("s3")
_client: S3Client | None = None


def _get_client() -> S3Client:
    """Get the S3Client instance, raising if not initialized."""
    if _client is None:
        raise RuntimeError("S3Client not initialized")
    return _client


# ------------------------------------------------------------------
# Read tools (cached, like filesystem/server.py)
# ------------------------------------------------------------------


@lru_cache(maxsize=512)
def _file_exists_cached(path: str) -> bool:
    """Cached version of exists."""
    return _get_client().exists(path)


@mcp.tool()
def file_exists(path: str = "") -> bool:
    """Check if a file or directory exists in the bucket.

    Args:
        path: Relative path to check (empty = bucket root)

    Returns:
        True if the path exists
    """
    client = _get_client()
    logger.info(f"[S3] {_caller_module} -> file_exists: s3://{client.bucket}/{path}")
    return _file_exists_cached(path)


@lru_cache(maxsize=256)
def _get_file_info_cached(path: str) -> tuple:
    """Cached version of get_info."""
    result = _get_client().get_info(path)
    return tuple(sorted(result.model_dump().items()))


@mcp.tool()
def get_file_info(path: str = "") -> dict:
    """Get information about a file or directory.

    Args:
        path: Relative path (empty = bucket root)

    Returns:
        Dict with path, name, entry_type, size, modified_at, extension, etag, storage_class
    """
    client = _get_client()
    logger.info(f"[S3] {_caller_module} -> get_file_info: s3://{client.bucket}/{path}")
    return dict(_get_file_info_cached(path))


@lru_cache(maxsize=256)
def _list_directory_cached(path: str, include_hidden: bool) -> tuple:
    """Cached version of list_directory."""
    result = _get_client().list_directory(path, include_hidden)
    return tuple(sorted(result.model_dump().items()))


@mcp.tool()
def list_directory(path: str = "", include_hidden: bool = False) -> dict:
    """List files and subdirectories directly under a path (one level deep).

    Args:
        path: Relative directory path (empty = bucket root)
        include_hidden: Whether to include entries starting with '.'

    Returns:
        Dict with path, entries, total_files, total_directories
    """
    client = _get_client()
    logger.info(f"[S3] {_caller_module} -> list_directory: s3://{client.bucket}/{path}")
    return dict(_list_directory_cached(path, include_hidden))


@lru_cache(maxsize=128)
def _get_tree_cached(path: str, max_depth: int, include_hidden: bool) -> str:
    """Cached version of get_tree_string."""
    return _get_client().get_tree_string(path, max_depth, include_hidden)


@mcp.tool()
def get_tree(path: str = "", max_depth: int = 3, include_hidden: bool = False) -> str:
    """Get a string representation of the directory tree.

    Args:
        path: Relative directory path (empty = bucket root)
        max_depth: Maximum depth to traverse
        include_hidden: Whether to include entries starting with '.'

    Returns:
        String representation of the directory tree
    """
    client = _get_client()
    logger.info(
        f"[S3] {_caller_module} -> get_tree: s3://{client.bucket}/{path} (depth={max_depth})"
    )
    return _get_tree_cached(path, max_depth, include_hidden)


@lru_cache(maxsize=256)
def _read_file_cached(path: str, max_bytes: int, max_lines: int | None) -> tuple:
    """Cached version of read_file."""
    result = _get_client().read_file(path, max_bytes, max_lines)
    return tuple(sorted(result.model_dump().items()))


@mcp.tool()
def read_file(path: str, max_bytes: int = 100_000, max_lines: int | None = None) -> dict:
    """Read a file from the bucket as text.

    Args:
        path: Relative file path
        max_bytes: Maximum bytes to read (default 100 KB)
        max_lines: Maximum lines to read (optional)

    Returns:
        Dict with path, content, size, lines, truncated, content_type, error (if any)
    """
    client = _get_client()
    logger.info(f"[S3] {_caller_module} -> read_file: s3://{client.bucket}/{path}")
    return dict(_read_file_cached(path, max_bytes, max_lines))


# ------------------------------------------------------------------
# Cache invalidation
# ------------------------------------------------------------------


def _invalidate_read_caches() -> None:
    """Clear all cached read results after a mutating operation.

    lru_cache has no per-key eviction, and a single write or delete can
    invalidate entries across several caches - including listings and trees
    for every ancestor prefix - so all read caches are cleared wholesale.
    Called unconditionally, since a failed put/delete may still have changed
    bucket state (timeouts, partial uploads, ambiguous retries).
    """
    _file_exists_cached.cache_clear()
    _get_file_info_cached.cache_clear()
    _list_directory_cached.cache_clear()
    _get_tree_cached.cache_clear()
    _read_file_cached.cache_clear()


# ------------------------------------------------------------------
# Write tools (not cached)
# ------------------------------------------------------------------


@mcp.tool()
def write_file(path: str, content: str, content_type: str = "text/plain") -> dict:
    """Write text content to a file in the bucket.

    Args:
        path: Relative file path to write to
        content: Text content to write (encoded as UTF-8)
        content_type: MIME type for the object (default 'text/plain')

    Returns:
        Dict with success, path, message, error (if any)
    """
    client = _get_client()
    logger.info(f"[S3] {_caller_module} -> write_file: s3://{client.bucket}/{path}")
    result = client.write_file(path, content, content_type)
    _invalidate_read_caches()
    return result.model_dump()


@mcp.tool()
def delete_file(path: str) -> dict:
    """Delete a file from the bucket.

    Args:
        path: Relative file path to delete

    Returns:
        Dict with success, path, message, error (if any)
    """
    client = _get_client()
    logger.info(f"[S3] {_caller_module} -> delete_file: s3://{client.bucket}/{path}")
    result = client.delete_file(path)
    _invalidate_read_caches()
    return result.model_dump()


if __name__ == "__main__":
    # Suppress noisy MCP server "Processing request" logs
    logging.getLogger("mcp.server").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)

    if len(sys.argv) < 2:
        print("Usage: python server.py <bucket> [region] [endpoint_url]", file=sys.stderr)
        sys.exit(1)

    bucket = sys.argv[1]
    region = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("AWS_DEFAULT_REGION")
    endpoint_url = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("S3_ENDPOINT_URL")

    _client = S3Client(bucket=bucket, region=region, endpoint_url=endpoint_url)
    mcp.run()
