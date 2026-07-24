"""Persistence helpers for ContextMap — filesystem and S3 backends."""

from __future__ import annotations

from codespy.agents.hippocampus.context_map import ContextMap
from codespy.tools.storage.base import Storage


def save_map(store: Storage, path: str, cmap: ContextMap) -> None:
    """Serialize ``cmap`` to JSON and write it to ``path`` via ``store``.

    Args:
        store: A ``FileSystem`` or ``S3Client`` instance.
        path: Destination path (relative to the store's root / bucket).
        cmap: The context map to persist.

    Raises:
        IOError: If the write operation fails.
    """
    result = store.write_file(path, cmap.to_json(), content_type="application/json")
    if not result.success:
        raise IOError(f"Failed to save context map to {path!r}: {result.error}")


def load_map(store: Storage, path: str) -> ContextMap:
    """Load a context map from ``path`` via ``store``.

    Args:
        store: A ``FileSystem`` or ``S3Client`` instance.
        path: Source path (relative to the store's root / bucket).

    Returns:
        A ``ContextMap`` with ``next_id`` recomputed from loaded item IDs.

    Raises:
        FileNotFoundError: If the path does not exist in the store.
        IOError: If reading or parsing fails.
    """
    result = store.read_file(path)
    if not result.success:
        error = result.error or ""
        if "not found" in error.lower() or "NoSuchKey" in error:
            raise FileNotFoundError(f"Context map not found at {path!r}: {error}")
        raise IOError(f"Failed to load context map from {path!r}: {error}")
    if not result.content:
        raise IOError(f"Context map at {path!r} is empty")
    try:
        return ContextMap.from_json(result.content)
    except Exception as exc:
        raise IOError(f"Failed to parse context map from {path!r}: {exc}") from exc
