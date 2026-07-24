"""Persistence helpers for ContextMap — filesystem and S3 backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from codespy.agents.hippocampus.context_map import ContextMap


@runtime_checkable
class MapStore(Protocol):
    """Structural protocol satisfied by both ``FileSystem`` and ``S3Client``.

    Any object that provides ``read_file(path)`` / ``write_file(path, content)``
    with these signatures can be passed to ``Hypocampus.save_map`` /
    ``Hypocampus.load_map``.
    """

    def read_file(self, path: str, **kwargs) -> object: ...  # returns object with .content / .error
    def write_file(self, path: str, content: str, **kwargs) -> object: ...  # returns object with .success


def save_map(store: MapStore, path: str, cmap: ContextMap) -> None:
    """Serialize ``cmap`` to JSON and write it to ``path`` via ``store``.

    Args:
        store: A ``FileSystem`` or ``S3Client`` instance.
        path: Destination path (relative to the store's root / bucket).
        cmap: The context map to persist.

    Raises:
        IOError: If the write operation fails.
    """
    result = store.write_file(path, cmap.to_json(), content_type="application/json")
    # Both FileSystem and S3Client return an object with a .success attribute.
    # FileSystem.write_file returns None; S3Client returns OperationResult.
    # Normalise: treat None (filesystem) as success; check .success for S3.
    if result is not None and not getattr(result, "success", True):
        error = getattr(result, "error", "unknown error")
        raise IOError(f"Failed to save context map to {path!r}: {error}")


def load_map(store: MapStore, path: str) -> ContextMap:
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
    # Both clients return a result object with .content and .error.
    error = getattr(result, "error", None)
    if error is not None:
        if "not found" in str(error).lower() or "NoSuchKey" in str(error):
            raise FileNotFoundError(f"Context map not found at {path!r}: {error}")
        raise IOError(f"Failed to load context map from {path!r}: {error}")
    content = getattr(result, "content", "")
    if not content:
        raise IOError(f"Context map at {path!r} is empty")
    try:
        return ContextMap.from_json(content)
    except Exception as exc:
        raise IOError(f"Failed to parse context map from {path!r}: {exc}") from exc
