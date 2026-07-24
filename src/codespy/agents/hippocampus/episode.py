"""Episode record and persistence helpers for Hypocampus."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from codespy.agents.hippocampus.context_map import ContextMap, _recompute_next_id
from codespy.tools.storage.base import Storage


class Episode(BaseModel):
    """A snapshot of an agent's consolidated memory at the end of an episode.

    Recorded by ``Hypocampus.end_episode()`` after the buffered trajectories
    have been distilled into the context map. It captures *what the agent knew*
    (the consolidated ``ContextMap``) together with lightweight identity and
    timing metadata, so a review/run leaves behind a durable, inspectable
    record of the memory it produced.

    Attributes:
        task: Name of the wrapped agent's top-level signature (e.g.
            ``"CodeReviewSignature"``). Falls back to the module class name when
            the wrapped module exposes no signature.
        module: Class name of the wrapped ``dspy.Module`` (e.g. ``"CodeReviewer"``).
        context_map: Deep-copied snapshot of the context map *after*
            consolidation, so later edits to the live map do not mutate this
            record.
        timestamp: UTC time the episode was recorded.
    """

    task: str = Field(description="Wrapped signature name (or module class name as fallback)")
    module: str = Field(description="Wrapped dspy.Module class name")
    context_map: ContextMap = Field(description="Consolidated context map snapshot")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="UTC time the episode was recorded"
    )


def save_episode(store: Storage, path: str, episode: Episode) -> None:
    """Serialise ``episode`` to JSON and write it to ``path`` via ``store``.

    Args:
        store: A ``FileSystem`` or ``S3Client`` instance.
        path: Destination path (relative to the store's root / bucket).
        episode: The episode to persist.

    Raises:
        OSError: If the write operation fails.
    """
    result = store.write_file(
        path, episode.model_dump_json(indent=2), content_type="application/json"
    )
    if not result.success:
        raise OSError(f"Failed to save episode to {path!r}: {result.error}")


def load_episode(store: Storage, path: str) -> Episode:
    """Load an episode from ``path`` via ``store``.

    The embedded ``ContextMap.next_id`` is recomputed from the loaded item IDs
    so the restored map can safely receive further ADD operations.

    Args:
        store: A ``FileSystem`` or ``S3Client`` instance.
        path: Source path (relative to the store's root / bucket).

    Returns:
        An ``Episode`` with ``context_map.next_id`` recomputed.

    Raises:
        FileNotFoundError: If the path does not exist in the store.
        OSError: If reading or parsing fails.
    """
    result = store.read_file(path)
    if not result.success:
        error = result.error or ""
        if "not found" in error.lower() or "NoSuchKey" in error:
            raise FileNotFoundError(f"Episode not found at {path!r}: {error}")
        raise OSError(f"Failed to load episode from {path!r}: {error}")
    if not result.content:
        raise OSError(f"Episode at {path!r} is empty")
    try:
        episode = Episode.model_validate_json(result.content)
    except Exception as exc:
        raise OSError(f"Failed to parse episode from {path!r}: {exc}") from exc
    _recompute_next_id(episode.context_map)
    return episode
