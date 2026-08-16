"""Episode record and persistence helpers for Hippocampus."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from codespy.agents.memory.hippocampus.context_memory import ContextMemory, Mutation
from codespy.tools.storage.base import Storage


class Episode(BaseModel):
    """A snapshot of an agent's consolidated memory at the end of an episode.

    Recorded by ``Hippocampus.end_episode()`` after the buffered trajectories
    have been distilled into the context memory. It captures *what the agent knew*
    (the consolidated ``ContextMemory``) together with lightweight identity and
    timing metadata, so a review/run leaves behind a durable, inspectable
    record of the memory it produced.

    Attributes:
        task: Name of the wrapped agent's top-level signature (e.g.
            ``"CodeReviewSignature"``). Falls back to the module class name when
            the wrapped module exposes no signature.
        module: Class name of the wrapped ``dspy.Module`` (e.g. ``"CodeReviewer"``).
        question: Question/task description derived from the first buffered
            call's inputs (via ``question_field`` or serialized input fields).
        context_memory: Deep-copied snapshot of the context memory *after*
            consolidation, so later edits to the live memory do not mutate this
            record.
        timestamp: UTC time the episode was recorded.
        artifacts: Named output artifacts produced by the wrapped agent for
            this episode (e.g. ``{"review": "<markdown>"}``). Agent-agnostic —
            any caller can attach whatever markdown/text output it
            produced under a key of its choosing. Empty by default.
        run_id: Identifier of the pipeline run that produced this episode.
            Shared by every agent/module invoked within the same
            ``ReviewPipeline.forward()`` call, so all episodes from one
            review run can be correlated. Also used as the ``<uuid>`` prefix
            in the episode filename: ``<run_id>-<task>-<index>.json``.
        mutations: Ordered sequence of Cartographer mutations applied during this episode.
    """
    run_id: str = Field(
        default="",
        description=(
            "Identifier of the pipeline run that produced this episode. "
            "Shared across all agents invoked within the same review run."
        ),
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC time the episode was recorded",
    )
    task: str = Field(description="Wrapped signature name (or module class name as fallback)")
    module: str = Field(description="Wrapped dspy.Module class name")
    question: str = Field(description="Question/task description for this episode (passed as 'question' or derived from serialized inputs)")
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Named output artifacts produced by the agent (e.g. {'review': '<markdown>'})",
    )
    context_memory: ContextMemory = Field(description="Consolidated context memory snapshot")
    mutations: list[Mutation] = Field(
        default_factory=list,
        description="Ordered sequence of Cartographer mutations applied during this episode",
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

    Args:
        store: A ``FileSystem`` or ``S3Client`` instance.
        path: Source path (relative to the store's root / bucket).

    Returns:
        The loaded ``Episode``.

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
    return episode
