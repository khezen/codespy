"""Episode record and persistence helpers for Hippocampus."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from codespy.agents.memory.hippocampus.context_memory import ContextMemory, Mutation
from codespy.tools.storage.base import Storage
from codespy.tools.storage.models import Entry, EntryType


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


def find_latest_episode(
    store: Storage,
    dir: str,
    task: str | None = None,
    exclude_run_id: str | None = None,
) -> Episode | None:
    """Find and load the most recent episode for a given scope path.

    Searches ``orgs/{owner}/episodic/.codespy/`` for episodes whose filename
    starts with the slug derived from ``dir`` (same logic as
    ``Hippocampus.episode_file_path``). Optionally filters by task name and
    excludes a specific run_id.

    Args:
        store: Storage backend (FileSystem or S3Client).
        dir: Scope directory path (e.g., "/{repo_slug}/{subroot}/").
            Host segments (containing a dot) are stripped automatically.
        task: Optional task filter (e.g., "scope", "summary").
            Matches ``-{task}-`` substring in filename remainder.
            If None, any task matches.
        exclude_run_id: If set, skip episodes containing this run_id in
            filename (avoids loading current pipeline's own episodes).

    Returns:
        The most recent Episode by modified_at, or None if no matches found.
    """
    # Compute slug and episodic directory (mirrors Hippocampus.episode_file_path)
    segments = [s for s in dir.strip("/").split("/") if s]
    if segments and "." in segments[0]:
        segments = segments[1:]
    if not segments:
        return None
    owner = segments[0]
    slug = ".".join(segments)
    episodic_dir = f"orgs/{owner}/episodic/.codespy"

    try:
        listing = store.list_directory(episodic_dir)
    except (FileNotFoundError, OSError):
        return None
    # Filter entries: prefix match + optional task + exclude run_id
    # Filename: {slug}.{run_id}-{task}-{index}.json
    prefix = f"{slug}."
    candidates: list[Entry] = []
    for entry in listing.entries:
        if entry.entry_type != EntryType.FILE:
            continue
        if not entry.name.startswith(prefix):
            continue
        remainder = entry.name[len(prefix):]
        if task is not None and f"-{task}-" not in remainder:
            continue
        if exclude_run_id and exclude_run_id in remainder:
            continue
        candidates.append(entry)
    if not candidates:
        return None
    # Sort by modified_at descending; epoch fallback for entries without timestamp
    _epoch = datetime.min.replace(tzinfo=UTC)
    candidates.sort(
        key=lambda e: e.modified_at if e.modified_at is not None else _epoch,
        reverse=True,
    )
    # Load the newest candidate
    path = f"{episodic_dir}/{candidates[0].name}"
    try:
        return load_episode(store, path)
    except (FileNotFoundError, OSError):
        return None
