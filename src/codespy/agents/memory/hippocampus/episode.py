"""Episode record and persistence helpers for Hippocampus."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from codespy.agents.memory.hippocampus.context_memory import ContextMemory, Mutation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Background episode-save registry
# ---------------------------------------------------------------------------
# Modules fire episode consolidation+save in background threads so the review
# pipeline is not blocked.  Threads are registered here and joined before the
# pipeline returns, ensuring episodes are persisted even when the process exits
# shortly afterwards.

_save_threads: list[threading.Thread] = []
_save_lock = threading.Lock()


def submit_episode_save(target: Callable[[], object], *, name: str = "episode-save") -> None:
    """Start *target* in a non-daemon background thread and track it.

    The thread is **non-daemon**: even without an explicit
    ``join_episode_saves()`` call, the Python interpreter will wait for it
    to finish before exiting.  This guarantees episode persistence when the
    main thread returns quickly (e.g. CLI publishes the review and exits).

    Args:
        target: A zero-arg callable that performs the episode save.
        name: Thread name (for debugging / log messages).
    """
    t = threading.Thread(target=target, name=name, daemon=False)
    with _save_lock:
        _save_threads.append(t)
    t.start()


def join_episode_saves(timeout_per_thread: float = 120) -> None:
    """Block until every tracked episode-save thread has finished.

    Called once at the end of ``ReviewPipeline.forward()`` to guarantee
    episodes are persisted before the process may exit.

    Args:
        timeout_per_thread: Max seconds to wait per thread (default 120).
            If a thread exceeds the timeout it is abandoned with a warning.
    """
    with _save_lock:
        threads = list(_save_threads)
        _save_threads.clear()
    for t in threads:
        t.join(timeout=timeout_per_thread)
        if t.is_alive():
            logger.warning("Episode save thread %r did not finish in time", t.name)


class Episode(BaseModel):
    """A snapshot of an agent's consolidated memory at the end of an episode.

    Recorded by ``Hippocampus.end_episode()`` after the buffered trajectories
    have been distilled into the context memory. It captures *what the agent knew*
    (the consolidated ``ContextMemory``) together with lightweight identity and
    timing metadata, so a review/run leaves behind a durable, inspectable
    record of the memory it produced.

    Attributes:
        id: Unique identifier for this episode (caller-provided UUID).
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
            review run can be correlated.
        mutations: Ordered sequence of Cartographer mutations applied during this episode.
    """

    id: uuid.UUID = Field(description="Unique episode identifier (caller-provided)")
    run_id: str = Field(
        default="",
        description=(
            "Identifier of the pipeline run that produced this episode. "
            "Shared across all agents invoked within the same review run."
        ),
    )
    timestamp: datetime = Field(
        description="UTC time the episode was recorded (caller-provided)",
    )
    task: str = Field(description="Wrapped signature name (or module class name as fallback)")
    module: str = Field(description="Wrapped dspy.Module class name")
    question: str = Field(
        description=(
            "Question/task description for this episode "
            "(passed as 'question' or derived from serialized inputs)"
        )
    )
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Named output artifacts produced by the agent (e.g. {'review': '<markdown>'})",
    )
    context_memory: ContextMemory = Field(description="Consolidated context memory snapshot")
    mutations: list[Mutation] = Field(
        default_factory=list,
        description="Ordered sequence of Cartographer mutations applied during this episode",
    )
