"""Episode record and persistence helpers for Hippocampus.

Background episode saves (retain + consolidation) are tracked as non-daemon threads.
The module registers an at-exit hook that runs BEFORE thread pool executors are
shut down, ensuring saves complete before the Python interpreter destroys the
executors. This prevents "cannot schedule new futures after shutdown" errors
from embedding calls during Cerebral consolidation.
"""

from __future__ import annotations

import concurrent.futures.thread  # noqa: F401 - ensures thread module loaded
import logging
import threading
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from pydantic import BaseModel, Field

from codespy.agents.memory.hippocampus.context_memory import ContextMemory, Mutation
from codespy.agents.memory.recall import RecallRecord

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

# Deferred save queue: holds saves when _defer is True
_deferred: list[tuple[Callable[[], object], str]] = []
_defer: bool = False


def submit_episode_save(target: Callable[[], object], *, name: str = "episode-save") -> None:
    """Start *target* in a non-daemon background thread and track it.

    The thread is **non-daemon**, so the Python interpreter will wait for it at
    exit. However, non-daemon alone is not enough: atexit handlers run BEFORE
    non-daemon threads are joined, and concurrent.futures registers a handler
    that shuts down thread pools. Cerebral consolidation uses those pools for
    embeddings, so if the main thread exits first, embedding calls fail with
    "cannot schedule new futures after shutdown".

    To prevent this, episode.py registers its own at-exit hook via
    threading._register_atexit(_join_episode_saves_at_exit). Hooks run in
    reverse order, so this hook runs BEFORE the executor-shutdown hook.
    Therefore, saves finish while the executors are still alive, even when
    ReviewPipeline.forward() raised or the caller never joined explicitly.

    When _defer is True (inside defer_episode_saves() context), saves are
    queued instead of started immediately. Call start_deferred_episode_saves()
    to drain the queue and start them.

    Args:
        target: A zero-arg callable that performs the episode save.
        name: Thread name (for debugging / log messages).
    """
    with _save_lock:
        if _defer:
            _deferred.append((target, name))
            return

    t = threading.Thread(target=target, name=name, daemon=False)
    with _save_lock:
        _save_threads.append(t)
    t.start()


@contextmanager
def defer_episode_saves() -> Generator[None, None, None]:
    """Context manager that defers episode saves until exit.

    Inside this context, submit_episode_save() queues saves instead of
    starting them. When the context exits, saves remain queued until
    start_deferred_episode_saves() is called.
    """
    global _defer
    with _save_lock:
        _defer = True
    try:
        yield
    finally:
        # _defer stays True - caller must call start_deferred_episode_saves()
        pass


def start_deferred_episode_saves() -> None:
    """Drain the deferred queue and start all queued saves.

    Called after the review phase completes to start the memory phase.
    Saves submitted after this call start immediately (defer is off).
    """
    global _defer, _deferred
    with _save_lock:
        _defer = False
        to_start = list(_deferred)
        _deferred.clear()

    for target, name in to_start:
        t = threading.Thread(target=target, name=name, daemon=False)
        with _save_lock:
            _save_threads.append(t)
        t.start()
    if to_start:
        logger.info("started %d deferred episode save(s)", len(to_start))


def join_episode_saves(timeout_per_thread: float | None = None) -> None:
    """Block until every tracked episode-save thread has finished.

    Called once at the end of ``ReviewPipeline.forward()`` to guarantee
    episodes are persisted before the process may exit.

    Args:
        timeout_per_thread: Max seconds to wait per thread (default None = wait
            forever). If a timeout is given and a thread exceeds it, the thread
            is abandoned with a warning. If the timeout is None, we wait
            indefinitely for all saves to complete.
    """
    while True:
        with _save_lock:
            # Take a snapshot and clear the registry for the next iteration
            threads = list(_save_threads)
            _save_threads.clear()

        if not threads:
            # No more saves in flight
            break

        # Log once if we're about to wait for any live threads
        alive_count = sum(1 for t in threads if t.is_alive())
        if alive_count:
            logger.info("waiting for %d background episode save(s)", alive_count)

        for t in threads:
            t.join(timeout=timeout_per_thread)
            if t.is_alive() and timeout_per_thread is not None:
                # Only warn when a timeout was specified
                logger.warning("Episode save thread %r did not finish in time", t.name)


def _join_episode_saves_at_exit() -> None:
    """At-exit hook: join episode saves before thread pool executors shut down.

    Runs before concurrent.futures.thread._python_exit thanks to atexit hook
    ordering (reverse registration order). This ensures Cerebral consolidations
    that rely on ThreadPoolExecutor can still submit embedding tasks.
    """
    try:
        # Start any deferred saves that were never started
        start_deferred_episode_saves()
        join_episode_saves()
    except BaseException:
        # Log at debug level so Ctrl+C during exit doesn't print a traceback
        logger.debug("Exception during episode save join at exit", exc_info=True)


# Register the at-exit hook to run BEFORE thread pool executors are shut down.
# This runs after concurrent.futures.thread is imported (at module top).
_register_atexit = getattr(threading, "_register_atexit", None)
if _register_atexit is not None:
    _register_atexit(_join_episode_saves_at_exit)


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
        recalls: Prefrontal recalls made during the agent call (pre-call load and
            ``recall_memory`` tool calls), with the exact text the agent received,
            model, token usage and cost. For monitoring only: persisted to the
            ``recalls`` table, never given to the Distiller/Cartographer and never
            retained into Cerebral.
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
    recalls: list[RecallRecord] = Field(
        default_factory=list,
        description=(
            "Prefrontal recalls of this agent call "
            "(monitoring only; never distilled or retained)"
        ),
    )
