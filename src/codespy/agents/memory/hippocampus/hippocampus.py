from __future__ import annotations

import asyncio
import copy
import uuid
from datetime import datetime

import dspy

from codespy.agents.memory.hippocampus.budget import (
    _head_tail_text,
    count_tokens,
    evict,
    format_inputs,
    format_trajectory,
)
from codespy.agents.memory.hippocampus.context_map import ContextMap, ItemTag
from codespy.agents.memory.hippocampus.episode import Episode
from codespy.agents.memory.hippocampus.episode import load_episode as _load_episode
from codespy.agents.memory.hippocampus.episode import save_episode as _save_episode
from codespy.agents.memory.hippocampus.modules.cartographer import Cartographer
from codespy.agents.memory.hippocampus.modules.distiller import Distiller
from codespy.tools.storage.base import Storage


def prepend_context_map(sig):
    return sig.prepend(
        name="context_map",
        field=dspy.InputField(
            desc="Orientation cache about the external context. Use it before redundant tool calls."
        ),
        type_=ContextMap,
    )


class Hippocampus(dspy.Module):
    """Wraps a dspy.Module with a context map that evolves via LLM-driven reflection.

    The context map is prepended to every agent call so the agent starts each run
    with accumulated orientation knowledge (structure, entities, constants) about
    the external context. After calls, the Distiller extracts transferable
    understanding and the Cartographer edits the map — "caching understanding,
    not answers."

    ## Two independent controls

    Reflection behaviour is governed by two orthogonal knobs:

    1. **``max_reflects``** — maximum number of forward() calls that also reflect *online*
       (in-episode warm-up). ``None`` (default) = no limit, reflect after every call.
       ``0`` = never reflect online. ``N`` = reflect for the first N calls, buffer-only
       thereafter.

    2. **Calling ``end_episode()``** (or not) — whether to consolidate the buffered
       episode into the map at the end. Every call is *always* buffered so
       ``end_episode()`` is available regardless of the online setting.

    Common patterns::

        # Classic per-call (default) — reflect after every call, no end consolidation
        mem = Hippocampus(agent)
        pred = mem(task="…")

        # Pure batch — no online reflection, one holistic pass at the end
        mem = Hippocampus(agent, max_reflects=0)
        for task in tasks:
            pred = mem(task=task)
        mem.end_episode()

        # Hybrid — warm up online for the first 3 calls, then holistic consolidation
        mem = Hippocampus(agent, max_reflects=3)
        for task in tasks:
            pred = mem(task=task)
        mem.end_episode()

        # Read-only (map never changes) — pure inference
        mem = Hippocampus(agent, max_reflects=0)
        pred = mem(task="…")    # no end_episode() call

        # Async variants (for callers running inside an event loop, e.g.
        # reviewer modules using `await agent.acall(...)`)
        mem = Hippocampus(agent, max_reflects=0)
        pred = await mem.acall(task="…")
        await mem.aend_episode(store, dir)

    ## Trajectory bounding (two-stage)

    When ``max_trajectory_tokens`` is set:

    - **Stage 1** (per call) — each trajectory is head+tail bounded at ``format_trajectory``
      time. This keeps the buffer lightweight.
    - **Stage 2** (``end_episode``) — the joined episode is head+tail bounded again, so
      the combined result is guaranteed to fit the budget even if many calls are buffered.

    With ``max_trajectory_tokens=None`` (default) both stages are no-ops and the Distiller
    receives the full trajectory.
    """

    def __init__(
        self,
        module: dspy.Module,
        token_budget: int = 1024,
        max_trajectory_tokens: int | None = None,
        max_input_tokens: int | None = None,
        max_reflects: int | None = None,
        question_field: str | None = None,
    ):
        """
        Args:
            module: Any dspy.Module (ReAct, RLM, Predict, …) to wrap.
            token_budget: Maximum tokens kept in the context map.
            max_trajectory_tokens: Token budget for trajectories fed to the Distiller.
                None (default) = full trajectory, Distiller does all compression —
                recommended for most cases. When set, use ≤ ~50 % of the Distiller
                model's context window (e.g. ~8192 for 128k, ~4096 for smaller).
                Applied per call (stage 1) and again over the combined episode in
                end_episode() (stage 2). Step-aware head+tail bounding (60 % head /
                40 % tail) preserves both setup and conclusions.
            max_input_tokens: Token budget for serialized inputs used as the Distiller
                "question" (fallback path when question_field is None). None (default)
                = unbounded. Set ~1024 only when a large input field (e.g. an RLM
                document dump) is serialized; keep it well below max_trajectory_tokens.
                Ignored when question_field is set.
            max_reflects: Maximum number of forward() calls that also reflect online.
                None (default): no limit — reflect after every call (classic online learning).
                0: never reflect online — pure buffering until end_episode().
                N: reflect online for the first N calls, buffer-only afterwards.
                Every call is always buffered regardless of this setting, so
                end_episode() is always available.
            question_field: Name of the input field carrying the task description.
                If set, only that field is used as the Distiller "question".
                If None, all input fields are serialized (bounded by max_input_tokens).
                Set this when one field cleanly captures user intent.
        """
        super().__init__()

        module = copy.deepcopy(module)

        # Prepend context_map only to predictors that receive the module's own
        # input fields.
        top_sig = getattr(module, "signature", None)
        if top_sig is not None:
            module_inputs = set(top_sig.input_fields)
            module.signature = prepend_context_map(top_sig)
            for _, pred in module.named_predictors():
                if set(pred.signature.input_fields) & module_inputs:
                    pred.signature = prepend_context_map(pred.signature)
        else:
            for _, pred in module.named_predictors():
                pred.signature = prepend_context_map(pred.signature)

        self.agent = module
        self.distill = Distiller()
        self.cartograph = Cartographer()
        self.token_budget = token_budget
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_input_tokens = max_input_tokens
        self.max_reflects = max_reflects
        self.question_field = question_field
        self.cmap = ContextMap()
        self.scores: dict[str, int] = {}
        # Buffer of per-call bounded trajectory strings, cleared after end_episode().
        self._episode_trajectories: list[str] = []
        # Count of buffered trajectories already distilled online (via max_reflects).
        # Used to skip a redundant consolidation distill in the single-call case,
        # and to detect "everything was reflected online" so end_episode() still
        # persists a snapshot even when nothing remains to consolidate.
        self._reflected_count: int = 0
        # Question derived from the first buffered call; used as Distiller input.
        self._episode_question: str | None = None

        # Identity of the wrapped module/signature for Episode metadata.
        self._task_name: str = (
            top_sig.__name__ if top_sig is not None else type(module).__name__
        )
        self._module_name: str = type(module).__name__
        # The most recent consolidated Episode; set by end_episode(), None until then.
        self.episode: Episode | None = None

    @property
    def current_map_text(self) -> str:
        return self.cmap.render()

    def forward(self, **kwargs) -> dspy.Prediction:
        pred = self.agent(context_map=self.cmap, **kwargs)
        self._buffer_and_distill(pred, kwargs)
        return pred

    async def aforward(self, **kwargs) -> dspy.Prediction:
        """Async counterpart of :meth:`forward`.

        Awaits the wrapped agent's ``acall`` instead of invoking it
        synchronously — required when the caller is already inside a running
        event loop (e.g. reviewer modules using ``await agent.acall(...)``).
        The Distiller/Cartographer reflection pass is still synchronous under
        the hood but is offloaded to a thread so it never blocks the loop.
        """
        pred = await self.agent.acall(context_map=self.cmap, **kwargs)
        await asyncio.to_thread(self._buffer_and_distill, pred, kwargs)
        return pred

    def _buffer_and_distill(self, pred: dspy.Prediction, kwargs: dict) -> None:
        """Shared post-call work for both ``forward`` and ``aforward``.

        Buffers the (stage-1 bounded) trajectory and, depending on
        ``max_reflects``, runs an online distill+apply pass immediately.
        """
        traj = format_trajectory(pred, self.max_trajectory_tokens)
        self._episode_trajectories.append(traj)
        if self._episode_question is None:
            self._episode_question = self._make_question(kwargs)
        # Online reflection: None = no limit (always); N = for the first N calls.
        if (self.max_reflects is None
                or len(self._episode_trajectories) <= self.max_reflects):
            self._distill(traj, self._make_question(kwargs))
            self._reflected_count += 1

    def _consolidate(self) -> str | None:
        """Join buffered trajectories (stage-2 bounded) and distill+apply once.

        Returns the combined trajectory text used for consolidation, or
        ``None`` if the buffer is empty (no-op).
        """
        skip_double_distill = len(self._episode_trajectories)==1 and self._reflected_count>0
        if not self._episode_trajectories or skip_double_distill:
            return None
        combined = "\n\n".join(
            f"=== Call {i + 1} ===\n{t}"
            for i, t in enumerate(self._episode_trajectories)
        )
        if self.max_trajectory_tokens is not None:
            combined = _head_tail_text(combined, self.max_trajectory_tokens)
        self._distill(combined, self._episode_question or "")
        return combined

    def _finalize_episode(self) -> None:
        """Record the consolidated Episode snapshot and clear the buffer."""
        self.episode = Episode(
            task=self._task_name,
            module=self._module_name,
            context_map=self.cmap.model_copy(deep=True),
            timestamp=datetime.utcnow(),
        )
        self._episode_trajectories.clear()
        self._episode_question = None
        self._reflected_count = 0

    def _episode_file_path(self, dir: str) -> str:
        """Build the full episode file path from a directory.

        Prepends the ``episodes`` root and appends a hidden ``.codespy``
        folder holding the episode file, named after the wrapped task and a
        random UUID: ``episodes/<dir>/.codespy/<task>-<uuid>.json``.

        Args:
            dir: Directory identifying where this episode belongs (e.g. a
                scope's ``/{repo}/{subroot}/`` path).
        """
        file_id = uuid.uuid4().hex
        trimmed = dir.strip("/")
        return f"episodes/{trimmed}/.codespy/{self._task_name}-{file_id}.json"

    def end_episode(
        self,
        store: Storage | None = None,
        dir: str | None = None,
    ) -> None:
        """Consolidate the buffered trajectories into the map and record an Episode snapshot.

        A single Distiller pass sees all buffered trajectories joined with
        ``=== Call k ===`` headers. If ``max_trajectory_tokens`` is set, the
        combined text is head+tail bounded (stage 2) after per-call bounding
        (stage 1) already applied at append time. The question is derived from
        the first buffered call. No-op if the buffer is empty.

        After consolidation ``self.episode`` is set to a new :class:`Episode`
        containing the task/module identity and a deep-copy snapshot of the
        updated context map.

        If both ``store`` and ``dir`` are provided the episode is persisted
        via ``save_episode()`` after consolidation, at
        ``episodes/<dir>/.codespy/<task>-<uuid>.json``. ``store`` may be a
        ``FileSystem`` or an ``S3Client`` instance.

        Args:
            store: Optional ``Storage`` backend to persist the episode after
                consolidation (``FileSystem`` or ``S3Client``).
            dir: Directory identifying where this episode belongs (e.g. a
                scope's path). Required when ``store`` is set.

        Raises:
            OSError: If persistence is requested and the write fails.
        """
        nothing_to_persist = self._consolidate() is None and self._reflected_count==0
        if nothing_to_persist:
            return
        self._finalize_episode()
        if store is not None and dir is not None:
            _save_episode(store, self._episode_file_path(dir), self.episode)

    async def aend_episode(
        self,
        store: Storage | None = None,
        dir: str | None = None,
    ) -> None:
        """Async counterpart of :meth:`end_episode`.

        The (synchronous) Distiller/Cartographer consolidation pass and the
        storage write are both offloaded to a thread so they never block the
        caller's event loop.
        """
        combined = await asyncio.to_thread(self._consolidate)
        nothing_to_persist = combined is None and self._reflected_count == 0
        if nothing_to_persist:
            return
        await asyncio.to_thread(self._finalize_episode)
        if store is not None and dir is not None:
            path = self._episode_file_path(dir)
            await asyncio.to_thread(_save_episode, store, path, self.episode)

    def save_episode(self, store: Storage, path: str) -> None:
        """Persist the current episode to ``path`` via ``store``.

        Args:
            store: Storage backend (``FileSystem`` or ``S3Client``).
            path: Destination path within the store.

        Raises:
            ValueError: If no episode has been consolidated yet (call
                ``end_episode()`` first).
            OSError: If the write fails.
        """
        if self.episode is None:
            raise ValueError(
                "No episode to save — call end_episode() to consolidate first."
            )
        _save_episode(store, path, self.episode)

    def load_episode(self, store: Storage, path: str) -> None:
        """Replace the current state with an episode loaded from ``path`` via ``store``.

        Restores both ``self.episode`` and the live context map
        (``self.cmap = episode.context_map``) so the agent resumes from the
        persisted state. Also resets ``scores`` and clears the trajectory
        buffer since they belong to the previous state.

        Args:
            store: Storage backend (``FileSystem`` or ``S3Client``).
            path: Source path within the store.

        Raises:
            FileNotFoundError: If the path does not exist.
            OSError: If reading or parsing fails.
        """
        ep = _load_episode(store, path)
        self.episode = ep
        self.cmap = ep.context_map
        self.scores = {}
        self._episode_trajectories.clear()
        self._episode_question = None
        self._reflected_count = 0

    def reset_episode(self) -> None:
        """Discard the buffered trajectories without reflecting."""
        self._episode_trajectories.clear()
        self._episode_question = None
        self._reflected_count = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_question(self, inputs: dict) -> str:
        if self.question_field is not None:
            return str(inputs.get(self.question_field, ""))
        return format_inputs(inputs, self.max_input_tokens)

    def _distill(self, trajectory: str, question: str) -> None:
        distilled = self.distill(
            trajectory=trajectory,
            context_map=self.cmap,
            question=question,
        )

        known = self.cmap.ids()
        tags = {k: v for k, v in (distilled.item_tags or {}).items() if k in known}
        for bid, tag in tags.items():
            if tag == ItemTag.HELPFUL:
                self.scores[bid] = self.scores.get(bid, 0) + 1
            elif tag in (ItemTag.HARMFUL, ItemTag.STALE):
                self.scores[bid] = self.scores.get(bid, 0) - 1
            else:
                self.scores.setdefault(bid, 0)

        edits = self.cartograph(
            diagnosis=distilled.diagnosis,
            item_tags=tags,
            cache_candidates=list(distilled.cache_candidates or []),
            current_map=self.cmap,
            question=question,
            token_budget=self.token_budget,
            current_tokens=count_tokens(self.cmap.render()),
        )
        ops = list(edits.operations or [])

        if ops:
            self.cmap, new_ids = self.cmap.apply(ops)
            for nid in new_ids:
                self.scores[nid] = self.scores.get(nid, 0) + 1

        self.cmap = evict(self.cmap, self.scores, self.token_budget)

        live = self.cmap.ids()
        self.scores = {k: v for k, v in self.scores.items() if k in live}
