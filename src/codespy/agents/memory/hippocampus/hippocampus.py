from __future__ import annotations

import asyncio
import copy
import logging
import uuid
from datetime import UTC, datetime

import dspy

from codespy.agents.memory.hippocampus.budget import (
    MemoryBudget,
    _head_tail_text,
    count_tokens,
    evict,
    format_inputs,
    format_trajectory,
)
from codespy.agents.memory.hippocampus.context_memory import (
    ContextMemory,
    ItemTag,
    Mutation,
    Operation,
    OpType,
)
from codespy.agents.memory.hippocampus.episode import Episode
from codespy.agents.memory.hippocampus.episode import load_episode as _load_episode
from codespy.agents.memory.hippocampus.episode import save_episode as _save_episode
from codespy.agents.memory.hippocampus.modules.cartographer import Cartographer
from codespy.agents.memory.hippocampus.modules.distiller import Distiller
from codespy.tools.storage.base import Storage

logger = logging.getLogger(__name__)


def prepend_context_memory(sig):
    """Prepend context_memory field to signature.

    The context_memory is passed as a pre-rendered string to avoid
    Pydantic serialization in the LLM prompt.
    """
    return sig.prepend(
        name="context_memory",
        field=dspy.InputField(
            desc=(
                "Current context memory (topic-grouped, with item IDs and sections). "
                "Use it before redundant tool calls."
            )
        ),
        type_=str,
    )


class Hippocampus(dspy.Module):
    """Wraps a dspy.Module with a context memory that evolves via LLM-driven reflection.

    The context memory is prepended to every agent call so the agent starts each run
    with accumulated orientation knowledge (structure, entities, constants) about
    the external context. After calls, the Distiller extracts transferable
    understanding and the Cartographer edits the memory — "caching understanding,
    not answers."

    ## Two independent controls

    Reflection behaviour is governed by two orthogonal knobs:

    1. **``max_reflects``** — maximum number of forward() calls that also reflect *online*
       (in-episode warm-up). ``None`` (default) = no limit, reflect after every call.
       ``0`` = never reflect online. ``N`` = reflect for the first N calls, buffer-only
       thereafter.

    2. **Calling ``end_episode()``** (or not) — whether to consolidate the buffered
       episode into the memory at the end. Every call is *always* buffered so
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

        # Read-only (memory never changes) — pure inference
        mem = Hippocampus(agent, max_reflects=0)
        pred = mem(task="…")    # no end_episode() call

        # Async variants (for callers running inside an event loop, e.g.
        # reviewer modules using `await agent.acall(...)`)
        mem = Hippocampus(agent, max_reflects=0)
        pred = await mem.acall(task="…")
        await mem.aend_episode(store, dir)

    ## Trajectory bounding (two-stage)

    When ``budget.compact_trajectory`` is ``True`` and
    ``budget.max_trajectory_tokens`` is set:

    - **Stage 1** (per call) — each trajectory is head+tail bounded at ``format_trajectory``
      time. This keeps the buffer lightweight.
    - **Stage 2** (``end_episode``) — the joined episode is head+tail bounded again, so
      the combined result is guaranteed to fit the budget even if many calls are buffered.

    With ``budget.compact_trajectory=False`` both stages are skipped and the
    Distiller receives the full trajectory. The ContextSafe wrapper on the
    Distiller provides RLM fallback if the input exceeds the model's context
    window.

    With ``budget.max_trajectory_tokens=None`` both stages are no-ops
    regardless of ``compact_trajectory``.

    ## Token budgets

    The four token budgets are grouped into :class:`MemoryBudget`; see that class
    for what each one bounds and how to tune it.
    """

    def __init__(
        self,
        module: dspy.Module,
        budget: MemoryBudget | None = None,
        max_reflects: int | None = None,
        question: str | None = None,
        task_name: str | None = None,
        run_id: str | None = None,
        initial_memory: ContextMemory | None = None,
        topic_ids: list[str] | None = None,
    ):
        """
        Args:
            module: Any dspy.Module (ReAct, RLM, Predict, …) to wrap.
            budget: The four token budgets bounding memory, as a
                :class:`MemoryBudget`. Defaults to ``MemoryBudget()`` — see that
                class for per-field guidance. Resolve one from configuration with
                ``Settings.get_memory_budget(signature_name)``.
            max_reflects: Maximum number of forward() calls that also reflect online.
                None (default): no limit — reflect after every call (classic online learning).
                0: never reflect online — pure buffering until end_episode().
                N: reflect online for the first N calls, buffer-only afterwards.
                Every call is always buffered regardless of this setting, so
                end_episode() is always available.
            question: Pre-computed question string for the reflection "question".
                If set, this string is used directly as the Distiller question.
                If None, all input fields are serialized (bounded by
                ``budget.max_question_tokens``).
                Set this when one field cleanly captures user intent.
            task_name: Identity recorded in ``Episode.task`` and used in the episode
                filename. Pass the signature's snake_case name (``"doc"``,
                ``"code_review"``, …) — the same key that drives config, LM
                selection and cost attribution — so the episode path lines up with
                the rest of the system. Inference is a last resort: only
                ``dspy.ReAct``-style modules expose ``.signature``,
                ``dspy.ChainOfThought`` does not, so the fallback would yield a
                meaningless (and collision-prone) ``"ChainOfThought"``.
            run_id: Identifier of the pipeline run this agent belongs to. Passed
                down by the orchestrating ``ReviewPipeline`` so every module
                invoked within the same review run shares the same identifier,
                used as the ``<uuid>`` prefix in the episode filename
                (``<run_id>-<task>.json``) and recorded on ``Episode.run_id``.
                If ``None`` (standalone usage), a random UUID is generated.
            initial_memory: Optional context memory to seed the agent with. When
                provided, the agent starts with this memory instead of an empty one,
                inheriting accumulated understanding from upstream pipeline stages.
            topic_ids: Optional list of topic IDs to auto-assign to all new items
                created during this episode. Used for scope-aware memory organization.
        """
        super().__init__()

        module = copy.deepcopy(module)

        # Prepend context_memory only to predictors that receive the module's own
        # input fields.
        top_sig = getattr(module, "signature", None)
        if top_sig is not None:
            module_inputs = set(top_sig.input_fields)
            module.signature = prepend_context_memory(top_sig)
            for _, pred in module.named_predictors():
                if (
                    set(pred.signature.input_fields) & module_inputs
                    and "context_memory" not in pred.signature.input_fields
                ):
                    pred.signature = prepend_context_memory(pred.signature)
        else:
            for _, pred in module.named_predictors():
                pred.signature = prepend_context_memory(pred.signature)

        self.agent = module
        self.distill = Distiller()
        self.cartograph = Cartographer()
        self.budget = budget or MemoryBudget()
        self.max_reflects = max_reflects
        self.question = question
        self.cmem = initial_memory.model_copy(deep=True) if initial_memory else ContextMemory()
        self._topic_ids = topic_ids or []
        self.scores: dict[str, int] = {}
        # Buffer of per-call bounded trajectory strings, cleared after end_episode().
        self._episode_trajectories: list[str] = []
        # Count of buffered trajectories already distilled online (via max_reflects).
        # Used to skip a redundant consolidation distill in the single-call case,
        # and to detect "everything was reflected online" so end_episode() still
        # persists a snapshot even when nothing remains to consolidate.
        self._reflected_count: int = 0
        # Question derived from the latest buffered call; used as Distiller consolidation input.
        self._episode_question: str | None = None

        # Identity of the wrapped module/signature for Episode metadata. An explicit
        # task_name wins: inference only works for modules exposing .signature.
        self._task_name: str
        if task_name:
            self._task_name = task_name
        elif top_sig is not None:
            self._task_name = top_sig.__name__
        else:
            fallback = type(module).__name__
            logger.warning(
                "Hippocampus: task_name not provided and module %r has no .signature; "
                "using collision-prone fallback %r. Pass task_name explicitly.",
                module,
                fallback,
            )
            self._task_name = fallback
        self._module_name: str = type(module).__name__
        # Identifier of the pipeline run this agent belongs to (see run_id arg
        # above). Falls back to a random UUID for standalone usage where no
        # orchestrator provides one.
        self._run_id: str = run_id or uuid.uuid4().hex
        # Counter for episode filenames to avoid collisions when the same
        # signature is invoked multiple times on the same scope within one run.
        self._episode_index: int = 0
        # The most recent consolidated Episode; set by end_episode(), None until then.
        self.episode: Episode | None = None
        # Accumulated mutations across _distill() calls within the current episode.
        self._mutations: list[Mutation] = []
        # Step counter incremented per _distill() call for mutation grouping.
        self._distill_step: int = 0

    @property
    def current_memory_text(self) -> str:
        """Return the rendered context memory as text."""
        return self.cmem.render()

    def forward(self, **kwargs) -> dspy.Prediction:
        pred = self.agent(context_memory=self.cmem.render(), **kwargs)
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
        pred = await self.agent.acall(context_memory=self.cmem.render(), **kwargs)
        await asyncio.to_thread(self._buffer_and_distill, pred, kwargs)
        return pred

    def _buffer_and_distill(self, pred: dspy.Prediction, kwargs: dict) -> None:
        """Shared post-call work for both ``forward`` and ``aforward``.

        Buffers the (stage-1 bounded) trajectory and, depending on
        ``max_reflects``, runs an online distill+apply pass immediately.
        """
        max_traj = self.budget.max_trajectory_tokens if self.budget.compact_trajectory else None
        traj = format_trajectory(pred, max_traj)
        self._episode_trajectories.append(traj)
        self._episode_question = self._make_question(kwargs)
        # Online reflection: None = no limit (always); N = for the first N calls.
        if self.max_reflects is None or len(self._episode_trajectories) <= self.max_reflects:
            try:
                self._distill(traj, self._episode_question)
                self._reflected_count += 1
            except Exception:
                logger.warning(
                    "Online reflection failed for %s; trajectory buffered for end_episode().",
                    self._task_name,
                    exc_info=True,
                )

    def _consolidate(self) -> str | None:
        """Join buffered trajectories (stage-2 bounded) and distill+apply once.

        Returns the combined trajectory text used for consolidation, or
        ``None`` if the buffer is empty (no-op).
        """
        skip_double_distill = len(self._episode_trajectories) == 1 and self._reflected_count > 0
        if not self._episode_trajectories or skip_double_distill:
            return None
        combined = "\n\n".join(
            f"=== Call {i + 1} ===\n{t}" for i, t in enumerate(self._episode_trajectories)
        )
        if self.budget.compact_trajectory and self.budget.max_trajectory_tokens is not None:
            combined = _head_tail_text(combined, self.budget.max_trajectory_tokens)
        try:
            self._distill(combined, self._episode_question or "")
        except Exception:
            logger.warning(
                "Consolidation reflection failed for %s; episode saved without final distill.",
                self._task_name,
                exc_info=True,
            )
        return combined

    def _finalize_episode(self, artifacts: dict[str, str] | None = None) -> None:
        """Record the consolidated Episode snapshot and clear the buffer.

        Args:
            artifacts: Named output artifacts to attach to the recorded
                episode (e.g. ``{"review": "<markdown>"}``). Defaults to an
                empty dict when omitted.
        """
        self.episode = Episode(
            task=self._task_name,
            module=self._module_name,
            question=self._episode_question or "",
            context_memory=self.cmem.model_copy(deep=True),
            timestamp=datetime.now(UTC),
            artifacts=artifacts or {},
            run_id=self._run_id,
            mutations=self._mutations,
        )
        self._episode_trajectories.clear()
        self._episode_question = None
        self._reflected_count = 0
        self._mutations.clear()
        self._distill_step = 0

    def episode_file_path(self, dir: str, index: int = 0) -> str:
        """Build the full episode file path from a directory.

        Path format: ``orgs/<owner>/episodic/.codespy/<owner>.<repo>``
        ``[.<subroot>].<run_id>-<task>-<index>.json``  # noqa: E501

        Args:
            dir: Directory identifying where this episode belongs (e.g. a
                scope's ``/{host}/{owner}/{repo}/{subroot}/`` path or
                ``/{owner}/{repo}/{subroot}/`` without host).
            index: Episode index for this scope/task combination.
        """
        segments = [s for s in dir.strip("/").split("/") if s]
        # Strip host segment (contains a dot, e.g. github.com/gitlab.com)
        if segments and "." in segments[0]:
            segments = segments[1:]
        owner = segments[0] if segments else "unknown"
        slug = ".".join(segments)
        return (
            f"orgs/{owner}/episodic/.codespy/{slug}.{self._run_id}-{self._task_name}-{index}.json"
        )

    def end_episode(
        self,
        store: Storage | None = None,
        dir: str | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> None:
        """Consolidate the buffered trajectories into the memory and record an Episode snapshot.

        A single Distiller pass sees all buffered trajectories joined with
        ``=== Call k ===`` headers. If ``budget.max_trajectory_tokens`` is set, the
        combined text is head+tail bounded (stage 2) after per-call bounding
        (stage 1) already applied at append time. The question is derived from
        the first buffered call. No-op if the buffer is empty.

        After consolidation ``self.episode`` is set to a new :class:`Episode`
        containing the task/module identity and a deep-copy snapshot of the
        updated context memory.

        If both ``store`` and ``dir`` are provided the episode is persisted
        via ``save_episode()`` after consolidation, at
        ``orgs/<owner>/episodic/.codespy/<slug>.<run_id>-<task>-<index>.json``. ``store`` may be a
        ``FileSystem`` or an ``S3Client`` instance.

        Args:
            store: Optional ``Storage`` backend to persist the episode after
                consolidation (``FileSystem`` or ``S3Client``).
            dir: Directory identifying where this episode belongs (e.g. a
                scope's path). Required when ``store`` is set.
            artifacts: Named output artifacts to attach to the recorded
                episode (e.g. ``{"review": "<markdown>"}``). Agent-agnostic —
                any caller can attach whatever markdown/text output it
                produced under a key of its choosing.

        Raises:
            OSError: If persistence is requested and the write fails.
        """
        combined = self._consolidate()
        has_content = combined is not None or self._reflected_count > 0
        if not has_content:
            return
        self._finalize_episode(artifacts)
        if store is not None and dir is not None:
            _save_episode(store, self.episode_file_path(dir, self._episode_index), self.episode)
            self._episode_index += 1

    async def aend_episode(
        self,
        store: Storage | None = None,
        dir: str | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> None:
        """Async counterpart of :meth:`end_episode`.

        The (synchronous) Distiller/Cartographer consolidation pass and the
        storage write are both offloaded to a thread so they never block the
        caller's event loop.

        Args:
            store: Optional ``Storage`` backend to persist the episode after
                consolidation (``FileSystem`` or ``S3Client``).
            dir: Directory identifying where this episode belongs (e.g. a
                scope's path). Required when ``store`` is set.
            artifacts: Named output artifacts to attach to the recorded
                episode (e.g. ``{"review": "<markdown>"}``).
        """
        combined = await asyncio.to_thread(self._consolidate)
        has_content = combined is not None or self._reflected_count > 0
        if not has_content:
            return
        await asyncio.to_thread(self._finalize_episode, artifacts)
        if store is not None and dir is not None:
            path = self.episode_file_path(dir, self._episode_index)
            await asyncio.to_thread(_save_episode, store, path, self.episode)
            self._episode_index += 1

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
            raise ValueError("No episode to save — call end_episode() to consolidate first.")
        _save_episode(store, path, self.episode)

    def load_episode(self, store: Storage, path: str) -> None:
        """Replace the current state with an episode loaded from ``path`` via ``store``.

        Restores both ``self.episode`` and the live context memory
        (``self.cmem = episode.context_memory``) so the agent resumes from the
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
        self.cmem = ep.context_memory
        self.scores = {}
        self._episode_trajectories.clear()
        self._episode_question = None
        self._reflected_count = 0
        self._episode_index = 0
        self._mutations.clear()
        self._distill_step = 0

    def reset_episode(self) -> None:
        """Discard the buffered trajectories without reflecting."""
        self._episode_trajectories.clear()
        self._episode_question = None
        self._reflected_count = 0
        self._episode_index = 0
        self._mutations.clear()
        self._distill_step = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_question(self, inputs: dict) -> str:
        if self.question is not None:
            return self.question
        return format_inputs(inputs, self.budget.max_question_tokens)

    def _record_mutations(
        self, ops: list[Operation], new_ids: list[str], pre_memory: ContextMemory
    ) -> list[Mutation]:
        """Build Mutation records from operations and the new IDs generated by apply().

        For DELETE/REPLACE, looks up pre-mutation state (section and previous_content).
        For ADD, back-fills item_ids from new_ids in order.

        Args:
            ops: Cartographer operations (ADD/DELETE/REPLACE).
            new_ids: IDs of items created by apply() in the same order as ADD ops.
            pre_memory: Context memory state before apply() — used to look up
                previous content for DELETE/REPLACE.

        Returns:
            List of Mutation records for this step.
        """
        mutations: list[Mutation] = []
        add_mutations: list[Mutation] = []
        for op in ops:
            if op.type == OpType.DELETE and op.item_id:
                found = pre_memory.find_item(op.item_id)
                if found:
                    section, old_item = found
                    mutations.append(
                        Mutation(
                            step=self._distill_step,
                            type=OpType.DELETE,
                            item_id=op.item_id,
                            section=section,
                            content=None,
                            previous_content=old_item.content,
                            topic_ids=old_item.topic_ids,
                        )
                    )
            elif op.type == OpType.REPLACE and op.item_id and op.content:
                found = pre_memory.find_item(op.item_id)
                if found:
                    section, old_item = found
                    mutations.append(
                        Mutation(
                            step=self._distill_step,
                            type=OpType.REPLACE,
                            item_id=op.item_id,
                            section=section,
                            content=op.content,
                            previous_content=old_item.content,
                            topic_ids=old_item.topic_ids,
                        )
                    )
            elif op.type == OpType.ADD and op.section and op.content:
                mut = Mutation(
                    step=self._distill_step,
                    type=OpType.ADD,
                    item_id="",
                    section=op.section,
                    content=op.content,
                    previous_content=None,
                    topic_ids=list(self._topic_ids),
                )
                mutations.append(mut)
                add_mutations.append(mut)
        # Back-fill ADD mutation item_ids from new_ids
        for mut, new_id in zip(add_mutations, new_ids, strict=True):
            mut.item_id = new_id
        return mutations

    def _update_item_scores(self, tags: dict[str, ItemTag]) -> None:
        """Adjust item scores based on Distiller-assigned tags.

        HELPFUL: +1, HARMFUL/STALE: -1, NEUTRAL: ensure entry exists (default 0).
        """
        for bid, tag in tags.items():
            if tag == ItemTag.HELPFUL:
                self.scores[bid] = self.scores.get(bid, 0) + 1
            elif tag in (ItemTag.HARMFUL, ItemTag.STALE):
                self.scores[bid] = self.scores.get(bid, 0) - 1
            else:
                self.scores.setdefault(bid, 0)

    def _distill(self, trajectory: str, question: str) -> None:
        distilled = self.distill(
            trajectory=trajectory,
            context_memory=self.cmem.render(),
            question=question,
            max_context_item_tokens=self.budget.max_context_item_tokens,
        )

        known = self.cmem.ids()
        tags = {k: v for k, v in (distilled.item_tags or {}).items() if k in known}
        self._update_item_scores(tags)

        edits = self.cartograph(
            diagnosis=distilled.diagnosis,
            item_tags=tags,
            cache_candidates=list(distilled.cache_candidates or []),
            current_map=self.cmem.render(),
            question=question,
            # The Cartographer's input field keeps the generic name: it is prompt
            # text, already scoped by its description, and pairs with current_tokens.
            token_budget=self.budget.max_context_memory_tokens,
            current_tokens=count_tokens(self.cmem.render()),
            max_context_item_tokens=self.budget.max_context_item_tokens,
        )
        ops = list(edits.operations or [])

        if ops:
            pre_memory = self.cmem
            self.cmem, new_ids = self.cmem.apply(ops, topic_ids=self._topic_ids)
            mutations = self._record_mutations(ops, new_ids, pre_memory)
            self._mutations.extend(mutations)
            for nid in new_ids:
                self.scores[nid] = self.scores.get(nid, 0) + 1

        self._distill_step += 1
        self.cmem = evict(self.cmem, self.scores, self.budget.max_context_memory_tokens)

        live = self.cmem.ids()
        self.scores = {k: v for k, v in self.scores.items() if k in live}
