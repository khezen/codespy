"""Hippocampus memory module for context-aware agents."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import dspy

from codespy.agents.memory.hippocampus.budget import (
    MemoryBudget,
    _head_tail_text,
    count_tokens,
    evict,
    format_trajectory,
)
from codespy.agents.memory.hippocampus.context_memory import (
    ContextMemory,
    ObservationTag,
    Mutation,
    Operation,
    OpType,
    Topic,
    _PREFIX_TO_SECTION,
)
from codespy.agents.memory.hippocampus.episode import Episode
from codespy.agents.memory.hippocampus.distiller import Distiller
from codespy.agents.memory.hippocampus.cartographer import Cartographer

if TYPE_CHECKING:
    from codespy.agents.memory.postgres import EpisodeStore

logger = logging.getLogger(__name__)


def prepend_context_memory(sig):
    """Prepend context_memory field to signature."""
    return sig.prepend(
        name="context_memory",
        field=dspy.InputField(
            desc="Current context memory. Use it before redundant tool calls."
        ),
        type_=ContextMemory,
    )


def inject_context_memory(module: dspy.Module) -> dspy.Module:
    """Prepend context_memory input field to a dspy.Module's signatures.

    Idempotent — skips predictors that already have context_memory.
    Mutates the module in place and returns it for chaining.
    Works through ContextSafe's signature delegation.
    """
    top_sig = getattr(module, "signature", None)
    if top_sig is not None:
        module_inputs = set(top_sig.input_fields)
        if "context_memory" not in top_sig.input_fields:
            module.signature = prepend_context_memory(top_sig)
        for _, pred in module.named_predictors():
            if (
                set(pred.signature.input_fields) & module_inputs
                and "context_memory" not in pred.signature.input_fields
            ):
                pred.signature = prepend_context_memory(pred.signature)
    else:
        for _, pred in module.named_predictors():
            if "context_memory" not in pred.signature.input_fields:
                pred.signature = prepend_context_memory(pred.signature)
    return module


class Hippocampus:
    """Memory component that evolves via LLM-driven reflection.

    The context memory is provided to agents so they start each run
    with accumulated orientation knowledge (structure, entities, constants) about
    the external context. After calls, the Distiller extracts transferable
    understanding and the Cartographer edits the memory — "caching understanding,
    not answers."

    ## Lifecycle

    Typical usage pattern::

        # Construct Hippocampus with task name and optional memory budget
        hippo = Hippocampus(task_name="...")

        # Inject context memory into agent
        inject_context_memory(agent)

        # Run agent with context_memory input
        pred = agent(context_memory=hippo.context_memory, task="…")

        # Observe the result (buffers trajectory)
        hippo.observe(pred)

        # Optional: bind topics before end_episode
        hippo.bind_topics(topics, topic_ids)

        # End episode (consolidates and persists)
        hippo.end_episode(store, artifacts={"key": "value"})

        # Async variant (for callers running inside an event loop)
        pred = await agent.acall(context_memory=hippo.context_memory, task="…")
        await hippo.aobserve(pred)
        await hippo.aend_episode(store, artifacts={"key": "value"})

    ## Trajectory bounding (two-stage)

    When ``budget.compact_trajectory`` is ``True`` and
    ``budget.max_trajectory_tokens`` is set:

    - **Stage 1** (per observation) — each trajectory is head+tail bounded at ``format_trajectory``
      time. This keeps the buffer lightweight.
    - **Stage 2** (``end_episode``) — the joined episode is head+tail bounded again, so
      the combined result is guaranteed to fit the budget even if many observations are buffered.

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
        task_name: str,
        budget: MemoryBudget | None = None,
        question: str | None = None,
        run_id: str | None = None,
        initial_memory: ContextMemory | None = None,
        topics: list[Topic] | None = None,
    ):
        """
        Args:
            budget: The four token budgets bounding memory, as a
                :class:`MemoryBudget`. Defaults to ``MemoryBudget()`` — see that
                class for per-field guidance. Resolve one from configuration with
                ``Settings.get_memory_budget(signature_name)``.
            question: Pre-computed question string for the reflection "question".
                If set, this string is used directly as the Distiller question.
                If None, uses empty string (callers typically pass question at init).
            task_name: Identity recorded in ``Episode.task`` and used in the episode
                filename. Pass the signature's snake_case name (``"doc"``,
                ``"code_review"``, …) — the same key that drives config, LM
                selection and cost attribution — so the episode path lines up with
                the rest of the system. Required parameter.
            run_id: Identifier of the pipeline run this agent belongs to. Passed
                down by the orchestrating ``ReviewPipeline`` so every module
                invoked within the same review run shares the same identifier,
                used as the ``<uuid>`` prefix in the episode filename
                (``<run_id>-<task>.json``) and recorded on ``Episode.run_id``.
                If ``None`` (standalone usage), a random UUID is generated.
            initial_memory: Optional context memory to seed the agent with. When
                provided, the agent starts with this memory instead of an empty one,
                inheriting accumulated understanding from upstream pipeline stages.
            topics: Optional list of Topic objects to register in the context memory.
                Topic IDs are auto-assigned to all new observations created during this episode.
                Used for scope-aware memory organization.
        """
        self.distill = Distiller()
        self.cartograph = Cartographer()
        self.budget = budget or MemoryBudget()
        self.question = question
        self.cmem = initial_memory.model_copy(deep=True) if initial_memory else ContextMemory()
        self._topic_ids: list[str] = []
        if topics:
            existing_ids = {t.id for t in self.cmem.topics}
            for topic in topics:
                self._topic_ids.append(topic.id)
                if topic.id not in existing_ids:
                    self.cmem.topics.append(topic)
                    existing_ids.add(topic.id)
        self.scores: dict[str, int] = {}
        # Buffer of per-observation bounded trajectory strings, cleared after end_episode().
        self._episode_trajectories: list[str] = []
        # Question derived from the latest buffered observation; used as Distiller consolidation input.
        self._episode_question: str | None = None

        # Identity for Episode metadata.
        self._task_name = task_name
        self._module_name = task_name  # Default to task_name (more meaningful than "ContextSafe")
        # Identifier of the pipeline run this agent belongs to (see run_id arg
        # above). Falls back to a random UUID for standalone usage where no
        # orchestrator provides one.
        self._run_id: str = run_id or uuid.uuid4().hex
        # The most recent consolidated Episode; set by end_episode(), None until then.
        self.episode: Episode | None = None
        # Accumulated mutations across _distill() calls within the current episode.
        self._mutations: list[Mutation] = []
        # Step counter incremented per _distill() call for mutation grouping.
        self._distill_step: int = 0

    @property
    def context_memory(self) -> ContextMemory:
        """Read-only access to the current context memory.

        Returns the live reference (mutations through it like ``bind_topics`` work).
        Assignment is blocked (read-only property).
        """
        return self.cmem

    def observe(self, result: dspy.Prediction | str, *, question: str | None = None) -> None:
        """Feed an observation (trajectory) to Hippocampus for buffering.

        The trajectory is buffered and will be consolidated at ``end_episode()``.

        Args:
            result: Either a ``dspy.Prediction`` (stage-1 trajectory bounding applies)
                or a raw ``str`` (no bounding, caller controls input).
            question: Optional per-observation question override. If not provided,
                uses ``self.question`` or empty string.
        """
        # Extract trajectory
        if isinstance(result, str):
            traj = result  # raw string: no stage-1 bounding (caller controls input)
        else:
            max_traj = self.budget.max_trajectory_tokens if self.budget.compact_trajectory else None
            traj = format_trajectory(result, max_traj)  # stage-1 bounding for Predictions

        # Buffer
        self._episode_trajectories.append(traj)
        self._episode_question = question or self.question or ""

    async def aobserve(self, result: dspy.Prediction | str, *, question: str | None = None) -> None:
        """Async counterpart of :meth:`observe`.

        Offloads the observation processing to a thread so it never blocks
        the caller's event loop.
        """
        await asyncio.to_thread(self.observe, result, question=question)

    def bind_topics(self, topics: list[Topic], topic_ids: list[str]) -> None:
        """Bind topics to the context memory for episode persistence.

        Used by the scope agent post-call to bind scope-specific topics.
        Replaces the previous pattern of direct ``_topic_ids`` access and
        ``cmem.bind_topics()`` calls.

        Args:
            topics: List of Topic objects to register.
            topic_ids: List of topic IDs to stamp on new observations.
        """
        self._topic_ids = topic_ids
        self.cmem.bind_topics(topics, topic_ids)

    def _consolidate(self) -> str | None:
        """Join buffered trajectories (stage-2 bounded) and distill+apply once.

        Returns the combined trajectory text used for consolidation, or
        ``None`` if the buffer is empty (no-op).
        """
        if not self._episode_trajectories:
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
            id=uuid.uuid4(),
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
        self._mutations.clear()
        self._distill_step = 0

    def end_episode(
        self,
        store: EpisodeStore | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> None:
        """Consolidate the buffered trajectories into the memory and record an Episode snapshot.

        A single Distiller pass sees all buffered trajectories joined with
        ``=== Call k ===`` headers. If ``budget.max_trajectory_tokens`` is set, the
        combined text is head+tail bounded (stage 2) after per-observation bounding
        (stage 1) already applied at append time. The question is derived from
        the first buffered observation. No-op if the buffer is empty.

        After consolidation ``self.episode`` is set to a new :class:`Episode`
        containing the task/module identity and a deep-copy snapshot of the
        updated context memory.

        If ``store`` is provided, the episode is persisted via ``store.save_episode()``.

        Args:
            store: Optional ``EpisodeStore`` to persist the episode after consolidation.
            artifacts: Named output artifacts to attach to the recorded
                episode (e.g. ``{"review": "<markdown>"}``). Agent-agnostic —
                any caller can attach whatever markdown/text output it
                produced under a key of its choosing.

        Raises:
            OSError: If persistence is requested and the write fails.
        """
        combined = self._consolidate()
        if combined is None:
            return
        self._finalize_episode(artifacts)
        if store is not None:
            store.save_episode(self.episode)

    async def aend_episode(
        self,
        store: EpisodeStore | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> None:
        """Async counterpart of :meth:`end_episode`.

        The (synchronous) Distiller/Cartographer consolidation pass and the
        storage write are both offloaded to a thread so they never block the
        caller's event loop.

        Args:
            store: Optional ``EpisodeStore`` to persist the episode after consolidation.
            artifacts: Named output artifacts to attach to the recorded
                episode (e.g. ``{"review": "<markdown>"}``).
        """
        combined = await asyncio.to_thread(self._consolidate)
        if combined is None:
            return
        await asyncio.to_thread(self._finalize_episode, artifacts)
        if store is not None:
            await asyncio.to_thread(store.save_episode, self.episode)

    def reset_episode(self) -> None:
        """Discard the buffered trajectories without reflecting."""
        self._episode_trajectories.clear()
        self._episode_question = None
        self._mutations.clear()
        self._distill_step = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record_mutations(
        self, ops: list[Operation], new_ids: list[str], pre_memory: ContextMemory
    ) -> list[Mutation]:
        """Build Mutation records from operations and the new IDs generated by apply().

        For DELETE/REPLACE, looks up pre-mutation state (section and previous_content).
        For ADD, back-fills observation_ids from new_ids in order.

        Args:
            ops: Cartographer operations (ADD/DELETE/REPLACE).
            new_ids: IDs of observations created by apply() in the same order as ADD ops.
            pre_memory: Context memory state before apply() — used to look up
                previous content for DELETE/REPLACE.

        Returns:
            List of Mutation records for this step.
        """
        mutations: list[Mutation] = []
        add_mutations: list[Mutation] = []
        for op in ops:
            if op.type == OpType.DELETE and op.observation_id:
                found = pre_memory.find_observation(op.observation_id)
                if found:
                    section, old_obs = found
                    mutations.append(
                        Mutation(
                            step=self._distill_step,
                            type=OpType.DELETE,
                            observation_id=op.observation_id,
                            section=section,
                            content=None,
                            previous_content=old_obs.content,
                            topic_ids=old_obs.topic_ids,
                        )
                    )
            elif op.type == OpType.REPLACE and op.observation_id and op.content:
                found = pre_memory.find_observation(op.observation_id)
                if found:
                    section, old_obs = found
                    mutations.append(
                        Mutation(
                            step=self._distill_step,
                            type=OpType.REPLACE,
                            observation_id=op.observation_id,
                            section=section,
                            content=op.content,
                            previous_content=old_obs.content,
                            topic_ids=old_obs.topic_ids,
                        )
                    )
                else:
                    # Fallback REPLACE→ADD: mirrors apply()'s fallback
                    # so add_mutations stays aligned with new_ids
                    prefix = op.observation_id.split("-", 1)[0] if "-" in op.observation_id else ""
                    section_name = _PREFIX_TO_SECTION.get(prefix)
                    if section_name:
                        mut = Mutation(
                            step=self._distill_step,
                            type=OpType.ADD,
                            observation_id="",  # back-filled from new_ids
                            section=section_name,
                            content=op.content,
                            previous_content=None,
                            topic_ids=list(self._topic_ids),
                        )
                        mutations.append(mut)
                        add_mutations.append(mut)
                    # else: topic-ID — apply() already skipped, nothing to record
            elif op.type == OpType.ADD and op.section and op.content:
                mut = Mutation(
                    step=self._distill_step,
                    type=OpType.ADD,
                    observation_id="",
                    section=op.section,
                    content=op.content,
                    previous_content=None,
                    topic_ids=list(self._topic_ids),
                )
                mutations.append(mut)
                add_mutations.append(mut)
        # Back-fill ADD mutation observation_ids from new_ids
        for mut, new_id in zip(add_mutations, new_ids, strict=True):
            mut.observation_id = new_id
        return mutations

    def _update_observation_scores(self, tags: dict[str, ObservationTag]) -> None:
        """Adjust observation scores based on Distiller-assigned tags.

        HELPFUL: +1, HARMFUL/STALE: -1, NEUTRAL: ensure entry exists (default 0).
        """
        for bid, tag in tags.items():
            if tag == ObservationTag.HELPFUL:
                self.scores[bid] = self.scores.get(bid, 0) + 1
            elif tag in (ObservationTag.HARMFUL, ObservationTag.STALE):
                self.scores[bid] = self.scores.get(bid, 0) - 1
            else:
                self.scores.setdefault(bid, 0)

    def _distill(self, trajectory: str, question: str) -> None:
        distilled = self.distill(
            trajectory=trajectory,
            context_memory=self.cmem,
            question=question,
            max_context_item_tokens=self.budget.max_context_item_tokens,
        )

        known = self.cmem.ids()
        tags = {k: v for k, v in (distilled.observation_tags or {}).items() if k in known}
        self._update_observation_scores(tags)

        edits = self.cartograph(
            diagnosis=distilled.diagnosis,
            observation_tags=tags,
            cache_candidates=list(distilled.cache_candidates or []),
            current_map=self.cmem,
            question=question,
            # The Cartographer's input field keeps the generic name: it is prompt
            # text, already scoped by its description, and pairs with current_tokens.
            token_budget=self.budget.max_context_memory_tokens,
            current_tokens=count_tokens(self.cmem.model_dump_json()),
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
