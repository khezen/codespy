from __future__ import annotations

import copy

import dspy

from codespy.agents.hippocampus.budget import (
    _head_tail_text,
    count_tokens,
    evict,
    format_inputs,
    format_trajectory,
)
from codespy.agents.hippocampus.context_map import ContextMap, ItemTag
from codespy.agents.hippocampus.modules.cartographer import Cartographer
from codespy.agents.hippocampus.modules.distiller import Distiller


def prepend_context_map(sig):
    return sig.prepend(
        name="context_map",
        field=dspy.InputField(
            desc="Orientation cache about the external context. Use it before redundant tool calls."
        ),
        type_=ContextMap,
    )


class Hypocampus(dspy.Module):
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
        mem = Hypocampus(agent)
        pred = mem(task="…")

        # Pure batch — no online reflection, one holistic pass at the end
        mem = Hypocampus(agent, max_reflects=0)
        for task in tasks:
            pred = mem(task=task)
        mem.end_episode()

        # Hybrid — warm up online for the first 3 calls, then holistic consolidation
        mem = Hypocampus(agent, max_reflects=3)
        for task in tasks:
            pred = mem(task=task)
        mem.end_episode()

        # Read-only (map never changes) — pure inference
        mem = Hypocampus(agent, max_reflects=0)
        pred = mem(task="…")    # no end_episode() call

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
        self._episode: list[str] = []           # per-call bounded trajectory strings
        self._episode_question: str | None = None  # derived from first buffered call

    @property
    def current_map_text(self) -> str:
        return self.cmap.render()

    def forward(self, **kwargs) -> dspy.Prediction:
        pred = self.agent(context_map=self.cmap, **kwargs)
        # Format once (stage-1 bounded); reused for both buffering and online reflect.
        traj = format_trajectory(pred, self.max_trajectory_tokens)
        self._episode.append(traj)
        if self._episode_question is None:
            self._episode_question = self._make_question(kwargs)
        # Online reflection: None = no limit (always); N = for the first N calls.
        if (self.max_reflects is None
                or len(self._episode) <= self.max_reflects):
            self._distill_and_apply(traj, self._make_question(kwargs))
        return pred

    def reflect(self, pred: dspy.Prediction, inputs: dict) -> None:
        """Run one distillation cycle over ``pred``'s trajectory and update the map.

        For manual / selective reflection outside of the automatic per-call flow.
        The trajectory is bounded by ``max_trajectory_tokens`` (stage 1).
        """
        self._distill_and_apply(
            format_trajectory(pred, self.max_trajectory_tokens),
            self._make_question(inputs),
        )

    def end_episode(self) -> None:
        """Consolidate the buffered episode into the map and clear the buffer.

        A single Distiller pass sees all buffered trajectories joined with
        ``=== Call k ===`` headers. If ``max_trajectory_tokens`` is set, the
        combined text is head+tail bounded (stage 2) after per-call bounding
        (stage 1) already applied at append time. The question is derived from
        the first buffered call. No-op if the buffer is empty.
        """
        if not self._episode:
            return
        combined = "\n\n".join(
            f"=== Call {i + 1} ===\n{t}" for i, t in enumerate(self._episode)
        )
        if self.max_trajectory_tokens is not None:
            combined = _head_tail_text(combined, self.max_trajectory_tokens)
        self._distill_and_apply(combined, self._episode_question or "")
        self._episode.clear()
        self._episode_question = None

    def reset_episode(self) -> None:
        """Discard the buffered episode without reflecting."""
        self._episode.clear()
        self._episode_question = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_question(self, inputs: dict) -> str:
        if self.question_field is not None:
            return str(inputs.get(self.question_field, ""))
        return format_inputs(inputs, self.max_input_tokens)

    def _distill_and_apply(self, trajectory: str, question: str) -> None:
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
