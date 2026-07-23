from __future__ import annotations

import copy
from typing import Literal

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

    ## Update modes

    ``update_mode`` controls *when* reflection (Distiller → Cartographer → evict)
    runs:

    - **``per_call``** (default) — reflect after every ``forward()``. The map
      improves *within* the same task: call N+1 benefits from call N's insights.
      Higher LLM cost (one reflection cycle per call).

    - **``episode``** — ``forward()`` only *uses* the map and *buffers* each call's
      trajectory. Reflection runs once when you call ``end_episode()``, consolidating
      the whole episode into a single Distiller pass. Use this when you want a cheap,
      holistic end-of-task reflection rather than incremental updates. The map is
      static during the task (read-only) and updated only for *future* tasks.

    ## Examples

    Online (per-call, default)::

        mem = Hypocampus(agent)
        pred = mem(task="summarise section 3")

    Batch reflection (episode)::

        mem = Hypocampus(agent, update_mode="episode")
        for task in tasks:
            pred = mem(task=task)        # uses map; trajectories buffered
        mem.end_episode()               # single end-of-episode reflection

    """

    def __init__(
        self,
        module: dspy.Module,
        token_budget: int = 1024,
        max_trajectory_tokens: int | None = None,
        max_input_tokens: int | None = None,
        update_mode: Literal["per_call", "episode"] = "per_call",
        question_field: str | None = None,
    ):
        """
        Args:
            module: Any dspy.Module (ReAct, RLM, Predict, …) to wrap.
            token_budget: Maximum tokens kept in the context map.
            max_trajectory_tokens: Token budget for the trajectory fed to the Distiller.
                None (default) passes the full trajectory so the Distiller does all
                compression — recommended for most cases. Set a limit only as a safety
                net against runaway trajectories that would exceed the Distiller model's
                context window. When set, tie it to that window: keep it at roughly
                ≤ 50 % of the context window to leave room for the Distiller prompt,
                the current context map, the question, and the output (e.g. ~8192 for
                a 128k model, ~4096 for smaller windows). It should be the dominant
                share of the Distiller input. Step-aware head+tail bounding
                (60 % head / 40 % tail) preserves both setup and conclusions.
            max_input_tokens: Token budget for the serialized inputs used as the
                Distiller "question" (only active when question_field is None).
                None (default) = unbounded — suitable for most cases where inputs
                are normal-sized task strings. Only set this when a large input field
                (e.g. an RLM document dump) is serialized via the fallback path; in
                that case ~1024 is a reasonable value, keeping it well below
                max_trajectory_tokens and the model's context window. Head+tail
                bounding is applied (60 % head / 40 % tail) so both the leading
                instruction and any trailing intent survive. Ignored when
                question_field is set.
            update_mode: Controls when reflection (Distiller → Cartographer → evict)
                runs. See class docstring for full details.
                - "per_call" (default): reflect after every forward().
                - "episode": buffer trajectories; call end_episode() to reflect once.
            question_field: Name of the input field that carries the task description.
                If set, only that field is passed to the Distiller as the "question".
                If None, all input fields are serialized and head+tail bounded
                (controlled by max_input_tokens). Set this explicitly whenever one
                field cleanly captures the user's intent.
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
        self.update_mode = update_mode
        self.question_field = question_field
        self.cmap = ContextMap()
        self.scores: dict[str, int] = {}
        self._episode: list[tuple[dspy.Prediction, dict]] = []

    @property
    def current_map_text(self) -> str:
        return self.cmap.render()

    def forward(self, **kwargs) -> dspy.Prediction:
        pred = self.agent(context_map=self.cmap, **kwargs)
        if self.update_mode == "per_call":
            self.reflect(pred, kwargs)
        else:  # "episode"
            self._episode.append((pred, kwargs))
        return pred

    def reflect(self, pred: dspy.Prediction, inputs: dict) -> None:
        """Run one distillation cycle over ``pred``'s trajectory and update the map.

        Called automatically in ``per_call`` mode. Call manually when you want
        fine-grained control (e.g. selectively reflect on specific calls).
        """
        question = self._make_question(inputs)
        trajectory = format_trajectory(pred, self.max_trajectory_tokens)
        self._distill_and_apply(trajectory, question)

    def end_episode(self) -> None:
        """Consolidate the buffered episode into the map and clear the buffer.

        Only meaningful in ``episode`` mode. A single Distiller pass sees the
        entire episode's trajectories concatenated (head+tail bounded by
        ``max_trajectory_tokens``). The question is derived from the first
        buffered call. No-op if the buffer is empty.
        """
        if not self._episode:
            return
        # Build combined trajectory: each call separated by a call header.
        # Individual trajectories are formatted without per-call bounding so
        # the combined head+tail pass governs the final size.
        parts = []
        for i, (pred, _) in enumerate(self._episode):
            parts.append(f"=== Call {i + 1} ===")
            parts.append(format_trajectory(pred))
        combined = "\n\n".join(parts)
        if self.max_trajectory_tokens is not None:
            combined = _head_tail_text(combined, self.max_trajectory_tokens)
        # Derive question from the first buffered call
        _, first_inputs = self._episode[0]
        question = self._make_question(first_inputs)
        self._distill_and_apply(combined, question)
        self._episode.clear()

    def reset_episode(self) -> None:
        """Discard the buffered episode without reflecting."""
        self._episode.clear()

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
