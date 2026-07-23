from __future__ import annotations

import copy

import dspy

from codespy.agents.hippocampus.budget import (
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
    """Wraps a dspy.Module with a test-time-evolving context map.

    Prepends a `context_map: ContextMap` input field to every predictor
    inside the module, then delegates forward() to it. After each call the
    trajectory is distilled and the context map is updated.
    """

    def __init__(
        self,
        module: dspy.Module,
        token_budget: int = 1024,
        max_trajectory_tokens: int | None = None,
        max_input_tokens: int | None = None,
        freeze_after: int | None = None,
        question_field: str | None = None,
    ):
        """
        Args:
            module: Any dspy.Module (ReAct, RLM, Predict, …) to wrap.
            token_budget: Maximum tokens kept in the context map.
            max_trajectory_tokens: Token budget for the trajectory fed to the Distiller.
                None (default) passes the full trajectory so the Distiller does all
                compression — Set a limit only as a safety net against runaway trajectories 
                that would exceed the Distiller model's context window. When set, tie it 
                to that window: keep it at roughly ≤ 50 % of the context window to leave 
                room for the Distiller prompt, the current context map, the question, 
                and the output (e.g. ~8192 for a 128k model, ~4096 for smaller windows). 
                It should be the dominant share of the Distiller input. Step-aware
                head+tail bounding (60 % head / 40 % tail) preserves both setup and
                conclusions.
            max_input_tokens: Token budget for the serialized inputs used as the
                Distiller "question" (only active when question_field is None).
                None (default) = unbounded — Only set this when a large input field
                (e.g. an RLM document dump) is serialized via the fallback path; in
                that case ~1024 is a reasonable value, keeping it well below
                max_trajectory_tokens and the model's context window. Head+tail
                bounding is applied (60 % head / 40 % tail) so both the leading
                instruction and any trailing intent survive. Ignored when
                question_field is set.
            freeze_after: Stop updating the map after this many calls (None = always update).
            question_field: Name of the input field that carries the task description.
                If set, only that field is passed to the Distiller as the "question".
                If None, all input fields are serialized and truncated automatically —
                useful for simple signatures but lossy when inputs are large (e.g. RLM
                with document inputs). Set this explicitly whenever one field cleanly
                captures the user's intent.
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
        self.freeze_after = freeze_after
        self.question_field = question_field
        self.cmap = ContextMap()
        self.scores: dict[str, int] = {}
        self._calls = 0
        self._frozen = False

    @property
    def current_map_text(self) -> str:
        return self.cmap.render()

    def freeze(self) -> None:
        """Stop evolving the context map on subsequent forward() calls."""
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False

    def forward(self, **kwargs) -> dspy.Prediction:
        pred = self.agent(context_map=self.cmap, **kwargs)
        self._calls += 1
        if not self._frozen and (
            self.freeze_after is None or self._calls <= self.freeze_after
        ):
            self._update_cache(pred, inputs=kwargs)
        return pred

    def _update_cache(self, pred: dspy.Prediction, inputs: dict) -> None:
        if self.question_field is not None:
            question = str(inputs.get(self.question_field, ""))
        else:
            question = format_inputs(inputs, self.max_input_tokens)
        trajectory = format_trajectory(pred, self.max_trajectory_tokens)
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

