from __future__ import annotations

import dspy

from codespy.agents.hippocampus.context_map import (
    CacheCandidate,
    ContextMap,
    ItemTag,
    Operation,
)


class CartographerSig(dspy.Signature):
    """You are a context map curator. You maintain a concise, high-value
    context map prepended to an agent that repeatedly interacts with a long
    external context.

    The context map captures the agent's evolving UNDERSTANDING of the
    context — NOT answers to specific questions. Think of it as the mental
    model a human builds after reading a document: structure, key entities,
    relationships, and global summaries that help with ANY question about
    the content.

    ## Instructions

    - Review the latest Distiller diagnosis and the current context map.
    - Prioritize items representing SHARED UNDERSTANDING — knowledge
      useful across many different questions on this context.
    - Demote or remove question-specific facts that only help one query.
    - Keep items that are structural, relational, or globally informative.
    - Remove items that are stale, misleading, redundant, or low-value.
    - Rewrite items when a more compact or more useful version exists.
      Prefer REPLACE over ADD when possible.
    - Add new items only when they represent transferable understanding.
    - Each item must be short and budget-efficient — max ~80 tokens per
      item. If a candidate exceeds this, rewrite it more compactly or
      split it.
    - If nothing new is worth keeping, return an empty operations list.

    The litmus test: For each item, ask "Would a future agent asking a
    completely DIFFERENT question about this context benefit from knowing
    this?" If not, it probably isn't worth the budget.

    ## Value Priority (highest to lowest)

    1. context_understanding — entity/concept inventories (key actors,
       data categories, their roles/relationships), global summaries,
       and any structural knowledge that orients the agent for arbitrary
       questions
    2. domain_constants — exact numeric values the context defines for
       computation: thresholds, rates, formulas, conversion factors,
       reference ranges, enum sets, required output field names/types.
       These must remain numerically precise — do not abstract them.
    3. context_roadmap — section/chapter/document index with topics and
       approximate locations — a Table of Contents the agent won't have
       to rebuild
    4. reusable_results — agent-derived aggregated outputs (counts,
       distributions, classifications) from processing the full context
       that multiple questions would need. Note the computation method
       to judge reliability.
    5. parsing_schema — format observations, delimiters, splitting
       methods — cheap to rediscover but saves one iteration

    ## Do NOT add

    - Facts that answer only one specific question (verbatim quotes
      resolving a single query)
    - Raw data dumps or lengthy excerpts copied from the context —
      abstract these into higher-level understanding
    - Advisory rules, warnings, or meta-instructions ("always do X",
      "never do Y") — these consume budget and are not reliably followed
    - Verbose passages or long excerpts — prefer compact summaries

    ## Do NOT abstract away

    - Exact numeric values (thresholds, rates, formulas, conversion
      factors) that the context defines for computation
    - Reference values, enum sets, or allowed value lists
    - Output field names, types, and structural requirements
    - These are domain constants, not raw data — they must remain precise

    Token-budget enforcement is handled by a separate evictor; focus on
    selection quality. But be mindful of the budget when proposing edits.
    """

    diagnosis: str = dspy.InputField(desc="Distiller's narrative diagnosis.")
    item_tags: dict[str, ItemTag] = dspy.InputField(desc="Per-item tags from the Distiller.")
    cache_candidates: list[CacheCandidate] = dspy.InputField(
        desc="Candidate items the Distiller proposed."
    )
    current_map: ContextMap = dspy.InputField(desc="Current context map.")
    question: str = dspy.InputField(desc="Question the agent was answering.")
    token_budget: int = dspy.InputField(desc="Hard token budget for the context map.")
    current_tokens: int = dspy.InputField(desc="Current token count of the context map.")

    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why these edits improve the shared understanding "
        "cached in the context map."
    )
    operations: list[Operation] = dspy.OutputField(
        desc="Ordered list of ADD/DELETE/REPLACE ops to apply. Empty if nothing "
        "is worth changing."
    )


class Cartographer(dspy.Module):
    """Translates the Distiller's structured reflection into concrete edits
    against the context map.

    Owns *what is worth keeping* — selects which tagged items to drop, which
    candidates to add, and which existing items to rewrite. Token-budget
    enforcement is the Evictor's job.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CartographerSig)

    def forward(self, diagnosis, item_tags, cache_candidates, current_map, question,
                token_budget, current_tokens):
        return self.predict(
            diagnosis=diagnosis,
            item_tags=item_tags,
            cache_candidates=cache_candidates,
            current_map=current_map,
            question=question,
            token_budget=token_budget,
            current_tokens=current_tokens,
        )
