from __future__ import annotations

import dspy

from codespy.agents.memory.hippocampus.context_memory import (
    CacheCandidate,
    ItemTag,
    Operation,
)


class CartographerSig(dspy.Signature):
    """You are a context memory curator. You maintain a concise, high-value
    context memory prepended to an agent that repeatedly interacts with a long
    external context.

    The context memory captures the agent's evolving UNDERSTANDING of the
    context — NOT answers to specific questions. Think of it as the mental
    model a human builds after reading a document: structure, key entities,
    relationships, and global summaries that help with ANY question about
    the content.

    ## Instructions

    - Review the latest Distiller diagnosis and the current context memory.
    - Prioritize items representing SHARED UNDERSTANDING — knowledge
      useful across many different questions on this context.
    - Demote or remove question-specific facts that only help one query.
    - Keep items that are structural, relational, or globally informative.
    - Remove items that are stale, misleading, redundant, or low-value.
    - Rewrite items when a more compact or more useful version exists.
      Prefer REPLACE over ADD when possible.
    - Add new items only when they represent transferable understanding.
    - Each item must be short and budget-efficient — stay within the
      `max_context_item_tokens` budget given as an input. If a candidate exceeds
      it, rewrite it more compactly or split it.
    - If nothing new is worth keeping, return an empty operations list.

    The litmus test: For each item, ask "Would a future agent asking a
    completely DIFFERENT question about this context benefit from knowing
    this?" If not, it probably isn't worth the budget.

    ## How to use item_tags

    The Distiller assigns each existing item a tag. Let it drive your ops:
    - harmful / stale → DELETE the item (unless a corrected REPLACE is
      clearly the better fix).
    - helpful but verbose or redundant → REPLACE with a tighter version.
    - helpful and already compact → leave it; don't spend an op.
    - neutral → keep as-is; do not churn ops on neutral items.

    ## Operation rules

    Emit only well-formed operations that satisfy the schema:
    - ADD: requires `section` (one of the five section names) and `content`.
    - DELETE: requires `item_id`.
    - REPLACE: requires `item_id` and `content`.
    - Only reference `item_id`s that exist in the current memory. Never invent
      ids — new items get their ids assigned automatically on ADD.

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
    current_map: str = dspy.InputField(desc="Current context memory (topic-grouped, with item IDs and sections).")
    question: str = dspy.InputField(desc="Question the agent was answering.")
    token_budget: int = dspy.InputField(desc="Hard token budget for the context memory.")
    current_tokens: int = dspy.InputField(desc="Current token count of the context memory.")
    max_context_item_tokens: int = dspy.InputField(
        desc="Token budget for a SINGLE context memory item. Every ADD/REPLACE content "
        "must stay within it."
    )

    justification: str = dspy.OutputField(
        desc="Brief explanation of why these edits improve the shared understanding "
        "cached in the context memory."
    )

    operations: list[Operation] = dspy.OutputField(
        desc="Ordered list of ADD/DELETE/REPLACE ops to apply. Empty if nothing "
        "is worth changing."
    )


class Cartographer(dspy.Module):
    """Translates the Distiller's structured reflection into concrete edits
    against the context memory.

    Owns *what is worth keeping* — selects which tagged items to drop, which
    candidates to add, and which existing items to rewrite. Token-budget
    enforcement is the Evictor's job.
    """

    # Name this module's settings live under: memory.cartographer.
    SIGNATURE = "cartographer"

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(CartographerSig)

    def forward(self, diagnosis, item_tags, cache_candidates, current_map, question,
                token_budget, current_tokens, max_context_item_tokens):
        # See Distiller.forward: SignatureContext applies memory.cartographer's
        # LLM settings and gives this module its own cost line. Entered here
        # because DSPy's context is thread-scoped and reflection runs in a worker.
        from codespy.agents import SignatureContext, get_cost_tracker

        with SignatureContext(self.SIGNATURE, get_cost_tracker()):
            return self.predict(
                diagnosis=diagnosis,
                item_tags=item_tags,
                cache_candidates=cache_candidates,
                current_map=current_map,
                question=question,
                token_budget=token_budget,
                current_tokens=current_tokens,
                max_context_item_tokens=max_context_item_tokens,
            )
