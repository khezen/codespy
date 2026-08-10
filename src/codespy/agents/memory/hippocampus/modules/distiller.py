from __future__ import annotations

import dspy

from codespy.agents.memory.hippocampus.context_map import (
    CacheCandidate,
    ContextMap,
    ItemTag,
)



class DistillerSig(dspy.Signature):
    """You are an expert analyst reviewing an agent's execution trajectory
    after it interacted with a long external context to answer a question.

    ## Key Principle: Cache Understanding, Not Answers

    The context map prepended to the agent is a compact CACHE OF
    UNDERSTANDING about the external context — NOT answers to specific
    questions. Think of it as the mental model a human builds after reading
    a document: structure, key entities, relationships, and global summaries
    that help with ANY future question on the same context.

    ## Orientation vs. Question-Specific Work

    Observe what the agent spent its iterations doing. Work falls into two
    categories:
      1. ORIENTATION WORK — figuring out what the context is, how it's
         organized, what entities/concepts exist, how they relate. This
         understanding transfers to ANY future question.
      2. QUESTION-SPECIFIC WORK — locating the specific passage or fact
         needed for THIS question. This rarely helps other questions.

    Focus on caching category (1). Ask: "If a different, unrelated question
    were asked about this same context, would this cached item save the
    agent work?"

    ## Produce three outputs

    1. DIAGNOSIS — Brief (3-5 sentences; it feeds the next module's prompt,
       so keep it terse) analysis of:
       - How many iterations the agent spent on orientation vs.
         question-specific work
       - Whether the agent re-discovered structural information that was
         already available (or should have been cached)
       - What kind of contextual understanding the agent built that
         could transfer to future questions

    2. ITEM_TAGS — For EVERY item currently in the map, tag it exactly:
       - helpful: directly helped or would directly help this run
       - harmful: misleading, incorrect, or actively hurts performance
       - neutral: correct domain knowledge not relevant to THIS question
                  but plausibly useful for other questions
       - stale:   outdated, superseded, or no longer accurate
       When tagging, distinguish between "not needed for this question"
       (neutral) from "not useful for any question" (harmful/stale).
       Domain constants, formulas, and output schemas not exercised this
       run are typically NEUTRAL, not harmful.

    3. CACHE_CANDIDATES — Items to ADD. Value tiers:

       Highest value — structural understanding that transfers across
       questions:
       - Context structure map: what sections/chapters/documents exist,
         their topics, and approximate locations (like a Table of
         Contents the agent won't have to rebuild)
       - Entity/concept inventory: key characters, actors, concepts, or
         data categories and their roles or relationships — a brief
         "glossary" that orients the agent
       - Domain constants: exact numeric values the context defines —
         thresholds, rates, formulas, conversion factors, reference
         ranges, enum sets, required output field names/types. Keep
         these numerically precise.
       - Global summaries: high-level understanding of what the context
         is about — genre, time period, key themes, nature of the data
         — that frames any question

       Medium value:
       - Parsing schema: document delimiters, boundary patterns, field
         format, how to reliably split or locate items in the context
       - Shared intermediate computations: aggregated results (counts,
         distributions, classifications) that the agent derived by
         processing the full context and that multiple questions would
         need. Note the computation method to judge reliability.

       Do NOT cache:
       - Facts that answer only one specific question (e.g., a verbatim
         quote that resolves a single query)
       - Verbose passages or long excerpts — prefer compact summaries
       - Advisory rules, warnings, or meta-instructions ("always do X",
         "never do Y") — the cache is for understanding, not instructions
       - Results from naive surface-level text operations (e.g.,
         str.count() for frequency estimation)
       - Verbatim answers to the current question

       Do NOT abstract away exact numeric values, enum sets, output field
       names/types — these are domain constants and must remain precise.

    Assign each candidate to one of these exact section names (they map
    onto the context map schema): context_understanding, domain_constants,
    context_roadmap, reusable_results, parsing_schema.

    Each candidate is a JSON object with exactly these fields:
    - section: one of the five section names above
    - value: the compact cached content (stay within max_context_item_tokens)
    - transferability: what kinds of future questions this would help
    - rationale: why this is shared understanding, not a one-off fact

    Do NOT invent extra fields (no "id", no "content", no "name").

    The litmus test for every candidate: "Would a future agent asking a
    completely DIFFERENT question about this context benefit from knowing
    this?"
    """

    trajectory: str = dspy.InputField(desc="The agent's full execution trajectory.")
    context_map: ContextMap = dspy.InputField(desc="Current context map (with item IDs).")
    question: str = dspy.InputField(desc="The question the agent was answering.")
    max_context_item_tokens: int = dspy.InputField(
        desc="Token budget for a SINGLE context-map item. Keep every candidate within "
        "it; if one exceeds it, rewrite it more compactly or split it."
    )

    diagnosis: str = dspy.OutputField(
        desc="Brief (3-5 sentence) analysis of orientation vs. question-specific work, "
        "whether structural info was re-discovered that should have been cached, and "
        "what transferable understanding the agent built. Feeds the Cartographer prompt."
    )
    item_tags: dict[str, ItemTag] = dspy.OutputField(
        desc="Per-item-id tag for EVERY item currently in the context map. "
        "Keys must match existing item ids exactly."
    )
    cache_candidates: list[CacheCandidate] = dspy.OutputField(
        desc="Candidate items to add. Each within the max_context_item_tokens budget; "
        "structural/transferable only. "
        "Each candidate's `section` must be one of the five section names above."
    )


class Distiller(dspy.Module):
    """Extracts transferable orientation knowledge from an agent trajectory.

    The context map is a prompt-resident cache of *understanding*, not
    answers. The Distiller separates orientation work (what the context
    contains, how it's organized, which constants matter) from question-
    specific work, tags every existing item, and proposes new candidates.
    """

    # Name this module's settings live under: memory.distiller.
    SIGNATURE = "distiller"

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(DistillerSig)

    def forward(
        self,
        trajectory: str,
        context_map: ContextMap,
        question: str,
        max_context_item_tokens: int,
    ):
        # SignatureContext applies memory.distiller's model/temperature/reasoning
        # effort and attributes the cost to this module rather than to whichever
        # agent triggered the reflection. It must be entered here, not by the
        # caller: Hippocampus reflects inside an asyncio.to_thread worker and
        # DSPy's context is thread-scoped.
        from codespy.agents import SignatureContext, get_cost_tracker

        with SignatureContext(self.SIGNATURE, get_cost_tracker()):
            return self.predict(
                trajectory=trajectory,
                context_map=context_map,
                question=question,
                max_context_item_tokens=max_context_item_tokens,
            )

