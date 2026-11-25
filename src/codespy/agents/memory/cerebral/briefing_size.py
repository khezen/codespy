"""Size limit for briefings (Hindsight mental models), enforced by rewriting.

Hindsight treats a mental model's ``max_tokens`` as a target: a full refresh
makes one unchecked "shorten" call, and a delta refresh never shrinks the
document (it only warns). On peaks-com/mono#10519, full refreshes at
``max_tokens=1024`` stored 1,403-2,607 tokens. codespy enforces the limit by
having a model rewrite the briefing (``mental_models.shorten_model``) and
checking every result: it must fit, keep every section heading, and keep at
least ``COLLAPSE_RATIO`` of the budget. A failed check is sent back with the
measured size and the missing sections. Nothing is ever cut mechanically: if no
attempt passes, the best valid rewrite (or the original) is kept and the miss is
logged.

Sizes are in Hindsight's tokens (the unit ``max_tokens`` is in).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# A rewrite below this share of the budget lost too much: sent back.
COLLAPSE_RATIO = 0.25
# Rewrite attempts per briefing (Kimi K2.5 fit mono#10519's briefings in 2).
MAX_ATTEMPTS = 4
# Word ceiling as a share of the budget: models overshoot word targets, and
# Hindsight tokens per word vary, so the ceiling sits under the budget.
AIM_RATIO = 0.75

# A rewrite that shrank less than 5% from the previous one is stuck: the next
# word ceiling is lowered by this factor (Kimi repeats a near miss otherwise).
STUCK_RATIO = 0.95

# (chat messages, temperature) -> completion text
Complete = Callable[[list[dict[str, str]], float], Awaitable[str]]

# Temperature per attempt: retries vary more, so a stuck near miss can move.
TEMPERATURES = (0.2, 0.6, 0.9, 0.9)

SYSTEM_PROMPT = (
    "You rewrite an engineering briefing about a codebase so it is shorter. The briefing is "
    "read by an AI agent before it reviews or changes this code.\n"
    "- Keep every '## ' section heading, unchanged and in order.\n"
    "- Keep every distinct fact that matters to someone changing this code: names, paths, "
    "values, rules, invariants, pitfalls, and facts marked wrong.\n"
    "- Merge repeated or overlapping points, shorten wording, drop filler and examples that "
    "only restate a rule.\n"
    "- Never invent, generalise away, or soften a fact.\n"
    "- Output only the rewritten markdown, no preamble.\n"
    "Hard limit: at most {words} words. The text is {current} words now."
)


def count_tokens(text: str) -> int:
    """Tokens under Hindsight's tokenizer; chars/4 if it cannot be loaded."""
    try:
        from hindsight_api.engine.reflect.tokenization import count_prompt_tokens

        return count_prompt_tokens(text or "")
    except Exception:
        return (len(text or "") + 3) // 4


def to_document(markdown: str) -> Any:
    """Hindsight's canonical document for ``markdown`` (``.markdown``, ``.structure``)."""
    from hindsight_api.engine.reflect.structured_doc import canonical_document

    return canonical_document(markdown)


def headings(markdown: str) -> list[str]:
    """Section headings, in order (as Hindsight splits them)."""
    sections = to_document(markdown).structure.sections
    return [s.heading.strip() for s in sections if s.heading.strip()]


def _words_for(text: str, tokens: int, target_tokens: int) -> int:
    """Word count that ``text`` would have at ``target_tokens`` (its own words/token ratio)."""
    words = max(1, len(text.split()))
    return max(1, int(words * target_tokens / max(1, tokens)))


def _strip_fence(text: str) -> str:
    """Drop a ```markdown fence wrapping the whole answer."""
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


@dataclass
class SizeResult:
    markdown: str
    structure: Any
    before: int
    after: int
    attempts: list[int] = field(default_factory=list)  # tokens of each rewrite
    fits: bool = False
    changed: bool = False


async def enforce(markdown: str, budget: int, complete: Complete | None) -> SizeResult | None:
    """Rewrite a briefing until it fits ``budget``; None when it already fits.

    Keeps the best valid rewrite (all headings, not collapsed, shorter than the
    original) when no attempt fits; the original when none is valid.
    """
    before = count_tokens(markdown)
    if before <= budget:
        return None
    result = SizeResult(markdown, None, before, before)
    if complete is None:
        return result
    required = headings(markdown)
    aim = int(budget * AIM_RATIO)
    best: tuple[int, str] | None = None  # (tokens, text)
    source, source_tokens = markdown, before
    feedback = ""
    prev: int | None = None
    for attempt in range(MAX_ATTEMPTS):
        # Fresh conversation each attempt, on the best valid rewrite so far: in a
        # growing conversation Kimi repeated its previous answer word for word.
        system = SYSTEM_PROMPT.format(
            words=_words_for(source, source_tokens, aim), current=len(source.split())
        )
        messages = [
            {"role": "system", "content": system + (f"\n{feedback}" if feedback else "")},
            {"role": "user", "content": source},
        ]
        try:
            text = _strip_fence(await complete(messages, TEMPERATURES[attempt]))
        except Exception:
            logger.warning("cerebral: briefing rewrite call failed", exc_info=True)
            break
        tokens = count_tokens(text)
        result.attempts.append(tokens)
        missing = [h for h in required if h not in set(headings(text))]
        problems: list[str] = []
        if tokens > budget:
            problems.append(f"it came out at {tokens} tokens, over the {budget}-token limit")
        if tokens < budget * COLLAPSE_RATIO:
            problems.append(f"it came out at only {tokens} tokens and lost most of the content")
        if missing:
            problems.append(
                "it dropped these sections: " + "; ".join(f"'## {h}'" for h in missing)
            )
        valid = not missing and tokens >= budget * COLLAPSE_RATIO and tokens < before
        if valid and (best is None or tokens <= budget < best[0] or budget < tokens < best[0]):
            best = (tokens, text)
            source, source_tokens = text, tokens
        if not problems:
            break
        if tokens > budget and prev is not None and tokens > prev * STUCK_RATIO:
            aim = int(aim * STUCK_RATIO)  # barely moved: ask for clearly less
        prev = tokens
        feedback = "A previous rewrite failed: " + "; ".join(problems) + "."
    if best is not None:
        doc = to_document(best[1])
        result.markdown, result.structure = doc.markdown, doc.structure
        result.after = count_tokens(doc.markdown)
        result.fits = result.after <= budget
        result.changed = True
    return result
