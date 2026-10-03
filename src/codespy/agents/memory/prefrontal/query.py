"""Pre-call facet queries: five question-shaped recalls over the local pool.

Mental models (briefings) are the abstraction layer ("when X happens,
consider Y because Z"). The facets are the semantic layer: they answer the
messier questions a briefing cannot, so they always run, even when
briefings exist.

None of these queries is the Hippocampus ``question``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Facet names, in merge/render order (1 → 5).
FACET_CONTEXT = "context"
FACET_SEEN_BEFORE = "seen_before"
FACET_DECISIONS = "decisions"
FACET_PATTERNS = "patterns"
FACET_BELIEF_CHANGES = "belief_changes"
FACET_ORDER: tuple[str, ...] = (
    FACET_CONTEXT,
    FACET_SEEN_BEFORE,
    FACET_DECISIONS,
    FACET_PATTERNS,
    FACET_BELIEF_CHANGES,
)

# Share of memory.prefrontal.recall.max_tokens per facet (sums to 1.0).
SHARE_CONTEXT = 0.30
SHARE_SEEN_BEFORE = 0.20
SHARE_DECISIONS = 0.20
SHARE_PATTERNS = 0.20
SHARE_BELIEF_CHANGES = 0.10

# Query caps.
MAX_QUERY_WORDS = 120
MAX_SUMMARY_CHARS = 400
MAX_CONTEXT_PATHS = 8
MAX_ENTITY_TERMS = 12


@dataclass(frozen=True)
class Facet:
    """One pre-call recall.

    ``fact_types`` is None and ``query`` empty for ``belief_changes``, which
    is derived from the other facets' results (no recall).
    """

    name: str
    query: str
    fact_types: tuple[str, ...] | None
    prefer_observations: bool
    share: float


def truncate_chars(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` chars, at a word boundary."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[: limit + 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else text[:limit]).rstrip(" ,;:.")


def truncate_words(text: str, limit: int = MAX_QUERY_WORDS) -> str:
    """Keep at most ``limit`` whitespace-separated words."""
    words = (text or "").split()
    return " ".join(words[:limit])


def _join(parts: Sequence[str], sep: str) -> str:
    """Join non-empty parts; with a sentence separator, drop each part's final period."""
    strip = sep.strip()
    cleaned = (s.strip() for s in parts)
    if strip:
        cleaned = (s.rstrip(strip).rstrip() for s in cleaned)
    return sep.join(p for p in cleaned if p)


def _dedup(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = (item or "").strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_facets(
    task: str,
    target: str,
    description: str = "",
    pr_title: str = "",
    paths: Sequence[str] = (),
    symbols: Sequence[str] = (),
    summary: str = "",
    packages: Sequence[str] = (),
) -> list[Facet]:
    """Build the five pre-call facets for one agent call.

    Args:
        task: Agent task name (e.g. ``code_review``).
        target: Scope topic id, or the repo for repo-level agents.
        description: Scope description (empty for repo-level agents).
        pr_title: Pull request title.
        paths: Changed file paths.
        symbols: Changed function/class names, when already known.
        summary: PR summary (or truncated PR description); capped at 400 chars.
        packages: Package names of the scope (manifest identity).
    """
    paths = _dedup(paths)
    summary = truncate_chars(summary, MAX_SUMMARY_CHARS)
    head = _join([task, target], " ")
    if description:
        head = f"{head} — {truncate_chars(description, MAX_SUMMARY_CHARS)}"

    context_q = _join(
        [
            head,
            f"PR: {pr_title.strip()}" if pr_title and pr_title.strip() else "",
            summary,
            f"Files: {', '.join(paths[:MAX_CONTEXT_PATHS])}" if paths else "",
        ],
        ". ",
    )
    entities = _dedup([*paths, *symbols])[:MAX_ENTITY_TERMS]
    seen_q = _join([target, *_dedup(packages), *entities], " ")
    decisions = (
        "decisions, trade-offs, chosen approaches, rejected alternatives, agreed conventions"
    )
    patterns = "recurring issues, repeated findings, patterns across changes"
    decisions_q = f"{target}: {decisions}" if target else decisions
    patterns_q = f"{target}: {patterns}" if target else patterns

    return [
        Facet(
            FACET_CONTEXT, truncate_words(context_q), ("experience", "world"), False, SHARE_CONTEXT
        ),
        Facet(
            FACET_SEEN_BEFORE,
            truncate_words(seen_q),
            ("world", "observation"),
            True,
            SHARE_SEEN_BEFORE,
        ),
        Facet(
            FACET_DECISIONS,
            truncate_words(decisions_q),
            ("world", "observation"),
            True,
            SHARE_DECISIONS,
        ),
        Facet(FACET_PATTERNS, truncate_words(patterns_q), ("observation",), False, SHARE_PATTERNS),
        Facet(FACET_BELIEF_CHANGES, "", None, False, SHARE_BELIEF_CHANGES),
    ]
