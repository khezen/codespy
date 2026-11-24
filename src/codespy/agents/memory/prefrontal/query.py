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

    ``remote_query`` (``context`` only) is the PR topic used for the remote
    pool. ``relevance_floor`` marks facets whose raw-recall results are
    filtered by ``memory.prefrontal.recall.min_similarity``.
    """

    name: str
    query: str
    fact_types: tuple[str, ...] | None
    prefer_observations: bool
    share: float
    remote_query: str = ""
    relevance_floor: bool = False


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


def _topic_query(topic: str, phrasing: str) -> str:
    """``<topic>. <phrasing>``, cutting the topic tail so the phrasing always fits."""
    room = max(0, MAX_QUERY_WORDS - len(phrasing.split()))
    return _join([truncate_words(topic, room), phrasing], ". ")


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
    title = (pr_title or "").strip()
    head = _join([task, target], " ")
    if description:
        head = f"{head} — {truncate_chars(description, MAX_SUMMARY_CHARS)}"

    context_q = _join(
        [
            head,
            f"PR: {title}" if title else "",
            summary,
            f"Files: {', '.join(paths[:MAX_CONTEXT_PATHS])}" if paths else "",
        ],
        ". ",
    )
    # PR topic: what the PR is about, without task, repo/scope id or paths
    # (those match every fact of the scope, or of every repo, lexically and
    # semantically). Used for the remote pool and decisions/patterns.
    topic = _join(
        [
            truncate_chars(description, MAX_SUMMARY_CHARS),
            f"PR: {title}" if title else "",
            summary,
        ],
        ". ",
    )

    # seen_before: what the PR touches. The tag pool already restricts to the
    # PR's scopes, so target/packages only inflate similarity for every scope fact.
    entities = _dedup([*paths, *symbols])[:MAX_ENTITY_TERMS]
    if entities:
        seen_q, seen_floor = " ".join(entities), True
    else:
        seen_q, seen_floor = _join([target, *_dedup(packages)], " "), False

    decisions = (
        "decisions, trade-offs, chosen approaches, rejected alternatives, agreed conventions"
    )
    patterns = "recurring issues, repeated findings, patterns across changes"
    if topic:
        # Topic first: truncate_words cuts the summary tail, never the phrasing.
        decisions_q = _topic_query(topic, decisions)
        patterns_q = _topic_query(topic, patterns)
        topic_floor = True
    else:
        decisions_q = f"{target}: {decisions}" if target else decisions
        patterns_q = f"{target}: {patterns}" if target else patterns
        topic_floor = False

    return [
        Facet(
            FACET_CONTEXT,
            truncate_words(context_q),
            ("experience", "world"),
            False,
            SHARE_CONTEXT,
            remote_query=truncate_words(topic),
        ),
        Facet(
            FACET_SEEN_BEFORE,
            truncate_words(seen_q),
            ("world", "observation"),
            True,
            SHARE_SEEN_BEFORE,
            relevance_floor=seen_floor,
        ),
        Facet(
            FACET_DECISIONS,
            truncate_words(decisions_q),
            ("world", "observation"),
            True,
            SHARE_DECISIONS,
            relevance_floor=topic_floor,
        ),
        Facet(
            FACET_PATTERNS,
            truncate_words(patterns_q),
            ("observation",),
            False,
            SHARE_PATTERNS,
            relevance_floor=topic_floor,
        ),
        Facet(FACET_BELIEF_CHANGES, "", None, False, SHARE_BELIEF_CHANGES),
    ]
