"""Render Prefrontal context as one deterministic plain-text block."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from codespy.agents.memory.prefrontal.query import (
    FACET_BELIEF_CHANGES,
    FACET_CONTEXT,
    FACET_DECISIONS,
    FACET_ORDER,
    FACET_PATTERNS,
    FACET_SEEN_BEFORE,
)
from codespy.agents.memory.prefrontal.reach import (
    REACH_BANK,
    TAG_REPO,
    TAG_TASK,
    org_of,
    tag_value,
)

SECTION_TITLES: dict[str, str] = {
    FACET_CONTEXT: "Around this work",
    FACET_SEEN_BEFORE: "Seen before",
    FACET_DECISIONS: "Decisions",
    FACET_PATTERNS: "Recurring patterns",
    FACET_BELIEF_CHANGES: "What changed",
}
# Title prefixes of the briefing and remote-facts sections (titles vary).
BRIEFING_TITLE_PREFIX = "Briefing"
REMOTE_TITLE_PREFIX = "Other repositories"
# Rendered section header: ``## <title>`` on its own line.
SECTION_HEADER_PREFIX = "## "


def is_section_title(title: str) -> bool:
    """Whether ``title`` is one :func:`format_context_sections` renders.

    Used to split rendered memory on known headers only: briefing bodies and
    reflect answers carry their own markdown headings.
    """
    title = title.strip()
    return (
        title in SECTION_TITLES.values()
        or title.startswith(BRIEFING_TITLE_PREFIX)
        or title.startswith(REMOTE_TITLE_PREFIX)
    )


@dataclass(frozen=True)
class BeliefChange:
    """One past wording of an observation."""

    id: str
    current: str
    previous: str
    changed_at: str | None = None


@dataclass
class FacetResult:
    """Outcome of one facet. ``text`` holds a reflect answer (reflects > 0)."""

    name: str
    facts: list[Any] = field(default_factory=list)
    text: str = ""
    changes: list[BeliefChange] = field(default_factory=list)
    failed: bool = False
    reflect_summary: Any = None  # ReflectSummary from areflect, when reflects > 0
    dropped: int = 0  # facts removed by the relevance floor (min_similarity)
    # Every fact before the relevance floor, as (fact, kept), for score logging.
    scored: list[tuple[Any, bool]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.facts) + len(self.changes) + (1 if self.text else 0)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars per token), no tokenizer call."""
    return (len(text or "") + 3) // 4


def fget(fact: Any, name: str, default: Any = None) -> Any:
    """Read a field from a MemoryFact-like object or a dict."""
    if isinstance(fact, Mapping):
        return fact.get(name, default)
    return getattr(fact, name, default)


def semantic_score(fact: Any) -> float | None:
    """Cosine similarity of the fact to the query (``scores.semantic``), or None.

    None when the fact carries no scores or the semantic arm did not surface it.
    """
    scores = fget(fact, "scores")
    if scores is None:
        return None
    value = fget(scores, "semantic")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def filter_relevant(facts: Iterable[Any], floor: float) -> tuple[list[Any], list[Any]]:
    """Split ``facts`` into (kept, dropped) by the relevance floor.

    A fact is kept when its semantic score is not None and ``>= floor``.
    With ``floor <= 0`` every fact is kept (no filtering).
    """
    facts = list(facts or [])
    if floor <= 0:
        return facts, []
    kept: list[Any] = []
    dropped: list[Any] = []
    for fact in facts:
        score = semantic_score(fact)
        (kept if score is not None and score >= floor else dropped).append(fact)
    return kept, dropped


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def fact_date(fact: Any) -> str | None:
    """``YYYY-MM-DD`` of when the fact occurred, else when it was learned."""
    raw = fget(fact, "occurred_start") or fget(fact, "mentioned_at")
    if not raw:
        return None
    return str(raw)[:10]


def render_fact(fact: Any, *, with_repo: bool = False) -> str:
    """Render one fact as a bullet.

    - observation: ``- [observation] <text>`` (``[repo · observation]`` for remote)
    - otherwise: ``- [<date> · <task>] <text>`` (``[repo · date · task]`` for remote)
    """
    tags = fget(fact, "tags") or []
    parts: list[str] = []
    if with_repo:
        repo = tag_value(tags, TAG_REPO)
        if repo:
            parts.append(repo)
    if fget(fact, "fact_type") == "observation":
        parts.append("observation")
    else:
        date = fact_date(fact)
        if date:
            parts.append(date)
        task = tag_value(tags, TAG_TASK)
        if task:
            parts.append(task)
    text = _clean(fget(fact, "text"))
    label = f"[{' · '.join(parts)}] " if parts else ""
    return f"- {label}{text}"


def render_change(change: BeliefChange) -> str:
    """``- <current> (was: <previous>, changed <date>)``."""
    when = f", changed {change.changed_at}" if change.changed_at else ""
    return f"- {_clean(change.current)} (was: {_clean(change.previous)}{when})"


class _Dedup:
    """Keeps the first occurrence: by id, then by exact text."""

    def __init__(self) -> None:
        self.ids: set[str] = set()
        self.texts: set[str] = set()

    def admit(self, key_id: str | None, text: str) -> bool:
        text = _clean(text)
        if key_id and key_id in self.ids:
            return False
        if text and text in self.texts:
            return False
        if key_id:
            self.ids.add(key_id)
        if text:
            self.texts.add(text)
        return True


def format_context_sections(
    briefings: Sequence[Mapping[str, Any]],
    facet_results: Mapping[str, FacetResult],
    remote_facts: Sequence[Any],
    repo: str,
    reach: str = "org",
) -> list[tuple[str, str]]:
    """Render briefings, facet sections (order 1 → 5) and remote facts as sections.

    Returns a list of (title, body) tuples in the current order. Empty sections
    are omitted. The result can be joined by format_context to produce the same
    output as before.

    Facts are deduplicated across sections by id, then by exact text, keeping
    the first occurrence.
    """
    sections: list[tuple[str, str]] = []

    # Briefing sections
    for mm in briefings or []:
        content = (mm.get("content") or "").strip()
        if not content:
            continue
        name = (mm.get("name") or BRIEFING_TITLE_PREFIX).strip()
        if not name.startswith(BRIEFING_TITLE_PREFIX):
            name = f"{BRIEFING_TITLE_PREFIX}: {name}"
        sections.append((name, content))

    dedup = _Dedup()

    # Facet sections in FACET_ORDER
    for name in FACET_ORDER:
        result = facet_results.get(name)
        if result is None:
            continue
        lines: list[str] = []
        if result.text:
            lines.append(result.text.strip())
        for fact in result.facts:
            if dedup.admit(fget(fact, "id"), fget(fact, "text") or ""):
                lines.append(render_fact(fact))
        for change in result.changes:
            if dedup.admit(change.id, f"{change.current} (was: {change.previous})"):
                lines.append(render_change(change))
        if lines:
            sections.append((SECTION_TITLES[name], "\n".join(lines)))

    # Remote facts section
    remote_lines = [
        render_fact(fact, with_repo=True)
        for fact in remote_facts or []
        if dedup.admit(fget(fact, "id"), fget(fact, "text") or "")
    ]
    if remote_lines:
        where = "" if reach == REACH_BANK else f" in {org_of(repo)}"
        title = f"{REMOTE_TITLE_PREFIX}{where} — verify before relying on it"
        sections.append((title, "\n".join(remote_lines)))

    return sections


def format_context(
    briefings: Sequence[Mapping[str, Any]],
    facet_results: Mapping[str, FacetResult],
    remote_facts: Sequence[Any],
    repo: str,
    reach: str = "org",
) -> str:
    """Render briefings, facet sections (order 1 → 5) and remote facts.

    Facts are deduplicated across sections by id, then by exact text, keeping
    the first occurrence. Empty sections are omitted; nothing at all → ``""``.
    """
    sections = format_context_sections(briefings, facet_results, remote_facts, repo, reach)
    return join_sections(sections)


def join_sections(sections: Sequence[tuple[str, str]]) -> str:
    """Render (title, body) sections as ``## <title>\\n<body>`` blocks."""
    return "\n\n".join(f"{SECTION_HEADER_PREFIX}{title}\n{body}" for title, body in sections)


def format_facts(facts: Sequence[Any], *, with_repo: bool = False) -> str:
    """Render a flat, deduplicated bullet list (used by the recall tool)."""
    dedup = _Dedup()
    return "\n".join(
        render_fact(f, with_repo=with_repo)
        for f in facts or []
        if dedup.admit(fget(f, "id"), fget(f, "text") or "")
    )
