"""Run the pre-call facets against Cerebral."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from codespy.agents.memory.prefrontal.format import (
    BeliefChange,
    FacetResult,
    estimate_tokens,
    fget,
    render_change,
    render_fact,
)
from codespy.agents.memory.prefrontal.query import (
    FACET_BELIEF_CHANGES,
    FACET_CONTEXT,
    FACET_DECISIONS,
    FACET_PATTERNS,
    FACET_SEEN_BEFORE,
    Facet,
)

if TYPE_CHECKING:
    from codespy.agents.memory.cerebral import Cerebral

logger = logging.getLogger(__name__)

# belief_changes: observations taken from facets 2–4, and history entries kept per observation.
BELIEF_TOP_OBSERVATIONS = 3
BELIEF_HISTORY_ENTRIES = 2
BELIEF_SOURCE_FACETS: tuple[str, ...] = (FACET_SEEN_BEFORE, FACET_DECISIONS, FACET_PATTERNS)
CHANGE_MARKERS: tuple[str, ...] = ("RETRACTED", "supersedes:")


def facet_tokens(max_tokens: int, facet: Facet) -> int:
    """Token budget of one facet: its fixed share of the pre-call budget."""
    return max(1, int(max_tokens * facet.share))


async def _run_recall_facet(
    cerebral: Cerebral,
    facet: Facet,
    *,
    tag_groups: list[Any],
    max_tokens: int,
    budget: str,
    question_date: datetime | None,
    reflects: int,
    reflect_context: str | None,
) -> FacetResult:
    if not facet.query:
        return FacetResult(facet.name)
    tokens = facet_tokens(max_tokens, facet)
    try:
        if reflects > 0 and facet.name == FACET_CONTEXT:
            text, summary = await cerebral.areflect(
                facet.query,
                tag_groups=tag_groups,
                max_tokens=tokens,
                context=reflect_context,
            )
            return FacetResult(facet.name, text=text or "", reflect_summary=summary)
        facts = await cerebral.arecall(
            facet.query,
            tag_groups=tag_groups,
            max_tokens=tokens,
            fact_type=list(facet.fact_types) if facet.fact_types else None,
            prefer_observations=facet.prefer_observations,
            budget=budget,
            question_date=question_date,
        )
        return FacetResult(facet.name, facts=list(facts or []))
    except Exception:
        logger.warning("Prefrontal: facet %s failed", facet.name, exc_info=True)
        return FacetResult(facet.name, failed=True)


def _is_change_fact(fact: Any) -> bool:
    if fget(fact, "fact_type") != "world":
        return False
    text = fget(fact, "text") or ""
    return any(marker in text for marker in CHANGE_MARKERS)


async def _belief_changes(
    cerebral: Cerebral,
    results: Mapping[str, FacetResult],
    facet: Facet,
    max_tokens: int,
) -> FacetResult:
    """Facet 5: what did we believe before, and what changed?

    No recall and no embedding: reads the history of the top observations
    of facets 2–4 and moves ``RETRACTED``/``supersedes:`` world facts out
    of facets 1–4 into this section (so the merge's by-id dedup does not
    drop them). Capped at the facet's share.
    """
    out = FacetResult(facet.name)
    budget = facet_tokens(max_tokens, facet)

    # Top observations from facets 2–4, by facet order then result order.
    observations: list[Any] = []
    seen: set[str] = set()
    for name in BELIEF_SOURCE_FACETS:
        for fact in (results.get(name) or FacetResult(name)).facts:
            fid = fget(fact, "id")
            if fget(fact, "fact_type") == "observation" and fid and fid not in seen:
                seen.add(fid)
                observations.append(fact)
    observations = observations[:BELIEF_TOP_OBSERVATIONS]

    async def _history(obs: Any) -> list[dict]:
        try:
            return await cerebral.aget_observation_history(fget(obs, "id"))
        except Exception:
            logger.debug(
                "Prefrontal: observation history failed for %s", fget(obs, "id"), exc_info=True
            )
            return []

    histories = await asyncio.gather(*(_history(o) for o in observations))

    used = 0
    full = False
    for obs, history in zip(observations, histories, strict=True):
        if full:
            break
        for i, entry in enumerate((history or [])[:BELIEF_HISTORY_ENTRIES]):
            previous = (entry or {}).get("previous_text")
            if not previous:
                continue
            changed = (entry or {}).get("changed_at")
            change = BeliefChange(
                id=f"history:{fget(obs, 'id')}:{i}",
                current=fget(obs, "text") or "",
                previous=previous,
                changed_at=str(changed)[:10] if changed else None,
            )
            cost = estimate_tokens(render_change(change))
            if used + cost > budget:
                full = True
                break
            used += cost
            out.changes.append(change)

    # Move RETRACTED / supersedes: world facts out of facets 1–4 (those that
    # do not fit the share stay in their source facet).
    for name in (FACET_CONTEXT, *BELIEF_SOURCE_FACETS):
        source = results.get(name)
        if source is None:
            continue
        kept: list[Any] = []
        for fact in source.facts:
            if _is_change_fact(fact):
                cost = estimate_tokens(render_fact(fact))
                if used + cost <= budget:
                    used += cost
                    out.facts.append(fact)
                    continue
            kept.append(fact)
        source.facts = kept
    return out


async def run_facets(
    cerebral: Cerebral,
    facets: Sequence[Facet],
    *,
    tag_groups: list[Any],
    max_tokens: int,
    budget: str = "mid",
    question_date: datetime | None = None,
    reflects: int = 0,
    reflect_context: str | None = None,
) -> dict[str, FacetResult]:
    """Run facets 1–4 concurrently, then facet 5 from their observations.

    With ``reflects > 0`` only the ``context`` facet runs Hindsight's reflect
    loop; the others stay raw recall. Each facet fails on its own and yields
    an empty section.
    """
    recall_facets = [f for f in facets if f.name != FACET_BELIEF_CHANGES]
    gathered = await asyncio.gather(
        *(
            _run_recall_facet(
                cerebral,
                f,
                tag_groups=tag_groups,
                max_tokens=max_tokens,
                budget=budget,
                question_date=question_date,
                reflects=reflects,
                reflect_context=reflect_context,
            )
            for f in recall_facets
        )
    )
    results: dict[str, FacetResult] = {r.name: r for r in gathered}
    belief = next((f for f in facets if f.name == FACET_BELIEF_CHANGES), None)
    if belief is not None:
        try:
            results[FACET_BELIEF_CHANGES] = await _belief_changes(
                cerebral, results, belief, max_tokens
            )
        except Exception:
            logger.warning("Prefrontal: facet %s failed", FACET_BELIEF_CHANGES, exc_info=True)
            results[FACET_BELIEF_CHANGES] = FacetResult(FACET_BELIEF_CHANGES, failed=True)
    return results
