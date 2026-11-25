"""Prefrontal – hands useful prior knowledge from Cerebral to an agent.

Reads Cerebral (the Hindsight bank) and never writes to it. Delivery is
hybrid:

1. a pre-call context injected as the read-only ``prefrontal_memory`` input
   (mental-model briefings, then five facets, then remote facts), and
2. a ``recall_memory`` tool the RLM agents can call for follow-ups.

Prefrontal and Hippocampus never feed each other: the Distiller and the
Cartographer never see ``prefrontal_memory``.

Every failure logs a warning and degrades to less context, or to ``""``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import dspy  # type: ignore[import-untyped]

from codespy.agents.memory.prefrontal.facets import run_facets
from codespy.agents.memory.prefrontal.format import (
    BeliefChange,
    FacetResult,
    estimate_tokens,
    fget,
    filter_relevant,
    format_context_sections,
    format_facts,
    join_sections,
    semantic_score,
)
from codespy.agents.memory.prefrontal.query import (
    FACET_BELIEF_CHANGES,
    FACET_CONTEXT,
    FACET_DECISIONS,
    FACET_PATTERNS,
    FACET_SEEN_BEFORE,
    Facet,
    truncate_words,
)
from codespy.agents.memory.prefrontal.reach import (
    REACH_LOCAL,
    TAG_PROJECT_SCOPE,
    TAG_REPO,
    build_local_groups,
    build_reflect_groups,
    build_remote_groups,
    clamp_reach,
    is_own_repo_fact,
    local_tags,
    mental_model_ids,
    tag_value,
)
from codespy.agents.memory.recall import RecallRecord, RecallUsage, track_recall_usage

if TYPE_CHECKING:
    from codespy.agents.memory.cerebral import Cerebral
    from codespy.config_memory import PrefrontalConfig

logger = logging.getLogger(__name__)

PREFRONTAL_FIELD = "prefrontal_memory"
PREFRONTAL_FIELD_DESC = (
    "Prior knowledge from earlier runs (read-only, may be stale or wrong). "
    "See MEMORY in the instructions."
)

# Appended to the signature instructions of every agent that receives memory.
# No braces: code_review fills ``{max_output_chars}`` in its instructions later.
MEMORY_BLOCK_HEADER = "MEMORY (prefrontal_memory)"
MEMORY_BLOCK = f"""{MEMORY_BLOCK_HEADER}

prefrontal_memory holds what earlier codespy runs learned about this repository.
It is read-only and may be stale or wrong. It has "## <title>" sections:
- Briefing: standing conventions of a scope.
- Around this work: a synthesis of prior knowledge for this change.
- Seen before: prior facts about the changed files.
- Decisions: agreed choices and conventions.
- Recurring patterns: issues found before.
- What changed: beliefs that were revised.
- Other repositories: facts from the same organisation; verify before relying on them.

How to use it:
- Read it before you start judging the change.
- Check recurring patterns and prior findings against the changed files first.
- Respect recorded decisions: do not report what a decision explicitly accepts;
  do report a change that breaks a decision.
- Use it to direct tool calls.
- Never report an issue from memory alone: verify it in the code with tools.
- When memory and code disagree, trust the code."""

# Appended for agents whose SUBMIT is gated on reading memory (require_read).
MEMORY_READ_BLOCK_HEADER = "Reading memory:"
MEMORY_READ_BLOCK = f"""{MEMORY_READ_BLOCK_HEADER}
- You only see what you print: the prefrontal_memory preview shows 1,000 characters.
- Print prefrontal_memory in full early, before reviewing, in slices under the
  output cap (print(prefrontal_memory[a:b])).
- An early SUBMIT is sent back with the memory sections not read yet."""

RECALL_LIMIT_REACHED = "recall limit reached"
MEMORY_UNAVAILABLE = "memory unavailable"
NO_RELEVANT_MEMORY = "no relevant memory"
REMOTE_REFLECT_SUFFIX = (
    " For every fact that comes from another repository, name its source repository."
)
REMOTE_TOOL_HEADER = "Other repositories — verify before relying on it:"
# Max query chars in the per-recall INFO log line.
LOG_QUERY_CHARS = 120
# Max fact-text chars in the per-fact score INFO log line.
LOG_SCORE_TEXT_CHARS = 80
# Sections whose per-fact semantic scores are logged and stored (calibration).
SCORED_FACETS: tuple[str, ...] = (FACET_SEEN_BEFORE, FACET_DECISIONS, FACET_PATTERNS)
SECTION_REMOTE = "remote"
SECTION_TOOL_REMOTE = "tool_remote"

RecallStatus = Literal["ok", "empty", "limit", "error"]


def _tool_status(text: str) -> RecallStatus:
    """Classify a recall_memory result string."""
    if text == RECALL_LIMIT_REACHED:
        return "limit"
    if text == NO_RELEVANT_MEMORY:
        return "empty"
    if text == MEMORY_UNAVAILABLE:
        return "error"
    return "ok"

_SIGNATURES: dict[tuple[type, bool], type] = {}


def _change_observation_id(change: BeliefChange) -> str:
    """Observation id of a belief change (id ``history:<observation id>:<i>``)."""
    cid = change.id or ""
    if cid.startswith("history:"):
        return cid[len("history:"):].rsplit(":", 1)[0]
    return cid


def with_prefrontal_memory(
    signature: type[dspy.Signature], *, require_read: bool = False
) -> type[dspy.Signature]:
    """Return ``signature`` with a leading ``prefrontal_memory: str`` input (idempotent).

    Also appends the MEMORY block (what memory is, how to use it) to the
    instructions, and with ``require_read`` the block telling the model to print
    it before SUBMIT (pair with ``rlm_tools.memory_check``). Neither the field
    nor a block is added twice.

    Must be applied before the module is built: ``dspy.RLM`` builds its
    predictors from the signature in ``__init__``. Use the same returned
    signature for ``ContextSafe`` so its RLM fallback has the field too.
    """
    key = (signature, require_read)
    cached = _SIGNATURES.get(key)
    if cached is not None:
        return cached
    sig = signature
    if PREFRONTAL_FIELD not in sig.input_fields:
        sig = sig.prepend(
            PREFRONTAL_FIELD,
            dspy.InputField(desc=PREFRONTAL_FIELD_DESC),
            type_=str,
        )
    instructions = sig.instructions or ""
    blocks: list[str] = []
    if MEMORY_BLOCK_HEADER not in instructions:
        blocks.append(MEMORY_BLOCK)
    if require_read and MEMORY_READ_BLOCK_HEADER not in instructions:
        blocks.append(MEMORY_READ_BLOCK)
    if blocks:
        sig = sig.with_instructions("\n\n".join([instructions.rstrip(), *blocks]).lstrip())
    _SIGNATURES[key] = sig
    return sig


class Prefrontal:
    """Per-agent-call reader of Cerebral.

    Args:
        settings: ``memory.prefrontal`` config.
        cerebral: The Cerebral instance to read.
        task_name: Agent task (e.g. ``code_review``).
        repo_full_name: ``owner/repo``.
        scope_topic_ids: ``project_scope`` topic ids of the agent's scope(s).
        include_repo: Also read the repo-root scope (``project_scope:<owner/repo>``):
            repo-wide facts and the repo-wide briefing. Never every ``repo:`` fact.
    """

    def __init__(
        self,
        settings: PrefrontalConfig,
        cerebral: Cerebral,
        task_name: str,
        repo_full_name: str,
        scope_topic_ids: Sequence[str] | None = None,
        include_repo: bool = False,
    ) -> None:
        self._cfg = settings
        self._cerebral = cerebral
        self._task = task_name
        self._repo = repo_full_name
        self._scope_ids = [s for s in (scope_topic_ids or []) if s]
        self._include_repo = include_repo
        self._tool_calls = 0
        # One record per recall (pre-call load and each tool call), for monitoring.
        # Kept across aload() calls: one Prefrontal serves one agent call.
        self._records: list[RecallRecord] = []
        # Sections from the last aload() call, for structured memory rendering.
        self._last_sections: list[tuple[str, str]] = []
        # Inputs of the last render (briefings, facet results, remote facts),
        # kept so scoped_text() can re-render a per-scope subset without a recall.
        self._last_parts: tuple[list[dict], dict[str, FacetResult], list[Any]] | None = None

    @property
    def recalls(self) -> list[RecallRecord]:
        """Recall records so far (a copy). Persisted with the episode, never distilled."""
        return list(self._records)

    @property
    def last_sections(self) -> list[tuple[str, str]]:
        """Sections from the last aload() call as (title, body) tuples."""
        return list(self._last_sections)

    def scoped_text(self, scope_topic_id: str) -> str | None:
        """Re-render the last load for one scope, without a new recall.

        Keeps the briefings of ``scope_topic_id`` and of the repo-root scope,
        the facts tagged with either scope (facts without any ``project_scope:``
        tag are kept), the belief changes of kept observations, the shared
        "Around this work" synthesis and the remote facts (already filtered by
        the PR topic and the relevance floor).

        Returns None when nothing was loaded yet (callers fall back to the
        run-level text).
        """
        if self._last_parts is None:
            return None
        briefings, results, remote = self._last_parts
        keep_mm = set(mental_model_ids(self._repo, [scope_topic_id], True))
        keep_tags = set(local_tags(self._repo, [scope_topic_id], True))

        def in_scope(fact: Any) -> bool:
            scopes = [t for t in fget(fact, "tags") or [] if t.startswith(TAG_PROJECT_SCOPE)]
            return not scopes or any(t in keep_tags for t in scopes)

        kept_briefings = [mm for mm in briefings if not mm.get("id") or mm.get("id") in keep_mm]
        kept_obs = {fget(f, "id") for r in results.values() for f in r.facts if in_scope(f)}
        dropped_obs = {fget(f, "id") for r in results.values() for f in r.facts} - kept_obs
        scoped: dict[str, FacetResult] = {}
        for name, r in results.items():
            scoped[name] = dataclasses.replace(
                r,
                facts=[f for f in r.facts if in_scope(f)],
                changes=[c for c in r.changes if _change_observation_id(c) not in dropped_obs],
            )
        sections = format_context_sections(
            kept_briefings, scoped, remote, self._repo, self._cfg.recall.reach
        )
        return join_sections(sections)

    def _record(
        self,
        *,
        kind: Literal["load", "tool"],
        query: str,
        reach: str,
        started_at: datetime,
        t0: float,
        text: str,
        status: RecallStatus,
        usage: RecallUsage,
        details: dict[str, Any],
    ) -> RecallRecord | None:
        """Append one RecallRecord. Never raises."""
        try:
            rec = RecallRecord(
                task=self._task,
                kind=kind,
                timestamp=started_at,
                query=query,
                reach=reach,
                reflects=self._cfg.reflects,
                status=status,
                text=text,
                model=usage.to_model_field(),
                llm_calls=usage.llm_calls,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                input_cost=usage.input_cost,
                output_cost=usage.output_cost,
                latency_ms=int((time.monotonic() - t0) * 1000),
                details=details,
            )
            self._records.append(rec)
            return rec
        except Exception:
            logger.debug("Prefrontal[%s]: recall record failed", self._task, exc_info=True)
            return None

    # Remote share constant: fraction of token budget for other repos when reach is org/bank
    REMOTE_SHARE = 0.25
    # Recall budget constant: fixed at "mid" for raw recall
    RECALL_BUDGET = "mid"

    @property
    def reach(self) -> str:
        return self._cfg.recall.reach

    @property
    def reflects(self) -> int:
        return self._cfg.reflects

    def _split_tokens(self, total: int, reach: str) -> tuple[int, int]:
        """Split a token budget into (local, remote) by ``REMOTE_SHARE``.

        The remote share only applies when ``reach`` has a remote pool; otherwise
        the local pool gets the whole budget.
        """
        if build_remote_groups(self._repo, reach) is None:
            return total, 0
        remote = int(total * self.REMOTE_SHARE)
        return total - remote, remote

    def _reflect_context(self) -> str:
        if len(self._scope_ids) == 1 and not self._include_repo:
            target = self._scope_ids[0]
            if not target.startswith(self._repo):
                target = f"{self._repo}/{target}"
        else:
            target = self._repo
        return f"{self._task} on {target}"

    def _score_entries(
        self, section: str, scored: Sequence[tuple[Any, bool]]
    ) -> list[dict[str, Any]]:
        """Score records for facts before the relevance floor; logs one INFO line each.

        Used to calibrate ``memory.prefrontal.recall.min_similarity``.
        """
        entries: list[dict[str, Any]] = []
        for fact, kept in scored:
            score = semantic_score(fact)
            repo = tag_value(fget(fact, "tags") or [], TAG_REPO)
            entries.append(
                {
                    "id": fget(fact, "id"),
                    "section": section,
                    "repo": repo,
                    "semantic": round(score, 3) if score is not None else None,
                    "kept": kept,
                }
            )
            logger.info(
                "Prefrontal[%s] score section=%s semantic=%s kept=%s repo=%s text=%s",
                self._task,
                section,
                f"{score:.3f}" if score is not None else "none",
                "y" if kept else "n",
                repo or "-",
                " ".join((fget(fact, "text") or "").split())[:LOG_SCORE_TEXT_CHARS],
            )
        return entries

    def _min_semantic_kwargs(self) -> dict[str, Any]:
        floor = self._cfg.recall.min_similarity
        return {"min_semantic": floor} if floor > 0 else {}

    # ------------------------------------------------------------------
    # Pre-call context
    # ------------------------------------------------------------------

    def load(self, facets: Sequence[Facet]) -> str:
        """Blocking variant of :meth:`aload` (for the sync ChainOfThought agents)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aload(facets))
        # A loop is running in this thread: run on a helper thread instead.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.aload(facets)).result()

    async def aload(self, facets: Sequence[Facet]) -> str:
        """Build the pre-call context: briefings, facets 1 → 5, then remote facts.

        Records one ``kind="load"`` RecallRecord. Usage tracking is entered here,
        inside the coroutine, so it also works for :meth:`load`, whose helper
        thread does not copy the caller's context.
        """
        self._tool_calls = 0
        self._last_sections = []
        self._last_parts = None
        facet_list = list(facets)
        context_query = next((f.query for f in facet_list if f.name == FACET_CONTEXT), "")
        started_at = datetime.now(UTC)
        t0 = time.monotonic()
        text = ""
        status: RecallStatus = "error"
        details: dict[str, Any] = {}
        sections: list[tuple[str, str]] = []
        with track_recall_usage() as usage:
            try:
                text, details, sections = await self._aload(facet_list)
                self._last_sections = sections
                status = "ok" if text else "empty"
            except Exception:
                logger.warning("Prefrontal[%s]: load failed", self._task, exc_info=True)
                text = ""
        rec = self._record(
            kind="load",
            query=context_query,
            reach=self._cfg.recall.reach,
            started_at=started_at,
            t0=t0,
            text=text,
            status=status,
            usage=usage,
            details=details,
        )
        facet_counts = details.get("facets", {})

        def _n(name: str) -> int:
            return int((facet_counts.get(name) or {}).get("count", 0))

        def _d(name: str) -> int:
            return int((facet_counts.get(name) or {}).get("dropped", 0))

        # Reflect signals for monitoring split synthesis and quality
        reflect_info = details.get("reflect") or {}
        maps = reflect_info.get("map_calls", 0)
        rewrite = "rewrite" if reflect_info.get("rewrite") else ""
        thoughts = reflect_info.get("usage", {}).get("thoughts_tokens", 0)
        llm_calls = usage.llm_calls
        expected_max = self._cfg.reflects + 1  # +1 for potential rewrite

        # Warn if split synthesis occurred (map_calls > 0) or calls exceeded expected max
        if maps > 0:
            logger.warning("Prefrontal[%s]: split synthesis detected (maps=%d)", self._task, maps)
        if llm_calls > expected_max:
            logger.warning(
                "Prefrontal[%s]: reflect used %d LLM calls (> %d): maps=%d rewrite=%s tools=%s",
                self._task,
                llm_calls,
                expected_max,
                maps,
                rewrite or "no",
                ",".join(f"{name}:{size}" for name, size in reflect_info.get("tools", [])) or "-",
            )

        # Per-tool sizes for context calibration (e.g., "search_observations:1830,recall:2410")
        tool_sizes = reflect_info.get("tools", [])
        tools_str = ",".join(f"{name}:{size}" for name, size in tool_sizes) if tool_sizes else "-"

        logger.info(
            "Prefrontal[%s]: load status=%s %d briefings, facets context=%d seen_before=%d "
            "decisions=%d patterns=%d belief_changes=%d, remote=%d (%d tokens, reach=%s, "
            "reflects=%d) dropped remote=%d seen_before=%d decisions=%d patterns=%d "
            "model=%s calls=%d in=%d out=%d cost=$%.4f latency=%dms maps=%d%s thoughts=%d tools=%s",
            self._task,
            status,
            int(details.get("briefings", 0)),
            _n(FACET_CONTEXT),
            _n(FACET_SEEN_BEFORE),
            _n(FACET_DECISIONS),
            _n(FACET_PATTERNS),
            _n(FACET_BELIEF_CHANGES),
            int(details.get("remote", 0)),
            estimate_tokens(text),
            self._cfg.recall.reach,
            self._cfg.reflects,
            int(details.get("remote_dropped", 0)),
            _d(FACET_SEEN_BEFORE),
            _d(FACET_DECISIONS),
            _d(FACET_PATTERNS),
            usage.to_model_field() or "-",
            llm_calls,
            usage.input_tokens,
            usage.output_tokens,
            usage.input_cost + usage.output_cost,
            rec.latency_ms if rec else int((time.monotonic() - t0) * 1000),
            maps,
            f" {rewrite}" if rewrite else "",
            thoughts,
            tools_str,
        )
        return text

    async def _aload(self, facets: list[Facet]) -> tuple[str, dict[str, Any], list[tuple[str, str]]]:
        """Return the rendered context, load ``details``, and sections for the RecallRecord."""
        cfg = self._cfg
        now = datetime.now(UTC)
        local_groups = build_local_groups(self._repo, self._scope_ids, self._include_repo)
        ctx_facet = next((f for f in facets if f.name == FACET_CONTEXT), None)
        context_query = ctx_facet.query if ctx_facet else ""
        # Remote pool: the PR topic only (no task, repo/scope id or paths, which
        # match every repo lexically). Falls back to the context query.
        remote_query = (ctx_facet.remote_query if ctx_facet else "") or context_query
        floor = cfg.recall.min_similarity
        local_tokens, remote_tokens = self._split_tokens(
            cfg.recall.max_tokens, cfg.recall.reach
        )
        remote_scored: list[tuple[Any, bool]] = []

        async def _briefings() -> list[dict]:
            if cfg.reflects <= 0:
                return []
            ids = mental_model_ids(self._repo, self._scope_ids, self._include_repo)
            if not ids:
                return []
            try:
                return await self._cerebral.aget_mental_models(ids)
            except Exception:
                logger.warning("Prefrontal[%s]: briefings failed", self._task, exc_info=True)
                return []

        async def _facets() -> dict:
            if not local_groups:
                return {}
            return await run_facets(
                self._cerebral,
                facets,
                tag_groups=local_groups,
                max_tokens=local_tokens,
                budget="mid",
                question_date=now,
                reflects=cfg.reflects,
                reflect_context=self._reflect_context(),
                min_similarity=floor,
            )

        def _extract_reflect_summary(results: dict) -> dict | None:
            """Extract reflect summary from context facet for monitoring."""
            from codespy.agents.memory.prefrontal.query import FACET_CONTEXT

            ctx = results.get(FACET_CONTEXT)
            if ctx is None or ctx.reflect_summary is None:
                return None
            s = ctx.reflect_summary
            return {
                "iterations": s.iterations,
                "llm_calls": s.llm_calls,
                "map_calls": s.map_calls,
                "rewrite": s.rewrite,
                "tools": s.tools,
                "usage": s.usage,
                "empty": s.empty,
            }

        async def _remote() -> list[Any]:
            groups = build_remote_groups(self._repo, cfg.recall.reach)
            if groups is None or remote_tokens <= 0 or not remote_query:
                return []
            try:
                facts = await self._cerebral.arecall(
                    remote_query,
                    tag_groups=groups,
                    max_tokens=remote_tokens,
                    fact_type=["world", "observation"],
                    prefer_observations=True,
                    budget=self.RECALL_BUDGET,
                    question_date=now,
                    **self._min_semantic_kwargs(),
                )
            except Exception:
                logger.warning("Prefrontal[%s]: remote recall failed", self._task, exc_info=True)
                return []
            facts = [f for f in facts or [] if not is_own_repo_fact(fget(f, "tags"), self._repo)]
            kept, dropped = filter_relevant(facts, floor)
            dropped_ids = {id(f) for f in dropped}
            remote_scored.extend((f, id(f) not in dropped_ids) for f in facts)
            return kept

        briefings, results, remote = await asyncio.gather(_briefings(), _facets(), _remote())
        self._last_parts = (list(briefings or []), dict(results or {}), list(remote or []))
        # Compute sections once for structured rendering
        sections = format_context_sections(
            briefings, results, remote, self._repo, cfg.recall.reach
        )
        text = join_sections(sections)

        n_briefings = sum(
            1
            for mm in briefings
            if (mm.get("content") or "").strip()
        )
        facet_details: dict[str, dict[str, Any]] = {}
        for f in facets:
            r = results.get(f.name)
            facet_details[f.name] = {
                "query": f.query,
                "count": r.count if r is not None else 0,
                "failed": bool(r.failed) if r is not None else False,
                "dropped": r.dropped if r is not None else 0,
            }
        scores: list[dict[str, Any]] = []
        for name in SCORED_FACETS:
            r = results.get(name)
            if r is not None:
                scores.extend(self._score_entries(name, r.scored))
        scores.extend(self._score_entries(SECTION_REMOTE, remote_scored))
        details: dict[str, Any] = {
            "briefings": n_briefings,
            "remote": len(remote),
            "remote_query": remote_query,
            "remote_dropped": sum(1 for _, kept in remote_scored if not kept),
            "remote_scores": [
                round(s, 3) for s in (semantic_score(f) for f in remote) if s is not None
            ],
            "facets": facet_details,
            "reflect": _extract_reflect_summary(results),
            "scores": scores,
        }
        return text, details, sections

    # ------------------------------------------------------------------
    # Tool
    # ------------------------------------------------------------------

    def recall_tool(self) -> Callable[..., Any] | None:
        """Async ``recall_memory`` tool for RLM agents, or None when disabled."""
        if self._cfg.recall.max_tool_calls <= 0:
            return None

        async def recall_memory(query: str, reach: str = "local") -> str:
            """Search prior knowledge learned in earlier runs (read-only, may be stale or wrong).

            Args:
                query: What to look up, e.g. a file, symbol, package, problem or decision.
                reach: "local" (this scope/repository), "org" (also other repositories
                    of the same owner) or "bank" (all repositories). Clamped to the
                    configured reach.

            Returns:
                Matching facts as text. Verify with tools before reporting any issue.
            """
            return await self._recall(query, reach)

        return recall_memory

    async def _recall(self, query: str, reach: str = "local") -> str:
        """Run one recall_memory call and record it (every exit path, exactly once)."""
        cfg = self._cfg
        started_at = datetime.now(UTC)
        t0 = time.monotonic()
        # str(): RLM agents may pass non-string arguments; this must never raise.
        q = truncate_words(str(query or ""))
        effective = clamp_reach(str(reach or ""), cfg.recall.reach)
        details: dict[str, Any] = {
            "requested_reach": str(reach or ""),
            "mode": "reflect" if cfg.reflects > 0 else "recall",
        }
        with track_recall_usage() as usage:
            if self._tool_calls >= cfg.recall.max_tool_calls:
                text = RECALL_LIMIT_REACHED
            else:
                self._tool_calls += 1
                text = await self._recall_impl(q, effective, details)
        status = _tool_status(text)
        rec = self._record(
            kind="tool",
            query=q,
            reach=effective,
            started_at=started_at,
            t0=t0,
            text=text,
            status=status,
            usage=usage,
            details=details,
        )
        logger.info(
            "Prefrontal[%s]: recall_memory reach=%s status=%s mode=%s model=%s calls=%d "
            "in=%d out=%d cost=$%.4f latency=%dms query=%r",
            self._task,
            effective,
            status,
            details["mode"],
            usage.to_model_field() or "-",
            usage.llm_calls,
            usage.input_tokens,
            usage.output_tokens,
            usage.input_cost + usage.output_cost,
            rec.latency_ms if rec else int((time.monotonic() - t0) * 1000),
            q[:LOG_QUERY_CHARS],
        )
        return text

    async def _recall_impl(self, q: str, effective: str, details: dict[str, Any]) -> str:
        """recall_memory body. Fills fact counts into ``details`` (recall mode)."""
        cfg = self._cfg
        try:
            if not q:
                return NO_RELEVANT_MEMORY
            now = datetime.now(UTC)

            if cfg.reflects > 0:
                groups = build_reflect_groups(
                    self._repo, self._scope_ids, self._include_repo, effective
                )
                if groups == []:
                    return NO_RELEVANT_MEMORY
                prompt = q if effective == REACH_LOCAL else q + REMOTE_REFLECT_SUFFIX
                text, summary = await self._cerebral.areflect(
                    prompt,
                    tag_groups=groups,
                    max_tokens=cfg.recall.max_tool_tokens,
                    context=self._reflect_context(),
                )
                details["reflect"] = {
                    "iterations": summary.iterations,
                    "llm_calls": summary.llm_calls,
                    "map_calls": summary.map_calls,
                    "rewrite": summary.rewrite,
                    "tools": summary.tools,
                    "usage": summary.usage,
                    "empty": summary.empty,
                }
                # Warn if calls exceeded expected max (reflects + 1 for potential rewrite)
                expected_max = cfg.reflects + 1
                if summary.llm_calls > expected_max:
                    logger.warning(
                        "Prefrontal[%s]: reflect used %d LLM calls (> %d): maps=%d rewrite=%s tools=%s",
                        self._task,
                        summary.llm_calls,
                        expected_max,
                        summary.map_calls,
                        "yes" if summary.rewrite else "no",
                        ",".join(f"{name}:{size}" for name, size in summary.tools) or "-",
                    )
                # Warn if split synthesis occurred
                if summary.map_calls > 0:
                    logger.warning(
                        "Prefrontal[%s]: split synthesis detected (maps=%d)",
                        self._task,
                        summary.map_calls,
                    )
                return text or NO_RELEVANT_MEMORY

            parts: list[str] = []
            local_tokens, remote_tokens = self._split_tokens(
                cfg.recall.max_tool_tokens, effective
            )
            local_groups = build_local_groups(self._repo, self._scope_ids, self._include_repo)
            if local_groups:
                facts = await self._cerebral.arecall(
                    q,
                    tag_groups=local_groups,
                    max_tokens=local_tokens,
                    prefer_observations=True,
                    budget=self.RECALL_BUDGET,
                    question_date=now,
                )
                details["local"] = len(facts or [])
                parts.append(format_facts(facts))
            remote_groups = build_remote_groups(self._repo, effective)
            if remote_groups is not None and remote_tokens > 0:
                facts = await self._cerebral.arecall(
                    q,
                    tag_groups=remote_groups,
                    max_tokens=remote_tokens,
                    fact_type=["world", "observation"],
                    prefer_observations=True,
                    budget=self.RECALL_BUDGET,
                    question_date=now,
                    **self._min_semantic_kwargs(),
                )
                facts = [
                    f for f in facts or [] if not is_own_repo_fact(fget(f, "tags"), self._repo)
                ]
                kept, dropped = filter_relevant(facts, cfg.recall.min_similarity)
                dropped_ids = {id(f) for f in dropped}
                details["scores"] = self._score_entries(
                    SECTION_TOOL_REMOTE, [(f, id(f) not in dropped_ids) for f in facts]
                )
                facts = kept
                details["remote"] = len(facts)
                details["remote_dropped"] = len(dropped)
                rendered = format_facts(facts, with_repo=True)
                if rendered:
                    parts.append(f"{REMOTE_TOOL_HEADER}\n{rendered}")
            text = "\n\n".join(p for p in parts if p)
            return text or NO_RELEVANT_MEMORY
        except Exception:
            logger.warning("Prefrontal[%s]: recall_memory failed", self._task, exc_info=True)
            return MEMORY_UNAVAILABLE
