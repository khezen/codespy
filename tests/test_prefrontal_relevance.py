"""Prefrontal relevance floor (memory.prefrontal.recall.min_similarity).

Fake Cerebral, no database: facts are dicts carrying ``scores.semantic``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from codespy.agents.memory.prefrontal.facets import run_facets
from codespy.agents.memory.prefrontal.format import filter_relevant, semantic_score
from codespy.agents.memory.prefrontal.prefrontal import Prefrontal
from codespy.agents.memory.prefrontal.query import (
    FACET_BELIEF_CHANGES,
    FACET_CONTEXT,
    FACET_DECISIONS,
    FACET_PATTERNS,
    FACET_SEEN_BEFORE,
    MAX_QUERY_WORDS,
    build_facets,
)
from codespy.agents.memory.prefrontal.reach import build_remote_groups
from codespy.config_memory import PrefrontalConfig, RecallConfig

REPO = "acme/mono"
SCOPE = "acme/mono/svc/pension"
DECISIONS_PHRASE = (
    "decisions, trade-offs, chosen approaches, rejected alternatives, agreed conventions"
)
PATTERNS_PHRASE = "recurring issues, repeated findings, patterns across changes"


def fact(fid: str, semantic: float | None, *, fact_type="world", repo=REPO, text=None) -> dict:
    return {
        "id": fid,
        "text": text or f"fact {fid}",
        "fact_type": fact_type,
        "tags": [f"repo:{repo}", "task:code_review"],
        "scores": {"final": 0.5, "semantic": semantic, "keyword": None, "reranker": None},
    }


def facets(**overrides: Any):
    kw: dict[str, Any] = {
        "task": "code_review",
        "target": SCOPE,
        "description": "Pension switcher service",
        "pr_title": "Add drafts and sorting to switch requests",
        "paths": ["svc/pension/db.request.go", "web/constants.ts"],
        "summary": "Switch requests can be saved as drafts and listed by date.",
        "packages": ["pension-svc"],
    }
    kw.update(overrides)
    return build_facets(**kw)


class FakeCerebral:
    def __init__(self, by_facet: dict[str, list], remote: list | None = None, history=None):
        self.by_facet = by_facet
        self.remote = remote or []
        self.history = history or {}
        self.calls: list[dict[str, Any]] = []
        self.history_calls: list[str] = []
        self._facets: list = []
        self._remote_groups = build_remote_groups(REPO, "org")

    def bind(self, facet_list):
        self._facets = list(facet_list)
        return self

    async def arecall(self, query, **kwargs):
        is_remote = kwargs.get("tag_groups") == self._remote_groups
        name = (
            "remote"
            if is_remote
            else next((f.name for f in self._facets if f.query == query), "tool_local")
        )
        self.calls.append({"name": name, "query": query, **kwargs})
        if is_remote:
            return list(self.remote)
        return list(self.by_facet.get(name, []))

    async def areflect(self, *a, **k):  # pragma: no cover - reflects=0 in these tests
        raise AssertionError("reflect not expected")

    async def aget_mental_models(self, ids):
        return []

    async def aget_observation_history(self, oid):
        self.history_calls.append(oid)
        return self.history.get(oid, [])

    def call(self, name):
        return next(c for c in self.calls if c["name"] == name)


def prefrontal(cerebral, floor: float, *, reach="org", tool_calls=0) -> Prefrontal:
    cfg = PrefrontalConfig(
        reflects=0,
        recall=RecallConfig(min_similarity=floor, reach=reach, max_tool_calls=tool_calls),
    )
    return Prefrontal(cfg, cerebral, "code_review", REPO, scope_topic_ids=[SCOPE])


# ---------------------------------------------------------------------------
# format helpers
# ---------------------------------------------------------------------------


class TestFilterRelevant:
    def test_semantic_score_reads_dict_and_object(self):
        class S:
            semantic = 0.42

        assert semantic_score(fact("a", 0.7)) == 0.7
        assert semantic_score({"scores": S()}) == 0.42
        assert semantic_score({"id": "x"}) is None
        assert semantic_score(fact("b", None)) is None

    def test_floor_drops_below_and_none(self):
        facts_ = [fact("a", 0.8), fact("b", 0.3), fact("c", None), fact("d", 0.5)]
        kept, dropped = filter_relevant(facts_, 0.5)
        assert [f["id"] for f in kept] == ["a", "d"]
        assert [f["id"] for f in dropped] == ["b", "c"]

    def test_zero_floor_keeps_everything(self):
        facts_ = [fact("a", 0.1), fact("c", None)]
        kept, dropped = filter_relevant(facts_, 0)
        assert kept == facts_ and dropped == []


# ---------------------------------------------------------------------------
# build_facets
# ---------------------------------------------------------------------------


class TestBuildFacets:
    def _by_name(self, fl):
        return {f.name: f for f in fl}

    def test_context_remote_query_is_topic_only(self):
        ctx = self._by_name(facets())[FACET_CONTEXT]
        assert "Add drafts and sorting" in ctx.remote_query
        assert "saved as drafts" in ctx.remote_query
        assert "Files:" not in ctx.remote_query
        assert "db.request.go" not in ctx.remote_query
        assert SCOPE not in ctx.remote_query
        assert "code_review" not in ctx.remote_query
        assert ctx.relevance_floor is False
        # The local context query keeps its old shape.
        assert ctx.query.startswith(f"code_review {SCOPE}")
        assert "Files:" in ctx.query

    def test_seen_before_uses_paths_and_symbols(self):
        sb = self._by_name(facets(symbols=["ListDrafts"]))[FACET_SEEN_BEFORE]
        assert sb.query == "svc/pension/db.request.go web/constants.ts ListDrafts"
        assert sb.relevance_floor is True

    def test_seen_before_fallback_without_entities(self):
        sb = self._by_name(facets(paths=[], symbols=[]))[FACET_SEEN_BEFORE]
        assert sb.query == f"{SCOPE} pension-svc"
        assert sb.relevance_floor is False

    @pytest.mark.parametrize(
        "name,phrase", [(FACET_DECISIONS, DECISIONS_PHRASE), (FACET_PATTERNS, PATTERNS_PHRASE)]
    )
    def test_topic_queries(self, name, phrase):
        f = self._by_name(facets())[name]
        assert f.query.startswith("Pension switcher service. PR: Add drafts")
        assert f.query.endswith(phrase)
        assert f"{SCOPE}:" not in f.query
        assert f.relevance_floor is True

    @pytest.mark.parametrize(
        "name,phrase", [(FACET_DECISIONS, DECISIONS_PHRASE), (FACET_PATTERNS, PATTERNS_PHRASE)]
    )
    def test_topic_queries_keep_phrasing_with_long_topic(self, name, phrase):
        long = " ".join(f"word{i}" for i in range(300))
        f = self._by_name(facets(description=long, pr_title=long, summary=long))[name]
        assert f.query.endswith(phrase)
        assert len(f.query.split()) <= MAX_QUERY_WORDS

    @pytest.mark.parametrize(
        "name,phrase", [(FACET_DECISIONS, DECISIONS_PHRASE), (FACET_PATTERNS, PATTERNS_PHRASE)]
    )
    def test_topic_fallback(self, name, phrase):
        f = self._by_name(facets(description="", pr_title="", summary=""))[name]
        assert f.query == f"{SCOPE}: {phrase}"
        assert f.relevance_floor is False


# ---------------------------------------------------------------------------
# run_facets
# ---------------------------------------------------------------------------


def _run(cerebral, fl, floor):
    return asyncio.run(
        run_facets(
            cerebral.bind(fl),
            fl,
            tag_groups=[{"tags": ["x"]}],
            max_tokens=6000,
            reflects=0,
            min_similarity=floor,
        )
    )


class TestRunFacets:
    def test_floored_facets_filtered_and_counted(self):
        by = {
            FACET_SEEN_BEFORE: [fact("s1", 0.7), fact("s2", 0.2)],
            FACET_DECISIONS: [fact("d1", None), fact("d2", 0.6)],
            FACET_PATTERNS: [fact("p1", 0.1, fact_type="observation")],
        }
        cer = FakeCerebral(by)
        fl = facets()
        res = _run(cer, fl, 0.5)
        assert [f["id"] for f in res[FACET_SEEN_BEFORE].facts] == ["s1"]
        assert [f["id"] for f in res[FACET_DECISIONS].facts] == ["d2"]
        assert res[FACET_PATTERNS].facts == []
        assert res[FACET_SEEN_BEFORE].dropped == 1
        assert res[FACET_DECISIONS].dropped == 1
        assert res[FACET_PATTERNS].dropped == 1
        for name in (FACET_SEEN_BEFORE, FACET_DECISIONS, FACET_PATTERNS):
            assert cer.call(name)["min_semantic"] == 0.5

    def test_context_never_floored(self):
        cer = FakeCerebral({FACET_CONTEXT: [fact("c1", 0.1), fact("c2", None)]})
        res = _run(cer, facets(), 0.5)
        assert [f["id"] for f in res[FACET_CONTEXT].facts] == ["c1", "c2"]
        assert "min_semantic" not in cer.call(FACET_CONTEXT)

    def test_fallback_queries_not_floored(self):
        by = {
            FACET_SEEN_BEFORE: [fact("s1", 0.1)],
            FACET_DECISIONS: [fact("d1", None)],
            FACET_PATTERNS: [fact("p1", 0.1, fact_type="observation")],
        }
        cer = FakeCerebral(by)
        fl = facets(paths=[], symbols=[], description="", pr_title="", summary="")
        res = _run(cer, fl, 0.5)
        for name in (FACET_SEEN_BEFORE, FACET_DECISIONS, FACET_PATTERNS):
            assert len(res[name].facts) == 1
            assert res[name].dropped == 0
            assert "min_semantic" not in cer.call(name)

    def test_zero_floor_keeps_everything(self):
        by = {
            FACET_SEEN_BEFORE: [fact("s1", 0.01), fact("s2", None)],
            FACET_DECISIONS: [fact("d1", None)],
        }
        cer = FakeCerebral(by)
        res = _run(cer, facets(), 0.0)
        assert len(res[FACET_SEEN_BEFORE].facts) == 2
        assert len(res[FACET_DECISIONS].facts) == 1
        assert all("min_semantic" not in c for c in cer.calls)

    def test_belief_changes_only_from_kept(self):
        by = {
            FACET_DECISIONS: [
                fact("o_keep", 0.9, fact_type="observation"),
                fact("o_drop", 0.1, fact_type="observation"),
                fact("r_drop", 0.1, text="RETRACTED: old belief"),
                fact("r_keep", 0.8, text="RETRACTED: kept belief"),
            ],
        }
        history = {
            "o_keep": [{"previous_text": "before", "changed_at": "2026-01-01"}],
            "o_drop": [{"previous_text": "irrelevant", "changed_at": "2026-01-01"}],
        }
        cer = FakeCerebral(by, history=history)
        res = _run(cer, facets(), 0.5)
        assert cer.history_calls == ["o_keep"]
        moved = [f["id"] for f in res[FACET_BELIEF_CHANGES].facts]
        assert moved == ["r_keep"]
        assert [c.previous for c in res[FACET_BELIEF_CHANGES].changes] == ["before"]


# ---------------------------------------------------------------------------
# Prefrontal load (remote) and recall_memory
# ---------------------------------------------------------------------------


OTHER = "acme/ios"


class TestPrefrontalLoad:
    def _load(self, cer, floor, fl=None):
        fl = fl or facets()
        pf = prefrontal(cer.bind(fl), floor)
        text = asyncio.run(pf.aload(fl))
        return pf, text

    def test_remote_filtered_and_rendered(self):
        remote = [
            fact("r1", 0.8, repo=OTHER, text="pension drafts are soft-deleted"),
            fact("r2", 0.2, repo=OTHER, text="ios navigation bar colours"),
            fact("r3", None, repo=OTHER, text="ios keyword hit"),
        ]
        cer = FakeCerebral({}, remote=remote)
        pf, text = self._load(cer, 0.5)
        assert "pension drafts are soft-deleted" in text
        assert "ios navigation" not in text and "ios keyword hit" not in text
        details = pf.recalls[-1].details
        assert details["remote"] == 1
        assert details["remote_dropped"] == 2
        assert details["remote_scores"] == [0.8]

    def test_remote_section_omitted_when_none_pass(self):
        cer = FakeCerebral({}, remote=[fact("r2", 0.2, repo=OTHER)])
        _, text = self._load(cer, 0.5)
        assert "Other repositories" not in text

    def test_remote_query_is_topic_and_min_semantic(self):
        cer = FakeCerebral({}, remote=[])
        self._load(cer, 0.5)
        call = cer.call("remote")
        assert call["min_semantic"] == 0.5
        assert "Files:" not in call["query"]
        assert not call["query"].startswith("code_review")
        assert SCOPE not in call["query"]
        assert "Add drafts and sorting" in call["query"]

    def test_zero_floor_keeps_remote(self):
        cer = FakeCerebral({}, remote=[fact("r2", 0.2, repo=OTHER), fact("r3", None, repo=OTHER)])
        pf, text = self._load(cer, 0.0)
        assert "Other repositories" in text
        assert "min_semantic" not in cer.call("remote")
        assert pf.recalls[-1].details["remote_dropped"] == 0

    def test_facet_dropped_in_details(self):
        cer = FakeCerebral({FACET_SEEN_BEFORE: [fact("s1", 0.7), fact("s2", 0.1)]})
        pf, _ = self._load(cer, 0.5)
        facets_d = pf.recalls[-1].details["facets"]
        assert facets_d[FACET_SEEN_BEFORE]["dropped"] == 1
        assert facets_d[FACET_SEEN_BEFORE]["count"] == 1

    def test_score_logging_lists_dropped_facts(self, caplog):
        cer = FakeCerebral(
            {
                FACET_CONTEXT: [fact("c1", 0.1)],
                FACET_SEEN_BEFORE: [fact("s1", 0.7), fact("s2", 0.1)],
                FACET_PATTERNS: [fact("p1", None, fact_type="observation")],
            },
            remote=[fact("r1", 0.9, repo=OTHER), fact("r2", 0.3, repo=OTHER)],
        )
        with caplog.at_level("INFO", logger="codespy.agents.memory.prefrontal.prefrontal"):
            pf, _ = self._load(cer, 0.5)
        scores = pf.recalls[-1].details["scores"]
        by_id = {s["id"]: s for s in scores}
        assert set(by_id) == {"s1", "s2", "p1", "r1", "r2"}  # context is not scored
        assert by_id["s1"] == {
            "id": "s1",
            "section": FACET_SEEN_BEFORE,
            "repo": REPO,
            "semantic": 0.7,
            "kept": True,
        }
        assert by_id["s2"]["kept"] is False
        assert by_id["p1"]["semantic"] is None and by_id["p1"]["kept"] is False
        assert by_id["r1"]["section"] == "remote" and by_id["r1"]["repo"] == OTHER
        assert by_id["r2"]["kept"] is False
        lines = [r.getMessage() for r in caplog.records if " score section=" in r.getMessage()]
        assert len(lines) == 5
        assert any("section=remote semantic=0.300 kept=n repo=acme/ios" in m for m in lines)
        assert any("section=patterns semantic=none kept=n" in m for m in lines)

    def test_score_logging_with_zero_floor_keeps_all(self):
        cer = FakeCerebral(
            {FACET_DECISIONS: [fact("d1", 0.05)]}, remote=[fact("r", None, repo=OTHER)]
        )
        pf, _ = self._load(cer, 0.0)
        scores = pf.recalls[-1].details["scores"]
        assert {s["id"] for s in scores} == {"d1", "r"}
        assert all(s["kept"] for s in scores)


class TestRecallMemoryRaw:
    def test_remote_branch_filtered(self):
        remote = [
            fact("r1", 0.8, repo=OTHER, text="relevant remote"),
            fact("r2", 0.1, repo=OTHER, text="irrelevant remote"),
        ]
        cer = FakeCerebral(
            {"tool_local": [fact("l1", 0.05, text="local low score")]}, remote=remote
        )
        pf = prefrontal(cer, 0.5, tool_calls=1)
        text = asyncio.run(pf._recall("drafts", "org"))
        assert "relevant remote" in text
        assert "irrelevant remote" not in text
        # Tool local recall is not floored.
        assert "local low score" in text
        assert "min_semantic" not in cer.call("tool_local")
        assert cer.call("remote")["min_semantic"] == 0.5
        assert cer.call("remote")["query"] == "drafts"
        details = pf.recalls[-1].details
        assert details["remote"] == 1
        assert details["remote_dropped"] == 1
        assert {s["id"]: s["kept"] for s in details["scores"]} == {"r1": True, "r2": False}
        assert all(s["section"] == "tool_remote" for s in details["scores"])
