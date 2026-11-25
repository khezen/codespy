"""Per-scope Prefrontal memory: re-render the run-level load for one scope."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from codespy.agents.memory.prefrontal.format import BeliefChange, FacetResult
from codespy.agents.memory.prefrontal.prefrontal import Prefrontal
from codespy.agents.memory.prefrontal.query import (
    FACET_BELIEF_CHANGES,
    FACET_CONTEXT,
    FACET_DECISIONS,
    FACET_SEEN_BEFORE,
)
from codespy.agents.memory.prefrontal.reach import mental_model_ids, scope_tags
from codespy.agents.review.models import PRContext, ReviewContext
from codespy.agents.rlm_tools import memory_sections
from codespy.workflows.review.pipeline import scoped_memory

REPO = "acme/mono"
A = f"{REPO}/svc/a"
B = f"{REPO}/svc/b"


def _fact(fid: str, scope: str | None, text: str, fact_type: str = "observation") -> dict:
    tags = [f"repo:{REPO}"] + ([f"project_scope:{scope}"] if scope else [])
    return {"id": fid, "text": text, "tags": tags, "fact_type": fact_type}


def _mm(scope: str, content: str) -> dict:
    return {
        "id": mental_model_ids(REPO, [scope], False)[0],
        "name": f"Briefing: {scope}",
        "content": content,
        "tags": scope_tags(REPO, scope),
    }


def _pf() -> Prefrontal:
    cfg = SimpleNamespace(recall=SimpleNamespace(reach="org"), reflects=3)
    pf = Prefrontal(cfg, MagicMock(), "review", REPO, [A, B], include_repo=True)
    results = {
        FACET_CONTEXT: FacetResult(FACET_CONTEXT, text="# Review\n## Summary\nShared synthesis."),
        FACET_SEEN_BEFORE: FacetResult(
            FACET_SEEN_BEFORE,
            facts=[
                _fact("a1", A, "a fact one"),
                _fact("b1", B, "b fact one"),
                _fact("r1", REPO, "root fact one"),
                _fact("u1", None, "untagged fact"),
            ],
        ),
        FACET_DECISIONS: FacetResult(
            FACET_DECISIONS, facts=[_fact("a2", A, "a decision"), _fact("b2", B, "b decision")]
        ),
        FACET_BELIEF_CHANGES: FacetResult(
            FACET_BELIEF_CHANGES,
            changes=[
                BeliefChange(id="history:a1:0", current="a fact one", previous="a old"),
                BeliefChange(id="history:b1:0", current="b fact one", previous="b old"),
            ],
        ),
    }
    briefings = [_mm(A, "A conventions"), _mm(B, "B conventions"), _mm(REPO, "Root conventions")]
    remote = [_fact("x1", "acme/other", "remote fact", "world")]
    remote[0]["tags"] = ["repo:acme/other"]
    pf._last_parts = (briefings, results, remote)
    return pf


class TestScopedText:
    def test_none_before_load(self):
        cfg = SimpleNamespace(recall=SimpleNamespace(reach="org"), reflects=3)
        assert Prefrontal(cfg, MagicMock(), "review", REPO).scoped_text(A) is None

    def test_keeps_own_and_root_only(self):
        text = _pf().scoped_text(A)
        assert "A conventions" in text and "Root conventions" in text
        assert "B conventions" not in text
        assert "a fact one" in text and "a decision" in text and "root fact one" in text
        assert "b fact one" not in text and "b decision" not in text
        assert "untagged fact" in text
        assert "Shared synthesis." in text
        assert "remote fact" in text

    def test_belief_changes_follow_observation(self):
        text = _pf().scoped_text(A)
        assert "a old" in text and "b old" not in text

    def test_smaller_than_run_level(self):
        pf = _pf()
        briefings, results, remote = pf._last_parts
        from codespy.agents.memory.prefrontal.format import format_context

        full = format_context(briefings, results, remote, REPO)
        scoped = pf.scoped_text(A)
        assert len(scoped) < len(full)
        assert [t for t, *_ in memory_sections(scoped)][0] == f"Briefing: {A}"

    def test_does_not_mutate_load(self):
        pf = _pf()
        pf.scoped_text(A)
        assert len(pf._last_parts[1][FACET_SEEN_BEFORE].facts) == 4
        assert "b fact one" in pf.scoped_text(B)


class TestReviewContext:
    def test_memory_for_scope(self):
        ctx = ReviewContext(
            pr_context=PRContext(
                repo_full_name=REPO, repo_slug="mono", pr_url="u", pr_number=1,
                pr_title="t", summary="s",
            ),
            prefrontal_memory="run level",
            prefrontal_scoped={A: "scoped a", B: ""},
        )
        assert ctx.memory_for_scope(A) == "scoped a"
        assert ctx.memory_for_scope(B) == ""
        assert ctx.memory_for_scope("other") == "run level"


class TestScopedMemory:
    def _scope(self, topic_id: str, subroot: str):
        scope = MagicMock(subroot=subroot)
        scope.topic.return_value = SimpleNamespace(id=topic_id)
        return scope

    def test_builds_per_scope(self):
        out = scoped_memory(_pf(), [self._scope(A, "svc/a"), self._scope(B, "svc/b")], REPO, "x")
        assert set(out) == {A, B}
        assert "A conventions" in out[A] and "B conventions" in out[B]

    def test_empty_run_text(self):
        assert scoped_memory(_pf(), [self._scope(A, "svc/a")], REPO, "") == {}

    def test_failure_falls_back(self):
        pf = MagicMock()
        pf.scoped_text.side_effect = RuntimeError("x")
        assert scoped_memory(pf, [self._scope(A, "svc/a")], REPO, "x") == {}


_REVIEW = Path(__file__).resolve().parents[1] / "src" / "codespy" / "agents" / "review"


@pytest.mark.parametrize("name", ["code_review", "doc", "seo", "supply_chain"])
def test_per_scope_agents_use_scoped_memory(name):
    src = (_REVIEW / name / "agent.py").read_text()
    assert "review_context.memory_for_scope(" in src
    assert "review_context.prefrontal_memory" not in src
