"""Tests for the opt-in implementation planner (review.plan)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from codespy.agents.review.models import (
    Issue,
    IssueCategory,
    IssueSeverity,
    PRContext,
    ReviewContext,
    ReviewMetadata,
)
from codespy.agents.review.plan import agent as plan_agent
from codespy.agents.review.plan.agent import (
    ImplementationPlanner,
    assemble_plan,
    plan_budget,
    scope_for_path,
)
from codespy.agents.review.plan.models import PlanStep, ScopePlan
from codespy.agents.review.scope.models import ScopeResult, ScopeType
from codespy.tools.git.models import ChangedFile, FileStatus
from codespy.workflows.review.models import MAX_PLAN_CHARS, FullPathInfo, ReviewResult
from codespy.workflows.review.reporters.git import GitReporter
from codespy.workflows.review.reporters.pull_request import PullRequestReporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scope(subroot: str) -> ScopeResult:
    return ScopeResult(
        repo="o/r", subroot=subroot, scope_type=ScopeType.SERVICE, has_changes=True, reason="r"
    )


def _issue(filename: str, title: str = "t", severity=IssueSeverity.HIGH) -> Issue:
    return Issue(
        category=IssueCategory.BUG,
        severity=severity,
        title=title,
        description="Fix it.",
        filename=filename,
        line_start=1,
    )


def _step(title: str = "Fix", files=None, n: int = 0, scope: str = "", depends_on=None,
          number: int = 0) -> PlanStep:
    return PlanStep(
        title=title,
        scope=scope,
        files=files or ["a.py"],
        change="Change " + "x" * n,
        verification="Run tests",
        addresses=[title],
        depends_on=depends_on or [],
        number=number,
    )


def _review_ctx(tmp_path: Path) -> ReviewContext:
    pr_ctx = PRContext(
        repo_slug="github.com/o/r",
        pr_number=1,
        pr_title="T",
        pr_url="https://github.com/o/r/pull/1",
        summary="S",
    )
    return ReviewContext(pr_context=pr_ctx, metadata=ReviewMetadata(repo_path=tmp_path))


def _planner(enabled: bool = True) -> ImplementationPlanner:
    planner = ImplementationPlanner.__new__(ImplementationPlanner)
    planner._settings = MagicMock()
    planner._settings.is_signature_enabled.side_effect = lambda name: enabled and name == "plan"
    planner._settings.get_max_concurrent_scopes.return_value = 4
    planner._settings.get_max_iters.return_value = 3
    planner._settings.get_max_llm_calls.return_value = 5
    planner._settings.get_rlm_threshold.return_value = 0.3
    planner._settings.review.pr_context.max_comment_chars = 1000
    planner._settings.review.pr_context.max_block_chars = 12000
    planner._cost_tracker = MagicMock()
    return planner


def _result(plan: list[ScopePlan], full_path: bool = False, n_issues: int = 1, desc: str = "d"):
    fp = None
    if full_path:
        fp = FullPathInfo(
            path="src", ref="main", sha="s" * 40, platform="github", host="github.com",
            owner="o", repo="r", files_reviewed=1, scopes=["src"],
        )
    return ReviewResult(
        pr_number=1,
        pr_title="t",
        pr_url="https://github.com/o/r/pull/1",
        repo="o/r",
        run_id="r" * 32,
        model_used="m",
        issues=[
            Issue(
                category=IssueCategory.BUG, title=f"I{i}", description=desc,
                filename=f"src/f{i}.py", line_start=None if not full_path else 1,
            )
            for i in range(n_issues)
        ],
        recommendation="REQUEST_CHANGES",
        implementation_plan=plan,
        full_path=fp,
    )


def _balanced(text: str) -> bool:
    return (
        text.count("<details>") == text.count("</details>")
        and text.count("<blockquote>") == text.count("</blockquote>")
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_disabled_by_default(self, monkeypatch, tmp_path):
        from codespy.config import Settings

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("REVIEW_PLAN_ENABLED", raising=False)
        assert Settings().is_signature_enabled("plan") is False

    def test_env_enables(self, monkeypatch, tmp_path):
        from codespy.config import Settings

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("REVIEW_PLAN_ENABLED", "true")
        monkeypatch.setenv("REVIEW_PLAN_MAX_ITERS", "7")
        settings = Settings()
        assert settings.is_signature_enabled("plan") is True
        assert settings.review.plan.max_iters == 7

    def test_is_signature_and_run_prefrontal_consumer(self):
        from codespy.config_dspy import SIGNATURE_NAMES
        from codespy.config_memory import _RUN_PREFRONTAL_CONSUMERS

        assert "plan" in SIGNATURE_NAMES
        assert "plan" in _RUN_PREFRONTAL_CONSUMERS


# ---------------------------------------------------------------------------
# Grouping and paths
# ---------------------------------------------------------------------------


class TestScopeForPath:
    def test_longest_prefix_wins(self):
        subroots = ["apps", "apps/web", "."]
        assert scope_for_path(subroots, "apps/web/a.ts") == "apps/web"
        assert scope_for_path(subroots, "apps/b.ts") == "apps"

    def test_segment_aware(self):
        assert scope_for_path(["apps/web"], "apps/web2/x.ts") == "."

    def test_root_fallback_and_normalization(self):
        assert scope_for_path(["svc"], "README.md") == "."
        assert scope_for_path(["svc/"], "./svc/a.go") == "svc"
        assert scope_for_path([], "x.py") == "."


class TestAssemblePlan:
    def test_global_numbering_grouped_by_first_appearance(self):
        steps = [_step("A1", scope="a"), _step("B1", scope="b"), _step("A2", scope="a")]
        plans = assemble_plan(steps, ["a", "b"])
        assert [p.subroot for p in plans] == ["a", "b"]
        assert [s.number for s in plans[0].steps] == [1, 3]
        assert [s.number for s in plans[1].steps] == [2]

    def test_invalid_scope_inferred_from_files_else_root(self):
        steps = [
            _step("X", scope="nope", files=["b/pkg/x.py"]),
            _step("Y", scope="", files=["other/y.py"]),
            _step("Z", scope="./b/", files=["b/z.py"]),
        ]
        plans = {p.subroot: p.steps for p in assemble_plan(steps, ["a", "b"])}
        assert [s.title for s in plans["b"]] == ["X", "Z"]
        assert [s.title for s in plans["."]] == ["Y"]
        assert all(s.scope == sr for sr, ss in plans.items() for s in ss)

    def test_depends_on_sanitized(self):
        steps = [
            _step("A", scope="a"),
            _step("B", scope="a", depends_on=[1, 1, 2, 3, 0, -1, 99]),
        ]
        plans = assemble_plan(steps, ["a"])
        assert plans[0].steps[0].depends_on == []
        assert plans[0].steps[1].depends_on == [1]

    def test_paths_normalized_and_inputs_not_mutated(self):
        original = _step("A", scope="a", files=["./a/x.py", "/a/y.py", ""], depends_on=[5])
        plans = assemble_plan([original], ["a"])
        assert plans[0].steps[0].files == ["a/x.py", "a/y.py"]
        assert original.files == ["./a/x.py", "/a/y.py", ""]
        assert original.number == 0 and original.depends_on == [5]

    def test_empty(self):
        assert assemble_plan([], ["a"]) == []


class TestBudget:
    def test_scales_with_scopes_with_issues(self):
        subroots = [".", "a", "b", "c", "unused"]
        assert plan_budget(5, 16, subroots, [_issue("a/x"), _issue("a/y")]) == (5, 16)
        issues = [_issue("a/x"), _issue("b/x"), _issue("c/x")]
        assert plan_budget(5, 16, subroots, issues) == (15, 48)

    def test_orphans_count_as_root(self):
        assert plan_budget(5, 16, ["a"], [_issue("a/x"), _issue("lib/x")]) == (10, 32)

    def test_capped(self):
        subroots = [f"s{i}" for i in range(20)]
        issues = [_issue(f"s{i}/x") for i in range(20)]
        assert plan_budget(5, 16, subroots, issues) == (40, 128)

    def test_at_least_one(self):
        assert plan_budget(5, 16, [], []) == (5, 16)


# ---------------------------------------------------------------------------
# Planner agent
# ---------------------------------------------------------------------------


class TestPlanner:
    def test_disabled_or_no_issues_does_nothing(self, monkeypatch, tmp_path):
        called = MagicMock()
        monkeypatch.setattr(plan_agent, "build_rlm_agent", called)
        monkeypatch.setattr(plan_agent, "connect_mcp_server", called)
        ctx = _review_ctx(tmp_path)
        disabled = _planner(enabled=False)
        assert asyncio.run(disabled.aforward([_scope(".")], [_issue("a")], ctx)) == []
        assert asyncio.run(_planner().aforward([_scope(".")], [], ctx)) == []
        called.assert_not_called()

    def _patch_run(self, monkeypatch, run):
        captured: dict = {"runs": 0, "roots": []}

        async def fake_connect(server, roots, contexts, name):
            captured["roots"].append(roots)
            return []

        async def fake_cleanup(contexts):
            captured["cleaned"] = True

        async def fake_run(agent, **kwargs):
            captured["runs"] += 1
            captured.update(kwargs)
            return await run(**kwargs)

        def fake_build(*a, **k):
            captured["budget"] = (k["max_iters"], k["max_llm_calls"])
            return object()

        @asynccontextmanager
        async def fake_sig_ctx(*a, **k):
            yield

        monkeypatch.setattr(plan_agent, "connect_mcp_server", fake_connect)
        monkeypatch.setattr(plan_agent, "cleanup_mcp_contexts", fake_cleanup)
        monkeypatch.setattr(plan_agent, "build_rlm_agent", fake_build)
        monkeypatch.setattr(plan_agent, "run_rlm", fake_run)
        monkeypatch.setattr(plan_agent, "log_rlm_outcome", lambda *a, **k: None)
        monkeypatch.setattr(plan_agent, "get_prefrontal", lambda *a, **k: None)
        monkeypatch.setattr(plan_agent, "SignatureContext", fake_sig_ctx)
        return captured

    def test_single_holistic_run(self, monkeypatch, tmp_path):
        """One run over all scopes and issues, repo-rooted tools, repo-relative paths."""

        async def run(**kwargs):
            return SimpleNamespace(steps=[
                _step("Lib", scope="libs/common", files=["libs/common/x.py"]),
                _step("Svc", scope="svc", files=["svc/pkg/a.py"], depends_on=[1]),
            ])

        captured = self._patch_run(monkeypatch, run)
        svc = _scope("svc")
        svc.changed_files = [
            ChangedFile(filename="svc/pkg/a.py", status=FileStatus.MODIFIED, patch="@@ diff")
        ]
        scopes = [svc, _scope("libs/common")]
        issues = [_issue("svc/pkg/a.py", "Svc"), _issue("libs/common/x.py", "Lib")]
        plans = asyncio.run(
            _planner().aforward(
                scopes, issues, _review_ctx(tmp_path),
                quality_assessment="QA", recommendation="REQUEST_CHANGES",
            )
        )
        assert captured["runs"] == 1
        assert captured["roots"] and all(r == [str(tmp_path)] for r in captured["roots"])
        assert captured["budget"] == (6, 10)  # 3/5 per scope × 2 scopes with issues
        assert (captured["quality_assessment"], captured["recommendation"]) == (
            "QA", "REQUEST_CHANGES"
        )
        assert captured["summary"] == "S"
        assert [i.filename for i in captured["issues"]] == ["svc/pkg/a.py", "libs/common/x.py"]
        assert [s.subroot for s in captured["scopes"]] == ["svc", "libs/common"]
        assert captured["scopes"][0].changed_files[0].patch is None
        assert svc.changed_files[0].patch == "@@ diff"  # original untouched
        assert captured["cleaned"] is True
        assert "prefrontal_memory" not in captured
        assert [p.subroot for p in plans] == ["libs/common", "svc"]
        assert plans[1].steps[0].number == 2 and plans[1].steps[0].depends_on == [1]

    def test_run_failure_propagates_and_cleans_up(self, monkeypatch, tmp_path):
        async def run(**kwargs):
            raise RuntimeError("boom")

        captured = self._patch_run(monkeypatch, run)
        with pytest.raises(RuntimeError):
            asyncio.run(
                _planner().aforward([_scope("svc")], [_issue("svc/a")], _review_ctx(tmp_path))
            )
        assert captured["cleaned"] is True


class TestPipelineStep:
    def _pipeline(self, enabled: bool = True):
        from codespy.workflows.review.pipeline import ReviewPipeline

        pipeline = ReviewPipeline.__new__(ReviewPipeline)
        pipeline.settings = MagicMock()
        pipeline.settings.is_signature_enabled.side_effect = lambda n: enabled and n == "plan"
        pipeline.planner = MagicMock()
        return pipeline

    def test_skipped_when_disabled_or_no_issues(self, tmp_path):
        for pipeline, issues in ((self._pipeline(False), [_issue("a")]), (self._pipeline(), [])):
            assert pipeline._run_planner([], issues, _review_ctx(tmp_path), "qa", "rec") == []
            pipeline.planner.aforward.assert_not_called()

    def test_failure_yields_no_plan(self, tmp_path):
        pipeline = self._pipeline()

        async def boom(**kwargs):
            raise RuntimeError("planner down")

        pipeline.planner.aforward = boom
        assert pipeline._run_planner([], [_issue("a")], _review_ctx(tmp_path), "qa", "r") == []

    def test_passes_audit_verdict(self, tmp_path):
        pipeline = self._pipeline()
        seen = {}

        async def fake(**kwargs):
            seen.update(kwargs)
            return [ScopePlan(subroot=".", steps=[_step()])]

        pipeline.planner.aforward = fake
        plan = pipeline._run_planner([], [_issue("a")], _review_ctx(tmp_path), "qa", "rec")
        assert len(plan) == 1
        assert (seen["quality_assessment"], seen["recommendation"]) == ("qa", "rec")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_empty_plan_no_section(self):
        result = _result([])
        assert result.implementation_plan_markdown_lines() == []
        assert "Implementation Plan" not in result.to_markdown()
        result = _result([ScopePlan(subroot="a", steps=[])])
        assert result.implementation_plan_markdown_lines() == []

    def test_single_scope_flat(self):
        text = "\n".join(_result([ScopePlan(subroot="svc", steps=[_step("A"), _step("B")])])
                         .implementation_plan_markdown_lines())
        assert "<summary>🛠️ Implementation Plan (2 steps)</summary>" in text
        assert text.count("<details>") == 1
        assert "**1. A**" in text and "**2. B**" in text
        assert "**Verify:** Run tests" in text
        assert _balanced(text)

    def test_multi_scope_nested_and_numbering_continues(self):
        plan = [
            ScopePlan(subroot="a", steps=[_step("A")]),
            ScopePlan(subroot="b", steps=[_step("B")]),
        ]
        text = "\n".join(_result(plan).implementation_plan_markdown_lines())
        assert "(2 steps in 2 scopes)" in text
        assert "<summary>`a` (1 step)</summary>" in text
        assert "<summary>`b` (1 step)</summary>" in text
        assert "**2. B**" in text
        assert text.count("<details>") == 3
        assert _balanced(text)

    def test_global_numbers_and_depends_on(self):
        plan = assemble_plan(
            [
                _step("Lib", scope="libs/common"),
                _step("Svc", scope="svc", depends_on=[1]),
                _step("Lib2", scope="libs/common", depends_on=[1]),
            ],
            ["libs/common", "svc"],
        )
        text = "\n".join(_result(plan).implementation_plan_markdown_lines())
        lib_section = text[text.index("`libs/common`"):text.index("`svc` (")]
        assert "**1. Lib**" in lib_section and "**3. Lib2**" in lib_section
        assert "**Depends on:** 1\n" in lib_section + "\n"  # same scope: no label
        assert "**2. Svc**" in text
        assert "**Depends on:** 1 (`libs/common`)" in text
        assert _balanced(text)

    @pytest.mark.parametrize("nested", [False, True])
    def test_truncation_within_budget(self, nested):
        steps = [_step(f"S{i}", n=500) for i in range(50)]
        plan = (
            [ScopePlan(subroot="a", steps=steps[:25]), ScopePlan(subroot="b", steps=steps[25:])]
            if nested
            else [ScopePlan(subroot="a", steps=steps)]
        )
        lines = _result(plan).implementation_plan_markdown_lines(
            max_chars=5000, overflow_link="https://x/full.md"
        )
        text = "\n".join(lines)
        assert len(text) <= 5000
        assert _balanced(text)
        assert "more steps omitted" in text
        assert "[full plan](https://x/full.md)" in text

    def test_nothing_fits(self):
        plan = [ScopePlan(subroot="a", steps=[_step(n=2000)])]
        assert _result(plan).implementation_plan_markdown_lines(max_chars=500) == []

    def test_to_markdown_full_plan_after_recommendation(self):
        plan = [ScopePlan(subroot="a", steps=[_step(f"S{i}", n=500) for i in range(60)])]
        md = _result(plan).to_markdown()
        assert md.index("## Recommendation") < md.index("Implementation Plan")
        assert "**60. S59**" in md
        assert "omitted" not in md

    def test_json_has_plan(self):
        plan = assemble_plan([_step(scope="a"), _step(scope="a", depends_on=[1])], ["a"])
        data = _result(plan).to_json_dict()
        steps = data["implementation_plan"][0]["steps"]
        assert steps[0]["verification"] == "Run tests"
        assert [s["number"] for s in steps] == [1, 2]
        assert steps[1]["depends_on"] == [1] and steps[1]["scope"] == "a"


class TestReporters:
    def _huge_plan(self):
        return [ScopePlan(subroot="src", steps=[_step(f"S{i}", n=800) for i in range(200)])]

    def test_git_body_plan_after_recommendation(self):
        reporter = GitReporter.__new__(GitReporter)
        result = _result([ScopePlan(subroot="src", steps=[_step()])])
        body = reporter._build_review_body(result, result.issues)
        assert body.index("💡 Recommendation") < body.index("🛠️ Implementation Plan")
        assert _balanced(body)

    def test_git_body_within_limit(self):
        from codespy.workflows.review.models import MemorySection, RecalledMemory

        reporter = GitReporter.__new__(GitReporter)
        result = _result(self._huge_plan(), n_issues=60, desc="y" * 500)
        result.memories = [
            RecalledMemory(task="review", sections=[MemorySection(title="M", text="z" * 50000)])
        ]
        body = reporter._build_review_body(result, result.issues, reserve=1000)
        assert len(body) <= GitReporter.MAX_BODY_CHARS - 1000
        assert "🛠️ Implementation Plan" in body
        assert _balanced(body)

    def test_pr_report_body_within_limit_with_link(self):
        reporter = PullRequestReporter.__new__(PullRequestReporter)
        reporter.repo_url = "https://github.com/o/r"
        reporter.client = MagicMock(platform_name="GitHub")
        result = _result(self._huge_plan(), full_path=True, n_issues=300, desc="y" * 400)
        body = reporter.build_body(result)
        assert len(body) <= PullRequestReporter.MAX_BODY_CHARS
        assert body.index("💡 Recommendation") < body.index("🛠️ Implementation Plan")
        plan_part = body[body.index("🛠️ Implementation Plan"):]
        assert "[full plan](https://github.com/o/r/blob/codespy/review/src/" in plan_part
        assert _balanced(body)
        # The committed report keeps the whole plan
        assert "**200. S199**" in result.to_markdown()
        assert MAX_PLAN_CHARS < PullRequestReporter.MAX_BODY_CHARS
