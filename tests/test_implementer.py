"""Tests for the opt-in implementation agent (review.implement) and its fix PR."""

from __future__ import annotations

import asyncio
import json
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from codespy.agents.review.implement import FileChange, ImplementationResult, SkippedStep
from codespy.agents.review.implement import agent as impl_agent
from codespy.agents.review.implement.agent import (
    IMPLEMENT_BUDGET_STEP_CAP,
    Implementer,
    collect_changes,
    implement_budget,
    restore_clone,
    untracked_files,
)
from codespy.agents.review.models import (
    Issue,
    IssueCategory,
    PRContext,
    ReviewContext,
    ReviewMetadata,
)
from codespy.agents.review.plan.models import ImplementationPlan, PlanStep, ScopePlan
from codespy.agents.review.scope.models import ScopeResult, ScopeType
from codespy.tools.git.base import SubmittedPullRequest
from codespy.tools.git.conversation import has_self_signature
from codespy.tools.git.models import GitPlatform, PullRequest
from codespy.workflows.review.models import ReviewResult
from codespy.workflows.review.reporters.fix_pull_request import FixPullRequestReporter
from codespy.workflows.review.reporters.git import GitReporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "a.py").write_text("a = 1\n")
    (r / "gone.py").write_text("g = 1\n")
    (r / "run.sh").write_text("echo\n")
    (r / "run.sh").chmod(0o755)
    (r / "keep.txt").write_text("keep\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    return r


def _step(n: int, title: str = "Fix") -> PlanStep:
    return PlanStep(
        title=f"{title} {n}", scope=".", files=["a.py"], change="c", verification="v",
        addresses=[f"I{n}"], number=n,
    )


def _plan(n_steps: int = 2) -> ImplementationPlan:
    return ImplementationPlan(
        goal="g", scopes=[ScopePlan(subroot=".", steps=[_step(i + 1) for i in range(n_steps)])]
    )


def _scope() -> ScopeResult:
    return ScopeResult(repo="o/r", subroot=".", scope_type=ScopeType.SERVICE, has_changes=True,
                       reason="r")


def _issue() -> Issue:
    return Issue(category=IssueCategory.BUG, title="I1", description="d", filename="a.py",
                 line_start=1)


def _ctx(repo_path: Path) -> ReviewContext:
    pr_ctx = PRContext(repo_slug="github.com/o/r", pr_number=1, pr_title="T",
                       pr_url="https://github.com/o/r/pull/1", summary="S")
    return ReviewContext(pr_context=pr_ctx, metadata=ReviewMetadata(repo_path=repo_path))


def _implementer(cache_dir: Path, enabled: bool = True) -> Implementer:
    impl = Implementer.__new__(Implementer)
    impl._settings = MagicMock()
    impl._settings.is_signature_enabled.side_effect = lambda n: enabled and n == "implement"
    impl._settings.get_max_iters.return_value = 3
    impl._settings.get_max_llm_calls.return_value = 5
    impl._settings.get_rlm_threshold.return_value = 0.3
    impl._settings.review.cache_dir = cache_dir
    impl._settings.review.pr_context.max_comment_chars = 1000
    impl._settings.review.pr_context.max_block_chars = 12000
    impl._cost_tracker = MagicMock()
    return impl


def _patch_run(monkeypatch, run):
    captured: dict = {"runs": 0, "servers": []}

    async def fake_connect(server, args, contexts, name):
        captured["servers"].append((Path(server).parent.name, args))
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

    monkeypatch.setattr(impl_agent, "connect_mcp_server", fake_connect)
    monkeypatch.setattr(impl_agent, "cleanup_mcp_contexts", fake_cleanup)
    monkeypatch.setattr(impl_agent, "build_rlm_agent", fake_build)
    monkeypatch.setattr(impl_agent, "run_rlm", fake_run)
    monkeypatch.setattr(impl_agent, "log_rlm_outcome", lambda *a, **k: None)
    monkeypatch.setattr(impl_agent, "get_prefrontal", lambda *a, **k: None)
    monkeypatch.setattr(impl_agent, "SignatureContext", fake_sig_ctx)
    return captured


def _status(repo: Path) -> str:
    return subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout


def _pr(**kw) -> PullRequest:
    from datetime import UTC, datetime

    base = dict(
        number=12, title="T", state="open", author="a", base_branch="main",
        head_branch="feature/x", base_sha="b" * 40, head_sha="h" * 40,
        created_at=datetime.now(UTC), updated_at=datetime.now(UTC), repo_owner="o",
        repo_name="r", host="github.com", platform=GitPlatform.GITHUB,
    )
    base.update(kw)
    return PullRequest(**base)


def _impl() -> ImplementationResult:
    return ImplementationResult(
        summary="Fixed the bug.",
        completed_steps=[1],
        skipped_steps=[SkippedStep(number=2, reason="Needs a schema migration.")],
        changes=[FileChange(path="a.py", content="a = 2\n"), FileChange(path="gone.py")],
    )


def _result(impl: ImplementationResult | None = None) -> ReviewResult:
    return ReviewResult(
        pr_number=12, pr_title="T", pr_url="https://github.com/o/r/pull/12", repo="o/r",
        run_id="r" * 32, model_used="m", issues=[_issue()], recommendation="REQUEST_CHANGES",
        implementation_plan=_plan(), implementation=impl,
    )


# ---------------------------------------------------------------------------
# Budget, collection, restore
# ---------------------------------------------------------------------------


class TestBudget:
    def test_scales_with_steps_and_caps(self):
        assert implement_budget(5, 16, 0) == (5, 16)
        assert implement_budget(5, 16, 3) == (15, 48)
        assert implement_budget(5, 16, 50) == (5 * IMPLEMENT_BUDGET_STEP_CAP, 16 * IMPLEMENT_BUDGET_STEP_CAP)


class TestCollectAndRestore:
    def test_modified_added_deleted(self, repo):
        (repo / "a.py").write_text("a = 2\n")
        (repo / "gone.py").unlink()
        (repo / "pkg").mkdir()
        (repo / "pkg" / "new.py").write_text("n = 1\n")
        (repo / "run.sh").write_text("echo hi\n")
        changes = {c.path: c for c in collect_changes(repo)}
        assert set(changes) == {"a.py", "gone.py", "pkg/new.py", "run.sh"}
        assert changes["a.py"].content == "a = 2\n"
        assert changes["gone.py"].content is None
        assert changes["run.sh"].executable is True
        assert changes["pkg/new.py"].executable is False

    def test_preexisting_untracked_files_are_not_published(self, repo):
        (repo / ".cache.json").write_text("stale")
        before = untracked_files(repo)
        (repo / "agent.py").write_text("new\n")
        assert [c.path for c in collect_changes(repo, before)] == ["agent.py"]
        # Without the snapshot the stale file would be published
        assert {c.path for c in collect_changes(repo)} == {".cache.json", "agent.py"}

    def test_protected_paths_and_symlinks_skipped(self, repo):
        (repo / ".github" / "workflows").mkdir(parents=True)
        (repo / ".github" / "workflows" / "ci.yml").write_text("x")
        (repo / ".codespy").mkdir()
        (repo / ".codespy" / "r.md").write_text("x")
        (repo / "link").symlink_to(repo / "a.py")
        (repo / "ok.py").write_text("ok\n")
        assert [c.path for c in collect_changes(repo)] == ["ok.py"]

    def test_caps_fail_closed(self, repo, monkeypatch):
        monkeypatch.setattr(impl_agent, "MAX_CHANGED_FILES", 1)
        (repo / "x.py").write_text("x")
        (repo / "y.py").write_text("y")
        assert collect_changes(repo) == []
        monkeypatch.setattr(impl_agent, "MAX_CHANGED_FILES", 100)
        monkeypatch.setattr(impl_agent, "MAX_CHANGED_BYTES", 1)
        assert collect_changes(repo) == []

    def test_non_utf8_fails_closed(self, repo):
        (repo / "ok.py").write_text("ok")
        (repo / "bin.dat").write_bytes(b"\xff\xfe\x00")
        assert collect_changes(repo) == []

    def test_restore_leaves_clean_tree_and_keeps_earlier_untracked(self, repo, tmp_path):
        (repo / "mine.txt").write_text("pre-existing untracked")
        before = untracked_files(repo)
        (repo / "a.py").write_text("changed")
        (repo / "gone.py").unlink()
        (repo / "deep" / "dir").mkdir(parents=True)
        (repo / "deep" / "dir" / "new.py").write_text("n")
        restore_clone(repo, before, tmp_path)
        assert _status(repo) == "?? mine.txt\n"
        assert (repo / "a.py").read_text() == "a = 1\n"
        assert not (repo / "deep").exists()

    def test_restore_failure_deletes_cached_clone_only(self, repo, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("git broke")

        monkeypatch.setattr(impl_agent, "_git", boom)
        restore_clone(repo, set(), tmp_path / "elsewhere")
        assert repo.exists()  # not under the cache: never deleted
        restore_clone(repo, set(), tmp_path)
        assert not repo.exists()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TestImplementer:
    def test_skipped_when_disabled_or_no_plan(self, monkeypatch, repo, tmp_path):
        called = MagicMock()
        monkeypatch.setattr(impl_agent, "connect_mcp_server", called)
        ctx = _ctx(repo)
        assert asyncio.run(_implementer(tmp_path, enabled=False).aforward(_plan(), [], [], ctx)) is None
        assert asyncio.run(_implementer(tmp_path).aforward(None, [], [], ctx)) is None
        empty = ImplementationPlan(scopes=[])
        assert asyncio.run(_implementer(tmp_path).aforward(empty, [], [], ctx)) is None
        called.assert_not_called()

    def test_missing_metadata_raises_value_error(self, tmp_path):
        ctx = _ctx(tmp_path).model_copy(update={"metadata": None})
        with pytest.raises(ValueError, match="metadata required"):
            asyncio.run(_implementer(tmp_path).aforward(_plan(), [_scope()], [], ctx))

    def test_run_collects_edits_and_restores_clone(self, monkeypatch, repo, tmp_path):
        async def run(**kwargs):
            (repo / "a.py").write_text("a = 2\n")
            (repo / "new.py").write_text("n = 1\n")
            return SimpleNamespace(
                summary=" Did it. ",
                completed_steps=[1, 99],
                skipped_steps=[{"number": 2, "reason": "no"}, {"number": 42, "reason": "x"}],
            )

        captured = _patch_run(monkeypatch, run)
        result = asyncio.run(
            _implementer(tmp_path).aforward(_plan(3), [_scope()], [_issue()], _ctx(repo))
        )
        assert captured["runs"] == 1
        assert captured["budget"] == (9, 15)
        assert ("filesystem", [str(repo), "--writable"]) in captured["servers"]
        assert all(args[0] == str(repo) for _, args in captured["servers"])
        assert captured["pr_summary"] == "S"
        assert captured["cleaned"] is True
        assert result is not None
        assert {c.path: c.content for c in result.changes} == {"a.py": "a = 2\n", "new.py": "n = 1\n"}
        assert result.completed_steps == [1]
        assert [s.number for s in result.skipped_steps] == [2]
        assert result.summary == "Did it."
        assert _status(repo) == ""

    def test_preexisting_untracked_file_kept_and_not_published(self, monkeypatch, repo, tmp_path):
        (repo / "stale.log").write_text("from an earlier run")

        async def run(**kwargs):
            (repo / "a.py").write_text("a = 2\n")
            return SimpleNamespace(summary="s", completed_steps=[1], skipped_steps=[])

        _patch_run(monkeypatch, run)
        result = asyncio.run(_implementer(tmp_path).aforward(_plan(), [_scope()], [], _ctx(repo)))
        assert result is not None and [c.path for c in result.changes] == ["a.py"]
        assert (repo / "stale.log").read_text() == "from an earlier run"
        assert _status(repo) == "?? stale.log\n"

    def test_dirty_tracked_clone_is_skipped(self, monkeypatch, repo, tmp_path):
        (repo / "keep.txt").write_text("left over\n")
        called = MagicMock()
        monkeypatch.setattr(impl_agent, "connect_mcp_server", called)
        assert asyncio.run(_implementer(tmp_path).aforward(_plan(), [_scope()], [], _ctx(repo))) is None
        called.assert_not_called()
        assert (repo / "keep.txt").read_text() == "left over\n"  # untouched

    def test_no_changes_yields_none(self, monkeypatch, repo, tmp_path):
        async def run(**kwargs):
            return SimpleNamespace(summary="nothing", completed_steps=[], skipped_steps=[])

        _patch_run(monkeypatch, run)
        assert asyncio.run(_implementer(tmp_path).aforward(_plan(), [_scope()], [], _ctx(repo))) is None

    def test_failure_propagates_and_clone_restored(self, monkeypatch, repo, tmp_path):
        async def run(**kwargs):
            (repo / "a.py").write_text("half-done")
            (repo / "tmp.py").write_text("t")
            raise RuntimeError("boom")

        captured = _patch_run(monkeypatch, run)
        with pytest.raises(RuntimeError):
            asyncio.run(_implementer(tmp_path).aforward(_plan(), [_scope()], [], _ctx(repo)))
        assert captured["cleaned"] is True
        assert _status(repo) == ""


class TestPipelineStep:
    def _pipeline(self, implement: bool = True, plan: bool = True):
        from codespy.workflows.review.pipeline import ReviewPipeline

        pipeline = ReviewPipeline.__new__(ReviewPipeline)
        pipeline.settings = MagicMock()
        enabled = {"implement": implement, "plan": plan}
        pipeline.settings.is_signature_enabled.side_effect = lambda n: enabled.get(n, False)
        pipeline.implementer = MagicMock()
        return pipeline

    @pytest.mark.parametrize(
        ("implement", "plan_enabled", "plan", "is_local", "is_fork"),
        [
            (False, True, _plan(), False, False),
            (True, False, _plan(), False, False),
            (True, True, None, False, False),
            (True, True, _plan(), True, False),
            (True, True, _plan(), False, True),
        ],
    )
    def test_skips(self, implement, plan_enabled, plan, is_local, is_fork, tmp_path):
        pipeline = self._pipeline(implement, plan_enabled)
        out = pipeline._run_implementer([], [], _ctx(tmp_path), plan, is_local, is_fork)
        assert out is None
        pipeline.implementer.aforward.assert_not_called()

    def test_failure_soft_fails(self, tmp_path):
        pipeline = self._pipeline()

        async def boom(**kwargs):
            raise RuntimeError("x")

        pipeline.implementer.aforward = boom
        assert pipeline._run_implementer([], [], _ctx(tmp_path), _plan(), False, False) is None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class TestOutput:
    def test_json_has_paths_not_contents(self):
        data = json.loads(json.dumps(_result(_impl()).to_json_dict()))
        impl = data["implementation"]
        assert impl["changed_files"] == ["a.py", "gone.py"]
        assert [c["status"] for c in impl["changes"]] == ["written", "deleted"]
        assert all("content" not in c for c in impl["changes"])
        assert "a = 2" not in json.dumps(data)

    def test_markdown_section(self):
        md = _result(_impl()).to_markdown()
        assert "🤖 Implementation (2 files changed, 1 step(s) done, 1 skipped)" in md
        assert "- 2: Needs a schema migration." in md
        assert "`gone.py` (deleted)" in md
        assert "not published" in md
        assert md.index("Implementation Plan") < md.index("🤖 Implementation")
        assert _result().implementation_markdown_lines() == []

    def test_review_body_links_fix_pr(self):
        impl = _impl()
        impl.pull_request_url = "https://github.com/o/r/pull/13"
        reporter = GitReporter.__new__(GitReporter)
        body = reporter._build_review_body(_result(impl), [], split=False)
        assert "**Fix PR:** https://github.com/o/r/pull/13" in body
        split_body = reporter._build_review_body(_result(impl), [], split=True)
        assert "**Fix PR:** https://github.com/o/r/pull/13" in split_body
        assert body.count("<details>") == body.count("</details>")


class TestFixPullRequest:
    def _reporter(self) -> FixPullRequestReporter:
        reporter = FixPullRequestReporter.__new__(FixPullRequestReporter)
        reporter.url = "https://github.com/o/r/pull/12"
        reporter.client = MagicMock(platform_name="GitHub")
        return reporter

    def test_publish_targets_pr_branch_and_sets_url(self):
        reporter = self._reporter()
        reporter.client.upsert_branch_pull_request.return_value = SubmittedPullRequest(
            number=13, url="https://github.com/o/r/pull/13", created=True
        )
        result = _result(_impl())
        handle = reporter.publish(result, _pr())
        assert handle is not None and handle.number == 13
        kwargs = reporter.client.upsert_branch_pull_request.call_args.kwargs
        assert kwargs["base_branch"] == "feature/x"
        assert kwargs["base_sha"] == "h" * 40
        assert kwargs["head_branch"] == "codespy/fix/12"
        assert kwargs["files"] == {"a.py": "a = 2\n", "gone.py": None}
        assert kwargs["title"] == "codespy: fixes for #12"
        body = kwargs["body"]
        assert "<!-- codespy:fix v=1 run_id=" in body
        assert has_self_signature(body)
        assert "https://github.com/o/r/pull/12 (#12)" in body
        assert "- 1. Fix 1" in body
        assert "Needs a schema migration." in body
        assert "cannot run code" in body
        assert "Implementation Plan" in body
        assert len(body) <= FixPullRequestReporter.MAX_BODY_CHARS
        assert result.implementation is not None
        assert result.implementation.pull_request_url == "https://github.com/o/r/pull/13"

    def test_build_body_without_implementation_raises(self):
        with pytest.raises(ValueError, match="implementation required"):
            self._reporter().build_body(_result(None), _pr())

    def test_gitlab_title(self):
        assert self._reporter().title(_pr(platform=GitPlatform.GITLAB)) == "codespy: fixes for !12"

    def test_nothing_to_publish(self):
        reporter = self._reporter()
        assert reporter.publish(_result(None), _pr()) is None
        reporter.client.upsert_branch_pull_request.assert_not_called()


class TestConfig:
    def test_signature_and_env(self, monkeypatch, tmp_path):
        from codespy.config import Settings
        from codespy.config_dspy import SIGNATURE_NAMES

        assert "implement" in SIGNATURE_NAMES
        monkeypatch.chdir(tmp_path)
        assert Settings().is_signature_enabled("implement") is False
        monkeypatch.setenv("REVIEW_IMPLEMENT_ENABLED", "true")
        monkeypatch.setenv("REVIEW_IMPLEMENT_MAX_ITERS", "7")
        settings = Settings()
        assert settings.is_signature_enabled("implement") is True
        assert settings.get_max_iters("implement") == 7
