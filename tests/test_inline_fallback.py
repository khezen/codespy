"""Issues that can't be inline comments share one review-body section."""

from unittest.mock import MagicMock

import pytest

from codespy.agents.review.models import Issue, IssueCategory
from codespy.tools.git.base import SubmittedReview
from codespy.tools.git.models import ChangedFile, FileStatus
from codespy.workflows.review.models import ReviewResult
from codespy.workflows.review.reporters.git import GitReporter

TITLE = "Issues without line reference or could not post inline"
PATCH = "@@ -1,2 +1,3 @@\n a\n+b\n c\n"  # new lines 1-3 are in the diff


def _issue(title: str, filename: str = "a.py", line: int | None = 2, end: int | None = None):
    return Issue(
        category=IssueCategory.BUG, title=title, description="d", filename=filename,
        line_start=line, line_end=end,
    )


def _file(name: str = "a.py", patch: str | None = PATCH) -> ChangedFile:
    return ChangedFile(filename=name, status=FileStatus.MODIFIED, patch=patch)


def _result(*issues):
    return ReviewResult(pr_number=1, pr_title="t", pr_url="u", repo="o/r", model_used="m",
                        run_id="run42", issues=list(issues))


def _reporter(client=None) -> GitReporter:
    reporter = GitReporter.__new__(GitReporter)
    reporter.url = "https://github.com/o/r/pull/1"
    reporter.client = client or MagicMock(platform_name="GitHub")
    reporter.client.submit_review.return_value = SubmittedReview(id=7)
    return reporter


def _section(body: str) -> str:
    """Text of the single body section (from its summary to the next top-level section)."""
    assert body.count(f"<summary>⚠️ {TITLE}</summary>") == 1
    return body.split(f"<summary>⚠️ {TITLE}</summary>")[1]


class TestPrecheck:
    def test_moves_issues_outside_the_diff_into_the_section(self):
        inline = _issue("Inline", line=2)
        no_line = _issue("NoLine", line=None)
        bad_line = _issue("BadLine", line=40)
        bad_end = _issue("BadEnd", line=2, end=40)
        other_file = _issue("OtherFile", filename="gone.py", line=1)
        reporter = _reporter()
        reporter.publish(
            _result(inline, no_line, bad_line, bad_end, other_file), changed_files=[_file()]
        )
        kwargs = reporter.client.submit_review.call_args.kwargs
        assert [c["path"] for c in kwargs["comments"]] == ["a.py"]
        assert "Inline" in kwargs["comments"][0]["body"]
        section = _section(kwargs["body"])
        for title in ("NoLine", "BadLine", "BadEnd", "OtherFile"):
            assert title in section
        assert "Inline" not in section
        # Lines stay findable in the section
        assert "**File:** `a.py:40`" in section
        assert "**File:** `gone.py:1`" in section
        assert "**File:** `a.py:2-40`" in section
        assert "Additional Issues" not in kwargs["body"]

    def test_file_without_patch_goes_to_the_section(self):
        reporter = _reporter()
        reporter.publish(_result(_issue("Big")), changed_files=[_file(patch=None)])
        kwargs = reporter.client.submit_review.call_args.kwargs
        assert kwargs["comments"] == []
        assert "Big" in _section(kwargs["body"])

    def test_without_changed_files_nothing_is_prechecked(self):
        reporter = _reporter()
        reporter.publish(_result(_issue("X", filename="gone.py", line=99)))
        assert len(reporter.client.submit_review.call_args.kwargs["comments"]) == 1

    def test_comments_carry_unique_issue_key(self):
        a, b = _issue("A", line=1), _issue("B", line=2)
        comments = _reporter()._build_inline_comments([a, b], run_id="r")
        keys = [c["issue_key"] for c in comments]
        assert len(set(keys)) == 2 and all(keys)


class TestPlatformRejections:
    def test_rejected_issue_is_added_to_the_same_section(self):
        ok, rejected = _issue("Ok", line=1), _issue("Rejected", line=3)
        reporter = _reporter()
        key = reporter._issue_key(rejected)
        reporter.client.submit_review.return_value = SubmittedReview(id=7, rejected=[key])
        handle = reporter.publish(_result(ok, rejected), changed_files=[_file()])
        first_body = reporter.client.submit_review.call_args.kwargs["body"]
        assert "Rejected" not in first_body
        reporter.client.update_review.assert_called_once()
        body = reporter.client.update_review.call_args.args[2]
        assert "Rejected" in _section(body) and "Ok" not in _section(body)
        assert "Additional Issues" not in body
        assert handle is not None and handle.rejected == [key]

    def test_update_after_memory_phase_keeps_moved_issues(self):
        moved = _issue("Moved", filename="gone.py", line=1)
        reporter = _reporter()
        result = _result(moved, _issue("Fine", line=1))
        handle = reporter.publish(result, changed_files=[_file()])
        assert handle is not None
        reporter.client.update_review.reset_mock()
        reporter.update(handle, result)
        body = reporter.client.update_review.call_args.args[2]
        assert "Moved" in _section(body) and "Fine" not in _section(body)

    def test_rejection_with_split_posts_one_overflow_comment(self, monkeypatch):
        rejected = _issue("Rejected", line=3)
        reporter = _reporter()
        reporter.client.submit_review.return_value = SubmittedReview(
            id=7, rejected=[reporter._issue_key(rejected)]
        )
        monkeypatch.setattr(GitReporter, "_needs_split", lambda self, r, b: bool(b))
        handle = reporter.publish(_result(rejected), changed_files=[_file()])
        assert handle is not None and handle.split is True
        [comment] = [c.args[1] for c in reporter.client.post_comment.call_args_list]
        assert "Rejected" in comment and TITLE in comment
        body = reporter.client.update_review.call_args.args[2]
        assert "Rejected" not in body
        assert "posted as separate comments" in body

    def test_update_failure_does_not_lose_the_review(self):
        rejected = _issue("Rejected", line=3)
        reporter = _reporter()
        reporter.client.submit_review.return_value = SubmittedReview(
            id=7, rejected=[reporter._issue_key(rejected)]
        )
        reporter.client.update_review.side_effect = RuntimeError("boom")
        assert reporter.publish(_result(rejected)) is not None


class TestGitHubClient:
    @pytest.fixture
    def setup(self):
        from codespy.tools.git.github_client import GitHubClient

        client = GitHubClient(MagicMock())
        repo = MagicMock()
        gh_pr = MagicMock()
        gh_pr.head.sha = "h" * 40
        gh_pr.get_files.return_value = [
            MagicMock(filename="a.py", status="modified", additions=1, deletions=0,
                      patch=PATCH, previous_filename=None)
        ]
        gh_pr.create_review.return_value = MagicMock(id=5)
        repo.get_pull.return_value = gh_pr
        client._github = MagicMock()
        client._github.get_repo.return_value = repo
        return client, gh_pr

    def _c(self, key, path="a.py", line=1):
        return {"path": path, "body": key, "line": line, "issue_key": key}

    URL = "https://github.com/o/r/pull/1"

    def test_clean_submit_sends_no_issue_key_and_returns_nothing_rejected(self, setup):
        client, gh_pr = setup
        review = client.submit_review(self.URL, "BODY", [self._c("k1")])
        assert review is not None and review.rejected == []
        sent = gh_pr.create_review.call_args.kwargs
        assert sent["body"] == "BODY"
        assert all("issue_key" not in c for c in sent["comments"])

    def test_file_not_in_pr_is_rejected_not_dropped(self, setup):
        client, gh_pr = setup
        review = client.submit_review(
            self.URL, "BODY", [self._c("k1"), self._c("k2", path="gone.py")]
        )
        assert review is not None and review.rejected == ["k2"]
        assert [c["path"] for c in gh_pr.create_review.call_args.kwargs["comments"]] == ["a.py"]
        assert gh_pr.create_review.call_args.kwargs["body"] == "BODY"

    def test_invalid_line_retry_rejects_only_invalid(self, setup):
        client, gh_pr = setup
        gh_pr.create_review.side_effect = [
            RuntimeError("Line could not be resolved"),
            MagicMock(id=6),
        ]
        review = client.submit_review(
            self.URL, "BODY", [self._c("ok", line=1), self._c("bad", line=50)]
        )
        assert review is not None and review.id == 6 and review.rejected == ["bad"]
        retry = gh_pr.create_review.call_args.kwargs
        assert [c["line"] for c in retry["comments"]] == [1]
        assert retry["body"] == "BODY"

    def test_final_fallback_rejects_every_comment(self, setup):
        client, gh_pr = setup
        gh_pr.create_review.side_effect = [
            RuntimeError("Line could not be resolved"),
            RuntimeError("still failing"),
            MagicMock(id=8),
        ]
        review = client.submit_review(
            self.URL, "BODY", [self._c("ok", line=1), self._c("bad", line=50)]
        )
        assert review is not None and sorted(review.rejected) == ["bad", "ok"]
        assert gh_pr.create_review.call_args.kwargs["comments"] == []


class TestGitLabClient:
    @pytest.fixture
    def setup(self):
        import gitlab

        from codespy.tools.git.gitlab_client import GitLabClient

        client = GitLabClient(MagicMock())
        project = MagicMock()
        mr = MagicMock()
        mr.changes.return_value = {
            "diff_refs": {"head_sha": "h"},
            "changes": [{"new_path": "a.py", "diff": PATCH}],
        }
        mr.diff_refs = {"base_sha": "b", "start_sha": "s"}
        mr.notes.create.return_value = MagicMock(id=9)
        project.mergerequests.get.return_value = mr
        client._gitlab = MagicMock()
        client._gitlab.projects.get.return_value = project
        return client, mr, gitlab.exceptions.GitlabCreateError

    URL = "https://gitlab.com/g/p/-/merge_requests/1"

    def _c(self, key, path="a.py", line=1):
        return {"path": path, "body": key, "line": line, "issue_key": key}

    def test_failures_are_returned_and_body_is_unchanged(self, setup):
        client, mr, create_error = setup
        mr.discussions.create.side_effect = [None, create_error("nope")]
        review = client.submit_review(
            self.URL,
            "BODY",
            [
                self._c("ok", line=1),
                self._c("api", line=2),
                self._c("outside", line=50),
                self._c("nofile", path="gone.py"),
            ],
        )
        assert review is not None
        assert sorted(review.rejected) == ["api", "nofile", "outside"]
        assert mr.notes.create.call_args.args[0] == {"body": "BODY"}
