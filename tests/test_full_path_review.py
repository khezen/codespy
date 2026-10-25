"""Tests for full-path review (`codespy review-path`) and the sparse-checkout fix."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codespy.agents.review.models import Issue, IssueCategory, IssueSeverity
from codespy.agents.review.scope.agent import (
    ScopeResolver,
    apply_sparse_checkout,
    build_sparse_patterns,
)
from codespy.tools.git.base import SubmittedPullRequest
from codespy.tools.git.client import parse_repo_url
from codespy.tools.git.full_path import (
    TreeStats,
    blob_url,
    build_pr_from_tree,
    check_caps,
    normalize_path,
    path_slug,
    report_branch,
    report_file,
    tree_url,
)
from codespy.tools.git.models import ChangedFile, FileStatus, GitPlatform
from codespy.workflows.review.models import FullPathInfo, ReviewResult
from codespy.workflows.review.reporters.pull_request import PullRequestReporter

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _make_repo(root: Path, files: dict[str, str | bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _on_disk(repo: Path) -> set[str]:
    return {
        str(p.relative_to(repo))
        for p in repo.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(repo).parts
    }


TREE = {
    "README.md": "root\n",
    "pyproject.toml": "[project]\nname='x'\n",
    "rust/Cargo.toml": "[workspace]\n",
    "rust/Cargo.lock": "# lock\n",
    "rust/.kilo/agent/a.md": "rules\n",
    "rust/crates/x/Cargo.toml": "[package]\nname='x'\n",
    "rust/crates/x/src/lib.rs": "pub fn x() {}\n",
    "rust/other/big.rs": "fn other() {}\n",
    "docs/guide.md": "doc\n",
    "services/README.md": "svc\n",
    "services/api/main.go": "package main\n",
    "services/api/internal/h.go": "package internal\n",
    "services/web/index.ts": "export {}\n",
    "ios/App.xcodeproj/project.pbxproj": "// pbx\n",
    "ios/App/x.swift": "let x = 1\n",
    "ios/Other/y.swift": "let y = 1\n",
}


# --- sparse patterns -------------------------------------------------------------------


class TestSparsePatternOrder:
    def test_root_files_only_then_negation(self):
        patterns = build_sparse_patterns(["packages/auth/src/index.ts"])
        assert patterns[:2] == ["/*", "!/*/"]

    def test_no_negation_after_positive(self):
        patterns = build_sparse_patterns(["a/b/c/d.py", "x/y/z.go"])
        last_negation = max(i for i, p in enumerate(patterns) if p.startswith("!"))
        first_recursive = patterns.index("/a/b/")
        assert last_negation < first_recursive

    def test_unique(self):
        patterns = build_sparse_patterns(["a/b/c.py", "a/b/d.py"])
        assert len(patterns) == len(set(patterns))

    def test_path_mode_root_is_full_checkout(self):
        assert build_sparse_patterns([], recursive_dirs=["."]) == ["/*"]

    def test_path_mode(self):
        patterns = build_sparse_patterns([], recursive_dirs=["services/api"])
        assert patterns[:4] == ["/*", "!/*/", "/services/", "!/services/*/"]
        assert "/services/api/" in patterns


@needs_git
class TestSparseCheckoutRealGit:
    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        return _make_repo(tmp_path / "repo", TREE)

    def test_bare_star_selects_everything(self, repo: Path):
        """Reproduces the pre-fix bug: a bare /* materialises the whole repo."""
        apply_sparse_checkout(repo, ["/*"])
        assert _on_disk(repo) == set(TREE)

    def test_pr_mode_narrows_a_full_clone(self, repo: Path):
        # Starts as a full checkout (like every cached clone made before the fix)
        apply_sparse_checkout(repo, build_sparse_patterns(["rust/crates/x/src/lib.rs"]))
        files = _on_disk(repo)
        assert "rust/crates/x/src/lib.rs" in files
        assert {"rust/Cargo.lock", "rust/Cargo.toml", "README.md", "pyproject.toml"} <= files
        assert "rust/.kilo/agent/a.md" in files
        assert "rust/other/big.rs" not in files
        assert "docs/guide.md" not in files
        assert "services/api/main.go" not in files

    def test_path_mode(self, repo: Path):
        apply_sparse_checkout(repo, build_sparse_patterns([], recursive_dirs=["services/api"]))
        files = _on_disk(repo)
        assert {"services/api/main.go", "services/api/internal/h.go"} <= files
        assert {"services/README.md", "README.md"} <= files
        assert "services/web/index.ts" not in files
        assert "rust/Cargo.toml" not in files

    def test_xcode_bundle_in_ancestor(self, repo: Path):
        apply_sparse_checkout(repo, build_sparse_patterns(["ios/App/x.swift"]))
        files = _on_disk(repo)
        assert "ios/App.xcodeproj/project.pbxproj" in files
        assert "ios/App/x.swift" in files
        assert "ios/Other/y.swift" not in files

    def test_ensure_repo_writes_patterns_before_checkout(self, tmp_path: Path, monkeypatch):
        """Treeless clones fetch blobs on checkout: the new set must be written first."""
        import codespy.agents.review.scope.agent as agent_mod

        events: list[str] = []
        monkeypatch.setattr(agent_mod, "write_sparse_patterns", lambda *a: events.append("write"))
        monkeypatch.setattr(agent_mod, "apply_sparse_checkout", lambda *a: events.append("apply"))

        fake_repo = MagicMock()
        fake_repo.git.checkout.side_effect = lambda *a: events.append("checkout")
        monkeypatch.setattr("git.Repo", lambda *a, **k: fake_repo)

        repo_path = tmp_path / "clone"
        (repo_path / ".git").mkdir(parents=True)
        pr = MagicMock(head_sha="abc", changed_files=[])
        asyncio.run(ScopeResolver()._ensure_repo(pr, repo_path, False))
        assert events == ["write", "checkout", "apply"]


# --- URLs ------------------------------------------------------------------------------


class TestParseRepoUrl:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://github.com/o/r", ("github", "github.com", "o", "r")),
            ("https://github.com/o/r.git", ("github", "github.com", "o", "r")),
            ("https://github.com/o/r/", ("github", "github.com", "o", "r")),
            ("https://gitlab.com/g/sub/p", ("gitlab", "gitlab.com", "g/sub", "p")),
            ("https://git.acme.io/team/p.git", ("gitlab", "git.acme.io", "team", "p")),
        ],
    )
    def test_valid(self, url, expected):
        assert parse_repo_url(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/o/r/pull/1",
            "https://github.com/o/r/tree/main/src",
            "https://gitlab.com/g/p/-/merge_requests/1",
            "https://gitlab.com/p",
            "git@github.com:o/r.git",
            "ftp://github.com/o/r",
            "https://github.com/../..",
            "https://github.com/o/..",
            "https://github.com/-o/r",
            "https://gitlab.com/g/../p",
        ],
    )
    def test_rejected(self, url):
        with pytest.raises(ValueError):
            parse_repo_url(url)


class TestPrUrlStable:
    """PR-mode url is the PR memory topic id: it must not change with host."""

    def _pr(self, **kw):
        from datetime import UTC, datetime

        from codespy.tools.git.models import PullRequest

        now = datetime.now(UTC)
        base = dict(
            number=3, title="t", body="", state="open", author="a", base_branch="main",
            head_branch="f", base_sha="b", head_sha="h", created_at=now, updated_at=now,
            repo_owner="o", repo_name="r", changed_files=[],
        )
        return PullRequest(**{**base, **kw})

    def test_gitlab_self_hosted_unchanged(self):
        pr = self._pr(platform=GitPlatform.GITLAB, host="git.acme.io")
        assert pr.url == "https://gitlab.com/o/r/-/merge_requests/3"

    def test_local_with_gitlab_remote_unchanged(self):
        pr = self._pr(platform=GitPlatform.GITHUB, host="gitlab.com", number=0)
        assert pr.url == "https://github.com/o/r/pull/0"

    def test_web_url_override(self):
        assert self._pr(platform=GitPlatform.GITHUB, web_url="https://x/tree/s").url == (
            "https://x/tree/s"
        )


class TestUrlsAndSlugs:
    def test_slug(self):
        assert path_slug(".") == "root"
        assert path_slug("services/api") == "services-api"
        assert report_branch("services/api") == "codespy/review/services-api"
        assert report_file(".") == ".codespy/reviews/root.md"

    def test_normalize(self):
        assert normalize_path("") == "."
        assert normalize_path("/services/api/") == "services/api"
        assert normalize_path("./") == "."
        with pytest.raises(ValueError):
            normalize_path("../etc")

    def test_blob_links(self):
        gh = blob_url("github.com", "o", "r", "abc", "a/b.py", "github", 10, 20)
        assert gh == "https://github.com/o/r/blob/abc/a/b.py#L10-L20"
        gl = blob_url("gitlab.com", "g/s", "p", "abc", "a/b.py", "gitlab", 10, 20)
        assert gl == "https://gitlab.com/g/s/p/-/blob/abc/a/b.py#L10-20"
        assert blob_url("github.com", "o", "r", "abc", "f", "github", 3).endswith("#L3")

    def test_tree_links(self):
        assert tree_url("github.com", "o", "r", "s", ".", GitPlatform.GITHUB).endswith("/tree/s")
        assert tree_url("h", "g", "p", "s", "src", GitPlatform.GITLAB) == "https://h/g/p/-/tree/s/src"


# --- tree -> synthetic PR ----------------------------------------------------------------


@needs_git
class TestBuildPrFromTree:
    def _build(self, repo: Path, path: str = "."):
        return build_pr_from_tree(
            repo_path=repo,
            path=path,
            ref="main",
            sha="a" * 40,
            owner="o",
            repo="r",
            host="github.com",
            platform=GitPlatform.GITHUB,
            excluded_dirs=["node_modules"],
        )

    def test_synthetic_patch_and_skips(self, tmp_path: Path):
        repo = _make_repo(
            tmp_path / "r",
            {
                "src/a.py": "x = 1\ny = 2\nz = 3\n",
                "src/logo.png": b"\x89PNG\x00",
                "src/blob.dat2": b"ab\x00cd",
                "src/latin.txt": "caf\xe9".encode("latin-1"),
                "poetry.lock": "lock\n",
                "node_modules/m/index.js": "x\n",
                ".codespy/reviews/root.md": "# old report\n",
                "big.txt": "a" * (600 * 1024),
            },
        )
        pr, stats = self._build(repo)
        assert [f.filename for f in pr.changed_files] == ["src/a.py"]
        f = pr.changed_files[0]
        assert f.status == FileStatus.ADDED
        assert f.patch.startswith("@@ -0,0 +1,3 @@")
        assert f.valid_new_line_numbers == {1, 2, 3}
        assert stats.files_reviewed == 1
        assert stats.files_skipped == 7
        assert stats.skipped["codespy report"] == 1
        assert pr.url == "https://github.com/o/r/tree/" + "a" * 40
        assert pr.number == 0

    def test_subpath_only(self, tmp_path: Path):
        repo = _make_repo(tmp_path / "r", {"a/x.py": "1\n", "b/y.py": "2\n"})
        pr, _ = self._build(repo, "a")
        assert [f.filename for f in pr.changed_files] == ["a/x.py"]

    def test_symlinks_never_followed(self, tmp_path: Path):
        secret = tmp_path / "secret.txt"
        secret.write_text("TOKEN=hunter2\n")
        repo = _make_repo(tmp_path / "r", {"src/a.py": "x = 1\n"})
        (repo / "src" / "notes.txt").symlink_to(secret)  # absolute, outside the repo
        (repo / "src" / "alias.py").symlink_to("a.py")  # relative, inside the repo
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "links")
        pr, stats = self._build(repo)
        assert [f.filename for f in pr.changed_files] == ["src/a.py"]
        assert all("hunter2" not in (f.patch or "") for f in pr.changed_files)
        assert stats.skipped["symlink or not a regular file"] == 2

    def test_caps_count_only_files_left_after_skips(self, tmp_path: Path):
        files: dict[str, str | bytes] = {"src/a.py": "x = 1\n", "src/b.py": "y = 2\n"}
        files.update({f"node_modules/m{i}/index.js": "x\n" for i in range(50)})
        files.update({f"assets/i{i}.png": b"\x89PNG\x00" for i in range(50)})
        files.update({"poetry.lock": "lock\n", ".codespy/reviews/root.md": "# r\n"})
        repo = _make_repo(tmp_path / "r", files)
        (repo / ".gitignore").write_text("build/\n")
        (repo / "build").mkdir()
        (repo / "build" / "out.py").write_text("ignored = True\n")  # untracked, gitignored
        _, stats = self._build(repo)
        assert stats.files_reviewed == 2
        assert stats.files_skipped == 102
        check_caps(stats, max_files=2, max_bytes=10_000)  # 104 tracked, but only 2 count
        with pytest.raises(ValueError, match="after skipping 102 ignored files"):
            check_caps(stats, max_files=1, max_bytes=10_000)

    def test_empty_path_fails(self, tmp_path: Path):
        repo = _make_repo(tmp_path / "r", {"a/x.py": "1\n"})
        with pytest.raises(ValueError, match="No tracked files"):
            self._build(repo, "nope")


class TestCaps:
    def test_files_cap(self):
        stats = TreeStats(files_reviewed=5, bytes_reviewed=10)
        with pytest.raises(ValueError, match="max_files=4"):
            check_caps(stats, max_files=4, max_bytes=100)

    def test_bytes_cap(self):
        stats = TreeStats(files_reviewed=1, bytes_reviewed=101)
        with pytest.raises(ValueError, match="max_bytes=100"):
            check_caps(stats, max_files=4, max_bytes=100)

    def test_within(self):
        check_caps(TreeStats(files_reviewed=4, bytes_reviewed=100), max_files=4, max_bytes=100)


# --- scope resolution in path mode ------------------------------------------------------


class TestScopeResolverPathMode:
    def test_ensure_repo_skipped(self, tmp_path: Path, monkeypatch):
        import codespy.agents.review.scope.agent as agent_mod

        called = MagicMock()
        monkeypatch.setattr(agent_mod, "get_client", called)
        resolver = ScopeResolver()
        asyncio.run(resolver._ensure_repo(MagicMock(), tmp_path, False, is_full_path=True))
        called.assert_not_called()

    def test_every_manifest_scope_has_dependencies_changed(self, tmp_path: Path):
        for rel in ["services/api/go.mod", "packages/auth/package.json"]:
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_text("{}\n" if rel.endswith(".json") else "module x\n")
        files = [
            ChangedFile(filename=f, status=FileStatus.ADDED, patch="@@ -0,0 +1,1 @@\n+x")
            for f in [
                "services/api/go.mod",
                "services/api/main.go",
                "packages/auth/package.json",
                "packages/auth/index.ts",
            ]
        ]
        scopes, orphans = ScopeResolver()._resolve(tmp_path, files, "github.com/o/r")
        by_root = {s.subroot: s for s in scopes}
        assert set(by_root) == {"services/api", "packages/auth"}
        assert not orphans
        assert all(s.package_manifest.dependencies_changed for s in scopes)


# --- report rendering -------------------------------------------------------------------


def _result(n_issues: int = 2, platform: str = "github", desc: str = "Fix it.") -> ReviewResult:
    host = "github.com" if platform == "github" else "gitlab.com"
    issues = [
        Issue(
            category=IssueCategory.BUG,
            severity=IssueSeverity.HIGH if i % 2 else IssueSeverity.LOW,
            title=f"Issue {i}",
            description=desc,
            filename=f"src/f{i}.py",
            line_start=10,
            line_end=12,
            confidence=0.9,
        )
        for i in range(n_issues)
    ]
    return ReviewResult(
        pr_number=0,
        pr_title="Full review of src @ main",
        pr_url=f"https://{host}/o/r/tree/{'s' * 40}/src",
        repo="o/r",
        run_id="r" * 32,
        model_used="m",
        issues=issues,
        overall_summary="Summary.",
        recommendation="APPROVE",
        full_path=FullPathInfo(
            path="src",
            ref="main",
            sha="s" * 40,
            platform=platform,
            host=host,
            owner="o",
            repo="r",
            files_reviewed=3,
            files_skipped=1,
            skipped={"not UTF-8": 1},
            scopes=["src"],
        ),
    )


def _reporter() -> PullRequestReporter:
    reporter = PullRequestReporter.__new__(PullRequestReporter)
    reporter.repo_url = "https://github.com/o/r"
    reporter.client = MagicMock(platform_name="GitHub")
    return reporter


class TestReportRendering:
    def test_markdown_path_mode(self):
        md = _result().to_markdown()
        assert "**PR:**" not in md
        assert "# Full Review: src @ main" in md
        assert f"](https://github.com/o/r/tree/{'s' * 40}/src)" in md
        assert f"https://github.com/o/r/blob/{'s' * 40}/src/f0.py#L10-L12" in md
        assert "## Scope of this review" in md
        assert "not UTF-8: 1" in md

    def test_markdown_gitlab_links(self):
        md = _result(platform="gitlab").to_markdown()
        assert f"https://gitlab.com/o/r/-/blob/{'s' * 40}/src/f0.py#L10-12" in md

    def test_body_has_markers_and_all_issues(self):
        body = _reporter().build_body(_result(4))
        assert "<!-- codespy:review" in body
        assert body.count("<!-- codespy:issue") == 4
        assert "did not fit" not in body
        assert ".codespy/reviews/src.md" in body
        # Highest severity first
        assert body.index("[High]") < body.index("[Low]")

    def test_body_sections_collapsible(self):
        body = _reporter().build_body(_result(4))
        assert body.startswith("# 🔍 Code[Spy]")
        assert "<summary>🗂️ Scope of this review</summary>" in body
        assert "<summary>📋 Summary</summary>" in body
        assert "<summary>💡 Recommendation</summary>" in body
        assert "<summary>🟠 High (" in body
        assert "## Summary" not in body
        assert "## Scope of this review" not in body
        assert "## Recommendation" not in body
        assert body.count("<details>") == body.count("</details>")

    def test_body_truncates_with_link(self):
        result = _result(400, desc="x" * 400)
        body = _reporter().build_body(result)
        assert len(body) <= PullRequestReporter.MAX_BODY_CHARS
        assert body.count("<details>") == body.count("</details>")
        assert "did not fit in this description" in body
        assert "https://github.com/o/r/blob/codespy/review/src/.codespy/reviews/src.md" in body
        # The committed file keeps everything
        assert result.to_markdown().count("#### Issue") == 400

    def test_publish_and_update(self):
        reporter = _reporter()
        handle = SubmittedPullRequest(number=7, url="https://github.com/o/r/pull/7", created=True)
        reporter.client.upsert_report_pull_request.return_value = handle
        result = _result()
        assert reporter.publish(result) is handle
        kwargs = reporter.client.upsert_report_pull_request.call_args.kwargs
        assert kwargs["head_branch"] == "codespy/review/src"
        assert kwargs["file_path"] == ".codespy/reviews/src.md"
        assert kwargs["base_branch"] == "main"
        assert kwargs["base_sha"] == "s" * 40
        assert kwargs["file_content"] == result.to_markdown()
        reporter.update(handle, result)
        reporter.client.update_pull_request_body.assert_called_once()
        assert reporter.client.update_pull_request_body.call_args.args[0] == handle.url


# --- platform upsert --------------------------------------------------------------------


UPSERT = dict(
    owner="o",
    repo="r",
    base_branch="main",
    base_sha="base",
    head_branch="codespy/review/src",
    file_path=".codespy/reviews/src.md",
    file_content="# report",
    commit_message="msg",
    title="t",
    body="b",
    update_comment="c",
)


class TestGitHubUpsert:
    @pytest.fixture
    def setup(self):
        from github import GithubException

        from codespy.tools.git.github_client import GitHubClient

        client = GitHubClient(MagicMock())
        gh_repo = MagicMock()
        client._github = MagicMock()
        client._github.get_repo.return_value = gh_repo
        not_found = GithubException(404, {"message": "Not Found"}, None)
        return client, gh_repo, not_found

    def test_creates_branch_file_and_pr(self, setup):
        client, gh_repo, not_found = setup
        gh_repo.get_pulls.return_value = []
        gh_repo.get_git_ref.side_effect = not_found
        gh_repo.get_contents.side_effect = not_found
        gh_repo.create_pull.return_value = MagicMock(number=5, html_url="u5")
        handle = client.upsert_report_pull_request(**UPSERT)
        assert handle == SubmittedPullRequest(number=5, url="u5", created=True)
        gh_repo.create_git_ref.assert_called_once_with("refs/heads/codespy/review/src", "base")
        gh_repo.create_file.assert_called_once()
        assert gh_repo.create_file.call_args.kwargs["branch"] == "codespy/review/src"

    def test_updates_open_pr(self, setup):
        client, gh_repo, _ = setup
        pr = MagicMock(number=9, html_url="u9")
        gh_repo.get_pulls.return_value = [pr]
        gh_repo.get_contents.return_value = MagicMock(sha="blob")
        handle = client.upsert_report_pull_request(**UPSERT)
        assert handle == SubmittedPullRequest(number=9, url="u9", created=False)
        gh_repo.update_file.assert_called_once()
        assert gh_repo.update_file.call_args.kwargs["sha"] == "blob"
        pr.edit.assert_called_once_with(body="b")
        pr.create_issue_comment.assert_called_once_with("c")
        gh_repo.create_git_ref.assert_not_called()
        gh_repo.create_pull.assert_not_called()

    def test_stale_branch_is_reset(self, setup):
        client, gh_repo, not_found = setup
        gh_repo.get_pulls.return_value = []
        ref = MagicMock()
        gh_repo.get_git_ref.return_value = ref
        gh_repo.get_contents.side_effect = not_found
        gh_repo.create_pull.return_value = MagicMock(number=1, html_url="u")
        client.upsert_report_pull_request(**UPSERT)
        ref.edit.assert_called_once_with("base", force=True)
        gh_repo.create_git_ref.assert_not_called()

    def test_cleanup_only_created_branch(self, setup):
        client, gh_repo, not_found = setup
        gh_repo.get_pulls.return_value = []
        created_ref = MagicMock()
        gh_repo.get_git_ref.side_effect = [not_found, created_ref]
        gh_repo.get_contents.side_effect = not_found
        gh_repo.create_pull.side_effect = RuntimeError("forbidden")
        with pytest.raises(RuntimeError):
            client.upsert_report_pull_request(**UPSERT)
        created_ref.delete.assert_called_once()

    def test_no_cleanup_of_existing_branch(self, setup):
        client, gh_repo, not_found = setup
        gh_repo.get_pulls.return_value = []
        ref = MagicMock()
        gh_repo.get_git_ref.return_value = ref
        gh_repo.get_contents.side_effect = not_found
        gh_repo.create_pull.side_effect = RuntimeError("forbidden")
        with pytest.raises(RuntimeError):
            client.upsert_report_pull_request(**UPSERT)
        ref.delete.assert_not_called()

    def test_resolve_branch_rejects_non_branch(self, setup):
        client, gh_repo, not_found = setup
        gh_repo.get_branch.side_effect = not_found
        with pytest.raises(ValueError, match="not a branch"):
            client.resolve_branch("o", "r", "v1.0")


class TestGitLabUpsert:
    @pytest.fixture
    def setup(self):
        import gitlab

        from codespy.tools.git.gitlab_client import GitLabClient

        client = GitLabClient(MagicMock())
        project = MagicMock()
        client._gitlab = MagicMock()
        client._gitlab.projects.get.return_value = project
        return client, project, gitlab.exceptions.GitlabGetError

    def test_creates_branch_file_and_mr(self, setup):
        client, project, get_error = setup
        project.mergerequests.list.return_value = []
        project.branches.get.side_effect = get_error()
        project.files.get.side_effect = get_error()
        project.mergerequests.create.return_value = MagicMock(iid=3, web_url="u3")
        handle = client.upsert_report_pull_request(**UPSERT)
        assert handle == SubmittedPullRequest(number=3, url="u3", created=True)
        project.branches.create.assert_called_once_with(
            {"branch": "codespy/review/src", "ref": "base"}
        )
        action = project.commits.create.call_args.args[0]["actions"][0]["action"]
        assert action == "create"
        project.branches.delete.assert_not_called()

    def test_updates_open_mr(self, setup):
        client, project, _ = setup
        project.mergerequests.list.return_value = [MagicMock(iid=4)]
        mr = MagicMock(iid=4, web_url="u4")
        project.mergerequests.get.return_value = mr
        handle = client.upsert_report_pull_request(**UPSERT)
        assert handle == SubmittedPullRequest(number=4, url="u4", created=False)
        assert project.commits.create.call_args.args[0]["actions"][0]["action"] == "update"
        assert mr.description == "b"
        mr.save.assert_called_once()
        mr.notes.create.assert_called_once_with({"body": "c"})
        project.branches.create.assert_not_called()

    def test_stale_branch_recreated_and_cleanup_on_failure(self, setup):
        client, project, get_error = setup
        project.mergerequests.list.return_value = []
        project.files.get.side_effect = get_error()
        project.mergerequests.create.side_effect = RuntimeError("forbidden")
        with pytest.raises(RuntimeError):
            client.upsert_report_pull_request(**UPSERT)
        # Stale branch deleted, recreated at base, then cleaned up after the failure
        assert project.branches.delete.call_count == 2
        project.branches.create.assert_called_once()

    def test_resolve_branch_rejects_non_branch(self, setup):
        client, project, get_error = setup
        project.branches.get.side_effect = get_error()
        with pytest.raises(ValueError, match="not a branch"):
            client.resolve_branch("g", "p", "v1.0")


class TestConfig:
    def test_env_names(self, monkeypatch, tmp_path):
        from codespy.config import Settings

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("REVIEW_FULL_PATH_MAX_FILES", "7")
        monkeypatch.setenv("REVIEW_FULL_PATH_MAX_BYTES", "99")
        caps = Settings().review.full_path
        assert (caps.max_files, caps.max_bytes) == (7, 99)

    def test_not_a_signature(self):
        from codespy.config_dspy import SIGNATURE_NAMES

        assert "full_path" not in SIGNATURE_NAMES


class TestCodeInterpreterPreflight:
    """Without Deno every RLM agent soft-fails to "no findings": fail before any work."""

    def _pipeline(self):
        from codespy.workflows.review.pipeline import ReviewPipeline

        pipeline = ReviewPipeline.__new__(ReviewPipeline)
        return pipeline

    def test_missing_deno_fails_fast(self, monkeypatch):
        import dspy.primitives.python_interpreter as pi

        monkeypatch.setattr(pi, "_find_deno_executable", lambda: "deno-missing")
        with pytest.raises(ValueError, match="Code interpreter unavailable.*Deno 2.x"):
            self._pipeline()._verify_code_interpreter()

    def test_forward_checks_before_clone(self, monkeypatch):
        from codespy.workflows.review.models import PathReviewConfig

        pipeline = self._pipeline()
        pipeline.cost_tracker = MagicMock()
        pipeline._verify_code_interpreter = MagicMock(side_effect=ValueError("no deno"))
        pipeline._build_path_pr = MagicMock()
        with pytest.raises(ValueError, match="no deno"):
            pipeline.forward(PathReviewConfig(repo_url="https://github.com/o/r"))
        pipeline._build_path_pr.assert_not_called()

    def test_valid_deno_passes(self, monkeypatch):
        import dspy.primitives.python_interpreter as pi

        monkeypatch.setattr(pi, "_find_deno_executable", lambda: "deno")
        monkeypatch.setattr(pi, "_validate_deno_version", lambda exe: None)
        self._pipeline()._verify_code_interpreter()
