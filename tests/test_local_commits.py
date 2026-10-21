"""Commit messages in local mode."""

import subprocess

from codespy.tools.git.local_diff import build_pr_from_diff


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "Tester")
    (tmp_path / "a.txt").write_text("a\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "checkout", "-q", "-b", "feat")
    for i in range(3):
        (tmp_path / "a.txt").write_text(f"a{i}\n")
        _git(tmp_path, "commit", "-q", "-am", f"change {i}\n\nbody line {i}")
    return tmp_path


def test_commits_oldest_first_with_bodies(tmp_path):
    pr = build_pr_from_diff(_repo(tmp_path), base_ref="main")
    assert [c.message.splitlines()[0] for c in pr.commits] == ["change 0", "change 1", "change 2"]
    assert pr.commits[0].message.endswith("body line 0")
    assert pr.commits[0].author == "Tester"
    assert len(pr.commits[0].sha) == 40


def test_max_commits_keeps_most_recent(tmp_path):
    pr = build_pr_from_diff(_repo(tmp_path), base_ref="main", max_commits=2)
    assert [c.message.splitlines()[0] for c in pr.commits] == ["change 1", "change 2"]


def test_no_commits_for_uncommitted(tmp_path):
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("dirty\n")
    pr = build_pr_from_diff(repo, include_uncommitted=True)
    assert pr.commits == []
