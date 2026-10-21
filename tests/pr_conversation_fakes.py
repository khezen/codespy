"""Shared fakes for PR conversation / review feedback tests."""

from datetime import UTC, datetime, timedelta

from codespy.tools.git.models import GitPlatform, PRComment, PRCommit, PullRequest

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def ts(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def make_pr(comments=None, commits=None, reactions=None) -> PullRequest:
    return PullRequest(
        number=7,
        title="Add cache",
        body="Adds a cache",
        state="open",
        author="alice",
        base_branch="main",
        head_branch="feat",
        base_sha="a" * 40,
        head_sha="b" * 40,
        created_at=T0,
        updated_at=T0,
        repo_owner="o",
        repo_name="r",
        host="github.com",
        platform=GitPlatform.GITHUB,
        comments=list(comments or []),
        commits=list(commits or []),
        reactions=dict(reactions or {}),
    )


def comment(id, body, *, origin="human", kind="issue", author="bob", minutes=0, **kw) -> PRComment:
    # codespy comments default to platform-verified authorship (posted by the token's account)
    kw.setdefault("authored_by_self", origin == "codespy")
    return PRComment(
        id=id, kind=kind, author=author, body=body, created_at=ts(minutes), origin=origin, **kw
    )


def commit(sha, message, author="alice") -> PRCommit:
    return PRCommit(sha=sha, author=author, message=message)
