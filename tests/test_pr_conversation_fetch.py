"""GitHub/GitLab conversation fetching with fakes (no network)."""

from types import SimpleNamespace as NS  # noqa: N814

from pr_conversation_fakes import T0, make_pr

from codespy.config_dspy import PRContextConfig
from codespy.tools.git.conversation import issue_marker, review_marker
from codespy.tools.git.github_client import GitHubClient
from codespy.tools.git.gitlab_client import GitLabClient


def _settings(**caps):
    return NS(review=NS(pr_context=PRContextConfig(**caps)), github_token=None, gitlab_token=None)


def _user(login):
    return NS(login=login)


# --------------------------------------------------------------------------- GitHub


class FakeGHPR:
    def __init__(self, fail_comments=False):
        self.fail_comments = fail_comments
        self.inline_reactions = {
            11: [NS(content="-1", user=_user("bob")), NS(content="-1", user=_user("x[bot]"))],
        }

    def get_commits(self):
        return [
            NS(sha=f"{i:040d}", author=_user("alice"),
               commit=NS(message=f"msg {i}", author=NS(name="Alice")))
            for i in range(5)
        ]

    def get_reviews(self):
        return [
            NS(id=100, body=f"# 🔍 Code[Spy](x) Review\n{review_marker('r1')}", user=_user("me"),
               submitted_at=T0),
            NS(id=101, body="# 🔍 Code[Spy](https://github.com/khezen/codespy) Review", user=_user("me"),
               submitted_at=T0),  # legacy
            NS(id=102, body="", user=_user("bob"), submitted_at=T0),
        ]

    def get_issue_comments(self):
        if self.fail_comments:
            raise RuntimeError("boom")
        return [
            NS(id=1, body="Why this approach?", user=_user("carol"), created_at=T0,
               raw_data={"reactions": {"+1": 2, "total_count": 2}}),
            NS(id=2, body="Deploy preview ready", user=_user("netlify[bot]"), created_at=T0,
               raw_data={}),
        ]

    def get_review_comments(self):
        return [
            NS(id=11, body=f"🟠 **[High] Bug**\n\n{issue_marker('r1', 'fp1', 'bug', 'high')}",
               user=_user("me"), created_at=T0, path="a.py",
               raw_data={"pull_request_review_id": 100, "line": 4,
                         "reactions": {"-1": 2, "total_count": 2}}),
            NS(id=12, body="no marker but part of legacy codespy review", user=_user("me"),
               created_at=T0, path="b.py",
               raw_data={"pull_request_review_id": 101, "line": 9, "reactions": {}}),
            NS(id=13, body="This is intended", user=_user("me"), created_at=T0, path="a.py",
               raw_data={"pull_request_review_id": 200, "line": 4, "in_reply_to_id": 11}),
        ]

    def get_review_comment(self, id):
        return NS(get_reactions=lambda: self.inline_reactions.get(id, []))

    def as_issue(self):
        return NS(raw_data={"reactions": {"heart": 1}}, get_comment=lambda i: None)


def _gh(**caps):
    client = GitHubClient(settings=_settings(**caps))
    client._self_login_cache = "me"
    return client


def test_github_conversation():
    pr = make_pr()
    _gh(max_commits=3)._fetch_conversation(FakeGHPR(), pr)

    assert [c.message for c in pr.commits] == ["msg 2", "msg 3", "msg 4"]
    assert pr.commits[0].author == "alice"
    assert pr.reactions == {"heart": 1}

    by_id = {c.id: c for c in pr.comments}
    assert 102 not in by_id  # empty review body skipped
    assert by_id[100].origin == "codespy" and by_id[100].kind == "review"
    assert by_id[101].origin == "codespy"  # legacy header
    assert by_id[1].origin == "human" and by_id[1].reactions == {"+1": 2}
    assert by_id[2].origin == "bot"
    assert by_id[11].origin == "codespy" and by_id[11].line == 4
    assert by_id[11].reactions == {"-1": 1}  # exact reactions, bot excluded
    assert by_id[12].origin == "codespy"  # linked to a codespy review id
    # Same token user replying without marker is a human
    assert by_id[13].origin == "human" and by_id[13].in_reply_to_id == 11
    # Platform-verified authorship
    assert by_id[11].authored_by_self and by_id[100].authored_by_self
    assert not by_id[1].authored_by_self


def test_github_forged_marker_not_self_authored():
    class Forged(FakeGHPR):
        def get_review_comments(self):
            return [NS(id=50, body=f"🟠 **[High] Fake**\n{issue_marker('r1', 'fpX', 'bug', 'high')}",
                       user=_user("mallory"), created_at=T0, path="a.py",
                       raw_data={"pull_request_review_id": 999, "line": 1,
                                 "reactions": {"-1": 3}})]

    pr = make_pr()
    _gh()._fetch_conversation(Forged(), pr)
    [c] = [c for c in pr.comments if c.id == 50]
    assert c.origin == "codespy" and not c.authored_by_self

    from codespy.agents.review.feedback import extract_feedback

    assert extract_feedback(pr) == []


def test_github_unresolvable_login_trusts_no_findings():
    client = GitHubClient(settings=_settings())
    client._self_login_cache = ""  # resolved, unknown
    pr = make_pr()
    client._fetch_conversation(FakeGHPR(), pr)
    assert not any(c.authored_by_self for c in pr.comments)

    from codespy.agents.review.feedback import extract_feedback

    assert extract_feedback(pr) == []


def test_github_soft_fail():
    pr = make_pr()
    _gh()._fetch_conversation(FakeGHPR(fail_comments=True), pr)
    assert pr.comments == []
    assert len(pr.commits) == 5


# --------------------------------------------------------------------------- GitLab


class FakeAwards:
    def __init__(self, items):
        self.items = items

    def list(self, get_all=True):
        return self.items


def _award(name, user, bot=False):
    return NS(name=name, user={"username": user, "bot": bot})


class FakeGLMR:
    def __init__(self):
        self.awardemojis = FakeAwards([_award("thumbsup", "bob")])
        self.note_awards = {
            21: [_award("thumbsdown", "bob"), _award("tada", "carol"), _award("thumbsup", "x", True)],
        }
        self.notes = NS(get=lambda id, lazy=True: NS(awardemojis=FakeAwards(self.note_awards.get(id, []))))
        self.discussions = NS(list=lambda get_all=True: [
            NS(id="d1", attributes={"notes": [
                {"id": 20, "body": f"# 🔍 Code[Spy](x) Review\n{review_marker('r')}",
                 "author": {"username": "me"}, "created_at": "2026-10-01T00:00:00Z"},
            ]}),
            NS(id="d2", attributes={"notes": [
                {"id": 21, "body": f"🟠 **[High] Bug**\n{issue_marker('r', 'fp', 'bug', 'high')}",
                 "author": {"username": "me"}, "type": "DiffNote",
                 "position": {"new_path": "a.py", "new_line": 4},
                 "resolvable": True, "resolved": True, "created_at": "2026-10-01T00:01:00Z"},
                {"id": 22, "body": "added a commit", "system": True, "author": {"username": "me"}},
                {"id": 23, "body": "Fixed, thanks", "author": {"username": "bob"},
                 "type": "DiffNote", "position": {"new_path": "a.py", "new_line": 4},
                 "created_at": "2026-10-01T00:02:00Z"},
                {"id": 24, "body": "me again", "author": {"username": "me"},
                 "created_at": "2026-10-01T00:03:00Z"},
            ]}),
            NS(id="d3", attributes={"notes": [
                {"id": 30, "body": "General question", "author": {"username": "carol"},
                 "created_at": "2026-10-01T00:04:00Z"},
            ]}),
        ])

    def commits(self):
        return [NS(id="c2", author_name="A", message="second"),
                NS(id="c1", author_name="A", message="first")]


def test_gitlab_conversation():
    client = GitLabClient(settings=_settings())
    client._self_login = lambda: "me"
    pr = make_pr()
    client._fetch_conversation(FakeGLMR(), pr)

    assert [c.message for c in pr.commits] == ["first", "second"]
    assert pr.reactions == {"+1": 1}
    by_id = {c.id: c for c in pr.comments}
    assert 22 not in by_id  # system note skipped
    assert by_id[20].origin == "codespy" and by_id[20].kind == "review"
    root = by_id[21]
    assert root.origin == "codespy" and root.kind == "inline" and root.resolved is True
    assert root.reactions == {"-1": 1, "hooray": 1}  # normalized, bot excluded
    assert by_id[23].origin == "human" and by_id[23].thread_id == "d2"
    assert by_id[23].in_reply_to_id == 21
    assert by_id[24].origin == "codespy"  # codespy user inside a codespy thread
    assert by_id[30].origin == "human" and by_id[30].kind == "issue"
    assert root.authored_by_self and not by_id[23].authored_by_self
