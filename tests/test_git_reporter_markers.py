"""Reporter markers: present everywhere, and recognized back by extraction."""

from pr_conversation_fakes import comment, make_pr

from codespy.agents.review.feedback import drop_rejected, extract_feedback, issue_fingerprint
from codespy.agents.review.models import Issue, IssueCategory, IssueSeverity
from codespy.tools.git.conversation import (
    REVIEW_MARKER_RE,
    classify_author,
    parse_issue_marker,
)
from codespy.workflows.review.models import ReviewResult
from codespy.workflows.review.reporters.git import GitReporter


def _issue(line=True):
    return Issue(
        category=IssueCategory.SECURITY,
        severity=IssueSeverity.CRITICAL,
        title="SQL injection in query",
        description="Use parameters.",
        filename="svc/db.py",
        line_start=10 if line else None,
    )


def _reporter():
    return GitReporter.__new__(GitReporter)


def _result(*issues):
    return ReviewResult(pr_number=1, pr_title="t", pr_url="u", repo="o/r", model_used="m",
                        run_id="run42", issues=list(issues))


def test_review_body_has_review_marker_and_body_issue_marker():
    issue = _issue(line=False)
    body = _reporter()._build_review_body(_result(issue), [issue])
    m = REVIEW_MARKER_RE.search(body)
    assert m and "run_id=run42" in m.group(0)
    marker = parse_issue_marker(body)
    assert marker["fp"] == issue_fingerprint(issue)
    assert marker["cat"] == "security" and marker["sev"] == "critical"
    # The header still comes first (legacy detection keeps working)
    assert body.startswith("# 🔍 Code[Spy]")


def test_inline_comment_marker_and_first_line_shape():
    issue = _issue()
    [c] = _reporter()._build_inline_comments([issue], run_id="run42")
    assert c["body"].startswith("🔴 **[Critical] SQL injection in query**")
    marker = parse_issue_marker(c["body"])
    assert marker == {
        "v": "1", "run_id": "run42", "fp": issue_fingerprint(issue), "cat": "security",
        "sev": "critical",
    }


def test_round_trip_rejected_finding_is_suppressed():
    issue = _issue()
    [c] = _reporter()._build_inline_comments([issue], run_id="run42")
    origin = classify_author(c["body"], "some-user")
    assert origin == "codespy"
    pr = make_pr(comments=[comment(1, c["body"], origin=origin, kind="inline", path=c["path"],
                                   line=c["line"], reactions={"-1": 1})])
    feedback = extract_feedback(pr)
    assert [f.verdict for f in feedback] == ["rejected"]
    assert feedback[0].category == "security" and feedback[0].severity == "critical"
    assert drop_rejected([issue], feedback) == []
