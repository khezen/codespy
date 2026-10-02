"""Tests for the severity × category statistics matrix in review output."""

from codespy.agents.review.models import Issue, IssueCategory, IssueSeverity
from codespy.workflows.review.models import ReviewResult
from codespy.workflows.review.reporters.git import GitReporter


def _make_issue(severity: IssueSeverity, category: IssueCategory) -> Issue:
    """Create a minimal Issue with given severity and category."""
    return Issue(
        category=category,
        severity=severity,
        title="test issue",
        description="test description",
        filename="test.py",
    )


def _result_with_issues(*issues: Issue) -> ReviewResult:
    """Build a ReviewResult with the given issues."""
    return ReviewResult(
        pr_number=1,
        pr_title="t",
        pr_url="https://github.com/o/r/pull/1",
        repo="o/r",
        model_used="m",
        issues=list(issues),
    )


def _git_body(result: ReviewResult) -> str:
    """Build the git review body from a result."""
    reporter = GitReporter.__new__(GitReporter)
    return reporter._build_review_body(result, [])


def test_statistics_matrix_header():
    """The statistics table should have the correct header row."""
    result = _result_with_issues()
    body = _git_body(result)

    assert "| Severity | Security | Bugs | Documentation | Smells | Total |" in body
    assert "|----------|----------|------|---------------|--------|-------|" in body


def test_statistics_matrix_all_severities():
    """The statistics table should include all severity rows."""
    result = _result_with_issues()
    body = _git_body(result)

    assert "| Critical |" in body
    assert "| High |" in body
    assert "| Medium |" in body
    assert "| Low |" in body
    assert "| Info |" in body


def test_statistics_matrix_zero_issues():
    """When there are no issues, all cells should show 0."""
    result = _result_with_issues()
    body = _git_body(result)

    # Check that we have zeros in the matrix (each severity row has 6 zeros)
    # The pattern is: | Severity | 0 | 0 | 0 | 0 | 0 |
    lines = body.split("\n")
    data_rows = [l for l in lines if l.startswith("| Critical |") or l.startswith("| High |") or
                 l.startswith("| Medium |") or l.startswith("| Low |") or l.startswith("| Info |")]

    for row in data_rows:
        assert "| 0 | 0 | 0 | 0 | 0 |" in row, f"Expected zeros in row: {row}"

    # Check totals row
    assert "| **Total** | **0** | **0** | **0** | **0** | **0** |" in body


def test_statistics_matrix_counts_issues():
    """Issues should be counted in the correct severity × category cell."""
    result = _result_with_issues(
        _make_issue(IssueSeverity.CRITICAL, IssueCategory.SECURITY),
        _make_issue(IssueSeverity.HIGH, IssueCategory.SECURITY),
        _make_issue(IssueSeverity.HIGH, IssueCategory.BUG),
        _make_issue(IssueSeverity.MEDIUM, IssueCategory.DOCUMENTATION),
        _make_issue(IssueSeverity.INFO, IssueCategory.SMELL),
    )
    body = _git_body(result)

    lines = body.split("\n")
    data_rows = {l.split("|")[1].strip(): l for l in lines if l.startswith("| ") and "Severity" not in l}

    # Critical row: 1 security, 0 others
    assert "| Critical | 1 | 0 | 0 | 0 | 1 |" in body

    # High row: 1 security, 1 bug, 0 others
    assert "| High | 1 | 1 | 0 | 0 | 2 |" in body

    # Medium row: 1 documentation
    assert "| Medium | 0 | 0 | 1 | 0 | 1 |" in body

    # Info row: 1 smell
    assert "| Info | 0 | 0 | 0 | 1 | 1 |" in body

    # Totals row
    assert "| **Total** | **2** | **1** | **1** | **1** | **5** |" in body


def test_statistics_matrix_no_old_metric_table():
    """The old Metric | Count table format should not appear."""
    result = _result_with_issues()
    body = _git_body(result)

    assert "| Metric | Count |" not in body
    assert "| Total Issues |" not in body


def test_statistics_matrix_in_collapsible():
    """The statistics section should be inside a collapsible details block."""
    result = _result_with_issues()
    body = _git_body(result)

    # Check for the collapsible wrapper
    assert '<details>' in body
    assert '<summary>📊 Statistics</summary>' in body
    assert '</details>' in body
