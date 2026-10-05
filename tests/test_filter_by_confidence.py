"""Tests for filter_by_confidence helper."""

import logging

import pytest

from codespy.agents.review.helpers import filter_by_confidence
from codespy.agents.review.models import Issue, IssueCategory, IssueSeverity


class TestFilterByConfidence:
    """Tests for the filter_by_confidence function."""

    def test_filter_keeps_issues_above_threshold(self, caplog):
        """Issues with confidence >= threshold should be kept."""
        issues = [
            Issue(
                category=IssueCategory.BUG,
                severity=IssueSeverity.HIGH,
                title="Issue 1",
                description="Description 1",
                filename="file1.py",
                confidence=0.9,
            ),
            Issue(
                category=IssueCategory.SECURITY,
                severity=IssueSeverity.CRITICAL,
                title="Issue 2",
                description="Description 2",
                filename="file2.py",
                confidence=0.6,
            ),
        ]

        with caplog.at_level(logging.INFO):
            result = filter_by_confidence(issues, 0.5, "code_review", "scope1")

        assert len(result) == 2
        assert result[0].title == "Issue 1"
        assert result[1].title == "Issue 2"
        assert "2 raw issues, 2 kept" in caplog.text

    def test_filter_drops_issues_below_threshold(self, caplog):
        """Issues with confidence < threshold should be dropped."""
        issues = [
            Issue(
                category=IssueCategory.BUG,
                severity=IssueSeverity.HIGH,
                title="Issue 1",
                description="Description 1",
                filename="file1.py",
                confidence=0.9,
            ),
            Issue(
                category=IssueCategory.SECURITY,
                severity=IssueSeverity.CRITICAL,
                title="Issue 2",
                description="Description 2",
                filename="file2.py",
                confidence=0.4,
            ),
        ]

        with caplog.at_level(logging.INFO):
            result = filter_by_confidence(issues, 0.5, "code_review", "scope1")

        assert len(result) == 1
        assert result[0].title == "Issue 1"
        assert "2 raw issues, 1 kept" in caplog.text

    def test_filter_handles_none_issues(self, caplog):
        """None issues list should return empty list."""
        with caplog.at_level(logging.INFO):
            result = filter_by_confidence(None, 0.5, "code_review", "scope1")

        assert result == []
        assert "0 raw issues, 0 kept" in caplog.text

    def test_filter_handles_empty_list(self, caplog):
        """Empty issues list should return empty list."""
        with caplog.at_level(logging.INFO):
            result = filter_by_confidence([], 0.5, "code_review", "scope1")

        assert result == []
        assert "0 raw issues, 0 kept" in caplog.text

    def test_filter_logs_module_and_scope(self, caplog):
        """Log should include module and scope."""
        issues = [
            Issue(
                category=IssueCategory.BUG,
                severity=IssueSeverity.HIGH,
                title="Issue 1",
                description="Description 1",
                filename="file1.py",
                confidence=0.9,
            ),
        ]

        with caplog.at_level(logging.INFO):
            filter_by_confidence(issues, 0.5, "supply_chain", "packages/auth")

        assert "supply_chain scope packages/auth" in caplog.text

    def test_filter_logs_min_confidence(self, caplog):
        """Log should include min_confidence value."""
        issues = [
            Issue(
                category=IssueCategory.BUG,
                severity=IssueSeverity.HIGH,
                title="Issue 1",
                description="Description 1",
                filename="file1.py",
                confidence=0.9,
            ),
        ]

        with caplog.at_level(logging.INFO):
            filter_by_confidence(issues, 0.81, "code_review", "scope1")

        assert "min_confidence=0.81" in caplog.text

    def test_issue_default_confidence_is_0_8(self):
        """Issue.confidence should default to 0.8."""
        issue = Issue(
            category=IssueCategory.BUG,
            severity=IssueSeverity.HIGH,
            title="Test Issue",
            description="Test Description",
            filename="test.py",
        )
        # Default is 0.8, which is above 0.5 but below 0.81
        assert issue.confidence == 0.8

    def test_issue_with_default_confidence_passes_new_threshold(self):
        """Issue with default confidence (0.8) should pass threshold of 0.5."""
        issue = Issue(
            category=IssueCategory.BUG,
            severity=IssueSeverity.HIGH,
            title="Test Issue",
            description="Test Description",
            filename="test.py",
        )

        result = filter_by_confidence([issue], 0.5, "code_review", "scope1")
        assert len(result) == 1

    def test_issue_with_default_confidence_fails_old_threshold(self):
        """Issue with default confidence (0.8) should fail old threshold of 0.81."""
        issue = Issue(
            category=IssueCategory.BUG,
            severity=IssueSeverity.HIGH,
            title="Test Issue",
            description="Test Description",
            filename="test.py",
        )

        result = filter_by_confidence([issue], 0.81, "code_review", "scope1")
        assert len(result) == 0
