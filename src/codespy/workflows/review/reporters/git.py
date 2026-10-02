"""Git reporter for posting review comments to GitHub/GitLab."""

import logging
import time
from typing import TYPE_CHECKING

from collections import Counter

from codespy.agents.review.models import Issue, IssueCategory, IssueSeverity
from codespy.workflows.review.models import ReviewResult
from codespy.workflows.review.reporters.base import BaseReporter
from codespy.tools.git.base import SubmittedReview
from codespy.tools.git.client import get_client

if TYPE_CHECKING:
    from codespy.config import Settings

logger = logging.getLogger(__name__)


class GitReporter(BaseReporter):
    """Reporter that posts review results to GitHub PRs or GitLab MRs."""

    SEVERITY_EMOJI = {
        IssueSeverity.CRITICAL: "🔴",
        IssueSeverity.HIGH: "🟠",
        IssueSeverity.MEDIUM: "🟡",
        IssueSeverity.LOW: "🔵",
        IssueSeverity.INFO: "⚪",
    }

    # Review body budget, below GitHub's 65,536 char limit
    MAX_BODY_CHARS = 60000

    # Retry configuration for update
    UPDATE_RETRY_DELAYS = [2.0, 4.0, 8.0]  # seconds

    def __init__(
        self,
        url: str,
        settings: "Settings | None" = None,
    ) -> None:
        """Initialize Git reporter.

        Args:
            url: Merge request URL (GitHub PR or GitLab MR)
            settings: Application settings.
        """
        self.url = url
        self.client = get_client(url, settings)

    def report(self, result: ReviewResult) -> None:
        """Post review result to the merge request (backward compatibility).

        Args:
            result: The review result to post.
        """
        self.publish(result)

    def publish(self, result: ReviewResult) -> SubmittedReview | None:
        """Publish the review to the merge request.

        Posts the initial review after the audit phase (before memory phase).
        The review will show "Memory: pending" in costs if memory_pending is True.

        Args:
            result: The review result to publish.

        Returns:
            SubmittedReview handle for later update, or None if publish failed.
        """
        # Separate issues with and without line numbers
        inline_issues: list[Issue] = []
        body_issues: list[Issue] = []

        for issue in result.issues:
            if issue.line_start is not None:
                inline_issues.append(issue)
            else:
                body_issues.append(issue)

        # Build review body with collapsible sections
        body = self._build_review_body(result, body_issues)

        # Build inline comments
        comments = self._build_inline_comments(inline_issues)

        # Submit the review
        handle = self.client.submit_review(
            url=self.url,
            body=body,
            comments=comments,
        )

        if handle:
            logger.info(
                f"Posted {self.client.platform_name} review with {len(comments)} inline comments "
                f"and {len(body_issues)} issues in body"
            )
        return handle

    def update(self, handle: SubmittedReview, result: ReviewResult) -> None:
        """Update an existing review with new costs after memory phase.

        Rebuilds the body with full costs and edits the review.
        Retries up to 3 times with exponential backoff on failure.

        Args:
            handle: The SubmittedReview returned by publish().
            result: The updated review result with full costs.
        """
        # Separate issues with and without line numbers
        inline_issues: list[Issue] = []
        body_issues: list[Issue] = []

        for issue in result.issues:
            if issue.line_start is not None:
                inline_issues.append(issue)
            else:
                body_issues.append(issue)

        # Build review body, reserving space for the body suffix
        body = self._build_review_body(result, body_issues, reserve=len(handle.body_suffix))

        # Append the body suffix (fallback comments from original submission)
        if handle.body_suffix:
            body = body + handle.body_suffix

        # Try to update with retries
        last_error: Exception | None = None
        for attempt, delay in enumerate(self.UPDATE_RETRY_DELAYS):
            try:
                self.client.update_review(self.url, handle, body)
                logger.info(f"Updated {self.client.platform_name} review with full costs")
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Update attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                )
                time.sleep(delay)

        # All retries exhausted
        logger.warning(
            f"Failed to update {self.client.platform_name} review after "
            f"{len(self.UPDATE_RETRY_DELAYS)} attempts. Keeping original post."
        )
        # Log the actual error at debug level
        if last_error:
            logger.debug("Final update error", exc_info=last_error)

    def _build_review_body(
        self,
        result: ReviewResult,
        body_issues: list[Issue],
        reserve: int = 0,
    ) -> str:
        """Build the review body with collapsible sections.

        Args:
            result: The review result.
            body_issues: Issues without line numbers to include in body.
            reserve: Number of characters to reserve for body suffix (e.g., appended comments).

        Returns:
            Formatted markdown string for review body.
        """
        lines = []

        # Header with stats - link to CodeSpy repo
        lines.append("# 🔍 Code[Spy](https://github.com/khezen/codespy) Review")
        lines.append("")
        lines.append(
            f"**Issues Found:** {result.total_issues} | "
            f"**Critical:** {len(result.critical_issues)} | "
            f"**High:** {len([i for i in result.issues if i.severity == IssueSeverity.HIGH])} | "
            f"**Medium:** {len([i for i in result.issues if i.severity == IssueSeverity.MEDIUM])}"
        )
        lines.append("")

        # Record insertion point for memories section (after header)
        insert_at = len(lines)

        # Summary section
        if result.overall_summary:
            lines.extend(
                [
                    "<details>",
                    "<summary>📋 Summary</summary>",
                    "",
                    result.overall_summary,
                    "",
                    "</details>",
                    "",
                ]
            )

        # Quality Assessment section
        if result.quality_assessment:
            lines.extend(
                [
                    "<details>",
                    "<summary>🎯 Quality Assessment</summary>",
                    "",
                    result.quality_assessment,
                    "",
                    "</details>",
                    "",
                ]
            )

        # Statistics section (severity × category matrix)
        lines.extend(self._build_statistics_section(result))

        # Cost section
        if result.total_cost > 0 or result.llm_calls > 0:
            lines.extend(
                [
                    "<details>",
                    "<summary>💰 Cost Summary</summary>",
                    "",
                    f"**Total:** ${result.total_cost:.4f} | "
                    f"**LLM Calls:** {result.llm_calls}",
                ]
            )
            if result.memory_pending:
                lines.append("- **Memory:** pending (retain, consolidation, mental models)")
            lines.append("")

            lines.extend(result.cost_breakdown_markdown_lines("####"))

            lines.extend(
                [
                    "</details>",
                    "",
                ]
            )

        # Issues without line numbers
        if body_issues:
            lines.extend(
                [
                    "<details>",
                    "<summary>⚠️ Issues Without Line References</summary>",
                    "",
                ]
            )

            for issue in body_issues:
                emoji = self.SEVERITY_EMOJI.get(issue.severity, "⚪")
                confidence_pct = int(issue.confidence * 100)
                lines.extend(
                    [
                        f"### {emoji} [{issue.severity.value.title()}] {issue.title}",
                        "",
                        f"**File:** `{issue.filename}`",
                        f"**Category:** {issue.category.value} | **Confidence:** {confidence_pct}%",
                        "",
                        issue.description,
                        "",
                    ]
                )

                if issue.suggestion:
                    lines.extend(
                        [
                            "**Suggestion:**",
                            issue.suggestion,
                            "",
                        ]
                    )

                if issue.cwe_id:
                    cwe_number = issue.cwe_id.split("-")[1] if "-" in issue.cwe_id else issue.cwe_id
                    lines.append(
                        f"**Reference:** [{issue.cwe_id}](https://cwe.mitre.org/data/definitions/{cwe_number}.html)"
                    )
                    lines.append("")

                lines.append("---")
                lines.append("")

            lines.extend(
                [
                    "</details>",
                    "",
                ]
            )

        # Recommendation
        if result.recommendation:
            lines.extend(
                [
                    "<details>",
                    "<summary>💡 Recommendation</summary>",
                    "",
                    result.recommendation,
                    "",
                    "</details>",
                    "",
                ]
            )

        # Memories section (collapsible, size-limited for GitHub's 65,536 char body limit)
        # Insert after header, before Summary
        if result.memories:
            # Calculate budget after building all other sections, minus reserve
            other_sections_len = len("\n".join(lines)) + 1  # +1 for trailing newline
            budget = self.MAX_BODY_CHARS - reserve - other_sections_len - 1
            memory_lines = result.memories_markdown_lines(
                summary="🧠 memories", max_chars=max(0, budget)
            )
            if memory_lines:
                lines[insert_at:insert_at] = memory_lines

        return "\n".join(lines)

    def _build_statistics_section(self, result: ReviewResult) -> list[str]:
        """Build the statistics section as a severity × category matrix.

        Args:
            result: The review result containing issues.

        Returns:
            Markdown lines for the statistics section.
        """
        severities = [
            IssueSeverity.CRITICAL,
            IssueSeverity.HIGH,
            IssueSeverity.MEDIUM,
            IssueSeverity.LOW,
            IssueSeverity.INFO,
        ]
        categories = [
            (IssueCategory.SECURITY, "Security"),
            (IssueCategory.BUG, "Bugs"),
            (IssueCategory.DOCUMENTATION, "Documentation"),
            (IssueCategory.SMELL, "Smells"),
        ]

        # Build counts with one pass over issues
        counts = Counter((issue.severity, issue.category) for issue in result.issues)

        # Calculate row totals (per severity)
        row_totals = {s: sum(counts[(s, c)] for c, _ in categories) for s in severities}

        # Calculate column totals (per category)
        col_totals = {c: sum(counts[(s, c)] for s in severities) for c, _ in categories}

        # Grand total
        grand_total = result.total_issues

        # Build table lines
        lines = [
            "<details>",
            "<summary>📊 Statistics</summary>",
            "",
            "| Severity | Security | Bugs | Documentation | Smells | Total |",
            "|----------|----------|------|---------------|--------|-------|",
        ]

        # Data rows
        for severity in severities:
            cells = [severity.value.title()]
            for cat, _ in categories:
                cells.append(str(counts[(severity, cat)]))
            cells.append(str(row_totals[severity]))
            lines.append("| " + " | ".join(cells) + " |")

        # Totals row (bold)
        total_row = ["**Total**"]
        for cat, _ in categories:
            total_row.append(f"**{col_totals[cat]}**")
        total_row.append(f"**{grand_total}**")
        lines.append("| " + " | ".join(total_row) + " |")

        lines.extend([
            "",
            "</details>",
            "",
        ])

        return lines

    def _build_inline_comments(self, issues: list[Issue]) -> list[dict]:
        """Build inline comment dictionaries for the Git API.

        Args:
            issues: Issues with line numbers.

        Returns:
            List of comment dicts for Git API.
        """
        comments = []

        for issue in issues:
            emoji = self.SEVERITY_EMOJI.get(issue.severity, "⚪")

            # Build comment body - keep essential info visible
            confidence_pct = int(issue.confidence * 100)
            body_lines = [
                f"{emoji} **[{issue.severity.value.title()}] {issue.title}**",
                "",
                f"**Category:** {issue.category.value} | **Confidence:** {confidence_pct}%",
                "",
                issue.description,
            ]

            # Code snippet - collapsible
            if issue.code_snippet:
                body_lines.extend(
                    [
                        "",
                        "<details>",
                        "<summary>📝 Code Snippet</summary>",
                        "",
                        "```",
                        issue.code_snippet,
                        "```",
                        "",
                        "</details>",
                    ]
                )

            # Suggestion - collapsible
            if issue.suggestion:
                body_lines.extend(
                    [
                        "",
                        "<details>",
                        "<summary>💡 Suggestion</summary>",
                        "",
                        issue.suggestion,
                        "",
                        "</details>",
                    ]
                )

            # CWE reference - always visible (one line)
            if issue.cwe_id:
                cwe_number = issue.cwe_id.split("-")[1] if "-" in issue.cwe_id else issue.cwe_id
                body_lines.extend(
                    [
                        "",
                        f"**Reference:** [{issue.cwe_id}](https://cwe.mitre.org/data/definitions/{cwe_number}.html)",
                    ]
                )

            comment = {
                "path": issue.filename,
                "body": "\n".join(body_lines),
                "line": issue.line_start,
            }

            # Add multi-line support if applicable
            if issue.line_end and issue.line_end != issue.line_start:
                comment["start_line"] = issue.line_start
                comment["line"] = issue.line_end

            comments.append(comment)

        return comments


# Backward compatibility alias
GitHubPRReporter = GitReporter
