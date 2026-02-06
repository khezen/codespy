"""Unified Git reporter for posting review comments to GitHub/GitLab."""

import logging
from typing import TYPE_CHECKING

from codespy.agents.reviewer.models import Issue, IssueSeverity, ReviewResult
from codespy.agents.reviewer.reporters.base import BaseReporter
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
        """Post review result to the merge request.

        Args:
            result: The review result to post.
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
        self.client.submit_review(
            url=self.url,
            body=body,
            comments=comments,
        )

        logger.info(
            f"Posted {self.client.platform_name} review with {len(comments)} inline comments "
            f"and {len(body_issues)} issues in body"
        )

    def _build_review_body(
        self,
        result: ReviewResult,
        body_issues: list[Issue],
    ) -> str:
        """Build the review body with collapsible sections.

        Args:
            result: The review result.
            body_issues: Issues without line numbers to include in body.

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

        # Summary section
        if result.overall_summary:
            lines.extend([
                "<details>",
                "<summary>📋 Summary</summary>",
                "",
                result.overall_summary,
                "",
                "</details>",
                "",
            ])

        # Quality Assessment section
        if result.quality_assessment:
            lines.extend([
                "<details>",
                "<summary>🎯 Quality Assessment</summary>",
                "",
                result.quality_assessment,
                "",
                "</details>",
                "",
            ])

        # Statistics section
        lines.extend([
            "<details>",
            "<summary>📊 Statistics</summary>",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total Issues | {result.total_issues} |",
            f"| Critical | {len(result.critical_issues)} |",
            f"| High | {len([i for i in result.issues if i.severity == IssueSeverity.HIGH])} |",
            f"| Medium | {len([i for i in result.issues if i.severity == IssueSeverity.MEDIUM])} |",
            f"| Low | {len([i for i in result.issues if i.severity == IssueSeverity.LOW])} |",
            f"| Security | {len(result.security_issues)} |",
            f"| Bugs | {len(result.bug_issues)} |",
            f"| Documentation | {len(result.documentation_issues)} |",
            "",
            "</details>",
            "",
        ])

        # Cost section
        if result.total_cost > 0 or result.llm_calls > 0:
            lines.extend([
                "<details>",
                "<summary>💰 Cost Summary</summary>",
                "",
                f"**Total:** ${result.total_cost:.4f} | "
                f"**Tokens:** {result.total_tokens:,} | "
                f"**LLM Calls:** {result.llm_calls}",
                "",
            ])

            if result.signature_stats:
                lines.extend([
                    "| Signature | Cost | Tokens | Calls | Duration |",
                    "|-----------|------|--------|-------|----------|",
                ])
                for stats in sorted(result.signature_stats, key=lambda x: x.cost, reverse=True):
                    duration_str = f"{stats.duration_seconds:.1f}s"
                    lines.append(
                        f"| {stats.name} | ${stats.cost:.4f} | {stats.tokens:,} | "
                        f"{stats.call_count} | {duration_str} |"
                    )
                lines.append("")

            lines.extend([
                "</details>",
                "",
            ])

        # Issues without line numbers
        if body_issues:
            lines.extend([
                "<details>",
                "<summary>⚠️ Issues Without Line References</summary>",
                "",
            ])

            for issue in body_issues:
                emoji = self.SEVERITY_EMOJI.get(issue.severity, "⚪")
                lines.extend([
                    f"### {emoji} [{issue.severity.value.title()}] {issue.title}",
                    "",
                    f"**File:** `{issue.filename}`",
                    f"**Category:** {issue.category.value}",
                    "",
                    issue.description,
                    "",
                ])

                if issue.suggestion:
                    lines.extend([
                        "**Suggestion:**",
                        issue.suggestion,
                        "",
                    ])

                if issue.cwe_id:
                    cwe_number = issue.cwe_id.split("-")[1] if "-" in issue.cwe_id else issue.cwe_id
                    lines.append(
                        f"**Reference:** [{issue.cwe_id}](https://cwe.mitre.org/data/definitions/{cwe_number}.html)"
                    )
                    lines.append("")

                lines.append("---")
                lines.append("")

            lines.extend([
                "</details>",
                "",
            ])

        # Recommendation
        if result.recommendation:
            lines.extend([
                "<details>",
                "<summary>💡 Recommendation</summary>",
                "",
                result.recommendation,
                "",
                "</details>",
                "",
            ])

        return "\n".join(lines)

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
            body_lines = [
                f"{emoji} **[{issue.severity.value.title()}] {issue.title}**",
                "",
                f"**Category:** {issue.category.value}",
                "",
                issue.description,
            ]

            # Code snippet - collapsible
            if issue.code_snippet:
                body_lines.extend([
                    "",
                    "<details>",
                    "<summary>📝 Code Snippet</summary>",
                    "",
                    "```",
                    issue.code_snippet,
                    "```",
                    "",
                    "</details>",
                ])

            # Suggestion - collapsible
            if issue.suggestion:
                body_lines.extend([
                    "",
                    "<details>",
                    "<summary>💡 Suggestion</summary>",
                    "",
                    issue.suggestion,
                    "",
                    "</details>",
                ])

            # CWE reference - always visible (one line)
            if issue.cwe_id:
                cwe_number = issue.cwe_id.split("-")[1] if "-" in issue.cwe_id else issue.cwe_id
                body_lines.extend([
                    "",
                    f"**Reference:** [{issue.cwe_id}](https://cwe.mitre.org/data/definitions/{cwe_number}.html)",
                ])

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