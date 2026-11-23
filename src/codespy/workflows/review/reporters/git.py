"""Git reporter for posting review comments to GitHub/GitLab."""

import logging
import time
from typing import TYPE_CHECKING

from collections import Counter

from codespy import __version__
from codespy.agents.review.models import Issue, IssueCategory, IssueSeverity
from codespy.workflows.review.models import MAX_PLAN_CHARS, ReviewResult
from codespy.workflows.review.reporters.base import BaseReporter
from codespy.tools.git.base import SubmittedReview
from codespy.tools.git.client import get_client
from codespy.tools.git.conversation import (
    fingerprint,
    issue_marker,
    issues_marker,
    plan_marker,
    review_marker,
)

if TYPE_CHECKING:
    from codespy.config import Settings

logger = logging.getLogger(__name__)

# Appended to content cut to fit a comment
TRUNCATED_MARKER = "…(truncated)"


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
        """Post review result to the merge request.

        Args:
            result: The review result to post.
        """
        self.publish(result)

    def publish(self, result: ReviewResult) -> SubmittedReview | None:
        """Publish the review to the merge request.

        Posts the initial review after the audit phase (before memory phase).
        The review will show "Memory: pending" in costs if memory_pending is True.

        When the review body plus issues and plan would exceed the budget,
        the issues without line references and the plan are posted as separate
        comments with hidden markers.

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

        # Decide if we need to split
        split = self._needs_split(result, body_issues)

        # Build review body with collapsible sections
        body = self._build_review_body(result, body_issues, reserve=0, split=split)

        # Build inline comments
        comments = self._build_inline_comments(inline_issues, run_id=result.run_id)

        # Submit the review first
        handle = self.client.submit_review(
            url=self.url,
            body=body,
            comments=comments,
            split=split,
        )

        if not handle:
            return None

        # Post overflow comments if in split mode
        if split:
            overflow = self._overflow_comments(result, body_issues)
            for idx, comment_body in enumerate(overflow, start=1):
                try:
                    self.client.post_comment(self.url, comment_body)
                    logger.debug(f"Posted overflow comment {idx}/{len(overflow)}")
                except Exception as e:
                    logger.warning(f"Failed to post overflow comment {idx}: {e}")

        issue_count = len(body_issues) if not split else 0
        plan_in_body = "with plan" if (result.implementation_plan and not split) else "split mode"
        logger.info(
            f"Posted {self.client.platform_name} review with {len(comments)} inline comments, "
            f"{issue_count} issues in body ({plan_in_body})"
        )
        return handle

    def update(self, handle: SubmittedReview, result: ReviewResult) -> None:
        """Update an existing review with new costs after memory phase.

        Rebuilds the body with full costs and edits the review.
        Retries up to 3 times with exponential backoff on failure.
        Does not touch overflow comments (issues and plan don't change).

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

        # Build review body in the same layout as publish
        body = self._build_review_body(
            result, body_issues, reserve=len(handle.body_suffix), split=handle.split
        )

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

    def _body_issue_lines(self, issue: Issue, run_id: str | None) -> list[str]:
        """Build markdown lines for a single body issue (without line number).

        Args:
            issue: The issue to render.
            run_id: Pipeline run ID for the hidden marker.

        Returns:
            Markdown lines for the issue block.
        """
        emoji = self.SEVERITY_EMOJI.get(issue.severity, "⚪")
        confidence_pct = int(issue.confidence * 100)
        lines = [
            f"### {emoji} [{issue.severity.value.title()}] {issue.title}",
            self._issue_marker(issue, run_id),
            "",
            f"**File:** `{issue.filename}`",
            f"**Category:** {issue.category.value} | **Confidence:** {confidence_pct}%",
            "",
            issue.description,
            "",
        ]

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

        lines.extend(["---", ""])
        return lines

    def _needs_split(self, result: ReviewResult, body_issues: list[Issue]) -> bool:
        """Decide whether the body needs to split issues and plan into separate comments.

        Split when the full body (with uncapped plan and all issues) would exceed
        the body budget, or when the plan alone exceeds MAX_PLAN_CHARS.

        Args:
            result: The review result.
            body_issues: Issues without line numbers.

        Returns:
            True if split mode is needed.
        """
        # Build the full body as if there were no split
        full_body = self._build_review_body(result, body_issues, reserve=0, split=False)

        # Check if the full body exceeds the budget
        if len(full_body) > self.MAX_BODY_CHARS:
            return True

        # Check if the plan alone exceeds MAX_PLAN_CHARS
        if result.implementation_plan:
            plan_lines = result.implementation_plan_markdown_lines(max_chars=None)
            plan_size = sum(len(line) + 1 for line in plan_lines)
            if plan_size > MAX_PLAN_CHARS:
                return True

        return False

    def _overflow_comments(
        self, result: ReviewResult, body_issues: list[Issue]
    ) -> list[str]:
        """Build overflow comments for split mode.

        Returns a list of comment bodies: first the issues comments (possibly
        split into parts), then the plan comment. Each starts with a hidden marker.

        Args:
            result: The review result.
            body_issues: Issues without line numbers.

        Returns:
            List of comment bodies, each ready to post.
        """
        comments: list[str] = []
        run_id = result.run_id or ""

        # Build issues comments
        if body_issues:
            issue_lines: list[str] = [
                "<details>",
                "<summary>⚠️ Issues Without Line References</summary>",
                "",
            ]
            for issue in body_issues:
                issue_lines.extend(self._body_issue_lines(issue, run_id))
            issue_lines.extend(["</details>", ""])

            # Split into parts if needed
            issue_text = "\n".join(issue_lines)
            if len(issue_text) > self.MAX_BODY_CHARS:
                # Split at issue boundaries
                parts: list[list[str]] = []
                current_part: list[str] = []
                header = ["<details>", "<summary>⚠️ Issues Without Line References</summary>", ""]
                footer = ["</details>", ""]
                header_size = sum(len(line) + 1 for line in header)
                footer_size = sum(len(line) + 1 for line in footer)
                available = self.MAX_BODY_CHARS - header_size - footer_size

                for issue in body_issues:
                    block = self._body_issue_lines(issue, run_id)
                    block_size = sum(len(line) + 1 for line in block)
                    if block_size > available:
                        # Single issue too long - truncate it
                        block = self._truncate_issue_block(block, available)
                    if current_part and sum(len(line) + 1 for line in current_part) + block_size > available:
                        # Start new part
                        parts.append(current_part)
                        current_part = block
                    else:
                        current_part.extend(block)
                if current_part:
                    parts.append(current_part)

                total = len(parts)
                for idx, part_lines in enumerate(parts, start=1):
                    body = "\n".join([issues_marker(run_id, idx, total)] + header + part_lines + footer)
                    comments.append(body)
            else:
                # Single issues comment
                body = "\n".join([issues_marker(run_id, 1, 1)] + issue_lines)
                comments.append(body)

        # Build plan comment
        if result.implementation_plan:
            plan_lines = result.implementation_plan_markdown_lines(
                max_chars=self.MAX_BODY_CHARS - len(plan_marker(run_id)) - 1
            )
            if plan_lines:
                body = "\n".join([plan_marker(run_id)] + plan_lines)
                comments.append(body)

        return comments

    def _truncate_issue_block(self, lines: list[str], max_chars: int) -> list[str]:
        """Truncate an issue block to fit within max_chars.

        Args:
            lines: The issue block lines.
            max_chars: Maximum character budget.

        Returns:
            Truncated lines with a truncation marker.
        """
        size = 0
        result: list[str] = []
        for line in lines:
            line_size = len(line) + 1
            if size + line_size > max_chars - len(TRUNCATED_MARKER) - 2:
                result.append(f"> {TRUNCATED_MARKER}")
                break
            result.append(line)
            size += line_size
        return result

    def _build_review_body(
        self,
        result: ReviewResult,
        body_issues: list[Issue],
        *,
        reserve: int,
        split: bool,
    ) -> str:
        """Build the review body with collapsible sections.

        Args:
            result: The review result.
            body_issues: Issues without line numbers to include in body.
            reserve: Number of characters to reserve for body suffix.
            split: True when issues and plan are posted as separate comments.

        Returns:
            Formatted markdown string for review body.
        """
        lines = []

        # Header with stats - link to CodeSpy repo
        lines.append(
            f"# 🔍 Code[Spy](https://github.com/khezen/codespy) Review <sub>v{__version__}</sub>"
        )
        # Hidden marker: lets later runs recognize (and exclude) codespy's own review
        lines.append(review_marker(result.run_id or ""))
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

        # Issues without line numbers (only when not in split mode)
        if body_issues and not split:
            lines.extend(
                [
                    "<details>",
                    "<summary>⚠️ Issues Without Line References</summary>",
                    "",
                ]
            )

            for issue in body_issues:
                lines.extend(self._body_issue_lines(issue, result.run_id))

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

        # Implementation Plan (only when not in split mode)
        if result.implementation_plan and not split:
            plan_lines = result.implementation_plan_markdown_lines(max_chars=None)
            lines.extend(plan_lines)

        # Split mode pointer
        if split:
            has_issues = bool(body_issues)
            has_plan = bool(result.implementation_plan)
            if has_issues and has_plan:
                pointer = "> ⚠️ Issues without line references and the implementation plan are posted as separate comments."
            elif has_issues:
                pointer = "> ⚠️ Issues without line references are posted as separate comments."
            elif has_plan:
                pointer = "> ⚠️ The implementation plan is posted as a separate comment."
            else:
                pointer = ""
            if pointer:
                lines.append(pointer)
                lines.append("")

        # Implementation agent's fix PR (small: kept in both layouts)
        lines.extend(result.implementation_markdown_lines())

        # Memories section (collapsible, size-limited for GitHub's 65,536 char body limit)
        # Insert after header, before Summary
        if result.memories:
            other_sections_len = len("\n".join(lines)) + 1
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
            (IssueCategory.SEO, "SEO"),
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
            "| Severity | Security | Bugs | Documentation | Smells | SEO | Total |",
            "|----------|----------|------|---------------|--------|-----|-------|",
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

    @staticmethod
    def _issue_marker(issue: Issue, run_id: str | None) -> str:
        """Hidden per-issue marker (fingerprint, category, severity)."""
        return issue_marker(
            run_id or "",
            fingerprint(issue.filename, issue.category.value, issue.title),
            issue.category.value,
            issue.severity.value,
        )

    def _build_inline_comments(self, issues: list[Issue], run_id: str | None = None) -> list[dict]:
        """Build inline comment dictionaries for the Git API.

        Args:
            issues: Issues with line numbers.
            run_id: Pipeline run ID embedded in the hidden issue marker.

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

            # Hidden marker last: the first line keeps the legacy header shape
            body_lines.extend(["", self._issue_marker(issue, run_id)])

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
