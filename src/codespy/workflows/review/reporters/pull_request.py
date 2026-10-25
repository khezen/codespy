"""Reporter that delivers a full-path review as a report PR/MR opened by codespy."""

import logging
import time
from typing import TYPE_CHECKING

from codespy import __version__
from codespy.agents.review.models import Issue, IssueSeverity
from codespy.tools.git.base import SubmittedPullRequest
from codespy.tools.git.client import get_repo_client
from codespy.tools.git.conversation import fingerprint, issue_marker, review_marker
from codespy.tools.git.full_path import blob_url, report_branch, report_file
from codespy.workflows.review.models import MAX_PLAN_CHARS, ReviewResult

if TYPE_CHECKING:
    from codespy.config import Settings

logger = logging.getLogger(__name__)

SEVERITY_ORDER = [
    IssueSeverity.CRITICAL,
    IssueSeverity.HIGH,
    IssueSeverity.MEDIUM,
    IssueSeverity.LOW,
    IssueSeverity.INFO,
]
SEVERITY_EMOJI = {
    IssueSeverity.CRITICAL: "🔴",
    IssueSeverity.HIGH: "🟠",
    IssueSeverity.MEDIUM: "🟡",
    IssueSeverity.LOW: "🔵",
    IssueSeverity.INFO: "⚪",
}


class PullRequestReporter:
    """Opens (or updates) ``codespy/review/<path-slug>`` with the report.

    The committed ``.codespy/reviews/<path-slug>.md`` holds the full report; the
    PR description holds the same report, with the issue list cut to fit the
    platform body limit and a link to the file when it does not fit.
    """

    # Below GitHub's 65,536 char body limit (same budget as GitReporter)
    MAX_BODY_CHARS = 60000
    UPDATE_RETRY_DELAYS = [2.0, 4.0, 8.0]  # seconds

    def __init__(self, repo_url: str, settings: "Settings | None" = None) -> None:
        self.repo_url = repo_url
        self.client = get_repo_client(repo_url, settings)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, result: ReviewResult) -> SubmittedPullRequest:
        """Open or update the report PR. Raises on platform errors."""
        fp = result.full_path
        if fp is None:
            raise ValueError("PullRequestReporter needs a full-path ReviewResult")

        handle = self.client.upsert_report_pull_request(
            owner=fp.owner,
            repo=fp.repo,
            base_branch=fp.ref,
            base_sha=fp.sha,
            head_branch=report_branch(fp.path),
            file_path=report_file(fp.path),
            file_content=result.to_markdown(),
            commit_message=f"codespy: full review of {fp.path} @ {fp.sha[:8]}",
            title=self.title(result),
            body=self.build_body(result),
            update_comment=(
                f"Updated by codespy run `{result.run_id[:8]}` at `{fp.sha[:8]}`: "
                f"{result.total_issues} issue(s) in {fp.files_reviewed} file(s)."
            ),
        )
        action = "Opened" if handle.created else "Updated"
        logger.info(f"{action} report {self.client.platform_name} PR: {handle.url}")
        return handle

    def update(self, handle: SubmittedPullRequest, result: ReviewResult) -> None:
        """Rewrite the description with the post-memory costs (3 retries, then keep)."""
        body = self.build_body(result)
        last_error: Exception | None = None
        for attempt, delay in enumerate(self.UPDATE_RETRY_DELAYS):
            try:
                self.client.update_pull_request_body(handle.url, body)
                logger.info("Updated report PR description with full costs")
                return
            except Exception as e:
                last_error = e
                logger.warning(f"Update attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
        logger.warning(
            f"Failed to update report PR after {len(self.UPDATE_RETRY_DELAYS)} attempts. "
            "Keeping the original description."
        )
        if last_error:
            logger.debug("Final update error", exc_info=last_error)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def title(result: ReviewResult) -> str:
        fp = result.full_path
        assert fp is not None
        shown = "root" if fp.path == "." else fp.path
        return f"codespy: full review of {shown} @ {fp.ref}"

    def report_file_url(self, result: ReviewResult) -> str:
        fp = result.full_path
        assert fp is not None
        return blob_url(
            fp.host, fp.owner, fp.repo, report_branch(fp.path), report_file(fp.path), fp.platform
        )

    def _issue_block(self, issue: Issue, result: ReviewResult) -> list[str]:
        emoji = SEVERITY_EMOJI.get(issue.severity, "⚪")
        lines = [
            f"#### {emoji} [{issue.severity.value.title()}] {issue.title}",
            issue_marker(
                result.run_id or "",
                fingerprint(issue.filename, issue.category.value, issue.title),
                issue.category.value,
                issue.severity.value,
            ),
            "",
            f"**Location:** {result.issue_location_md(issue)}",
            f"**Category:** {issue.category.value} | "
            f"**Confidence:** {int(issue.confidence * 100)}%",
            "",
            issue.description,
            "",
        ]
        if issue.suggestion:
            lines.extend(["**Suggestion:**", issue.suggestion, ""])
        if issue.cwe_id:
            num = issue.cwe_id.split("-")[1] if "-" in issue.cwe_id else issue.cwe_id
            lines.extend(
                [
                    f"**Reference:** [{issue.cwe_id}](https://cwe.mitre.org/data/definitions/{num}.html)",
                    "",
                ]
            )
        return lines

    @staticmethod
    def _size(lines: list[str]) -> int:
        return sum(len(line) + 1 for line in lines)

    def build_body(self, result: ReviewResult) -> str:
        """PR description: always under ``MAX_BODY_CHARS``.

        Issues are dropped (lowest severity last) before any other section; the
        memories section only gets the budget left over, as in GitReporter.
        """
        fp = result.full_path
        assert fp is not None

        head = [
            "# 🔍 Code[Spy](https://github.com/khezen/codespy) Full Review "
            f"<sub>v{__version__}</sub>",
            review_marker(result.run_id or ""),
            "",
            f"**Repository:** [{fp.owner}/{fp.repo}@{fp.sha[:8]}]({fp.tree_url}) | "
            f"**Issues Found:** {result.total_issues} | "
            f"**Critical:** {len(result.critical_issues)} | **Model:** {result.model_used}",
            "",
            *result.full_path_scope_lines(),
        ]
        if result.overall_summary:
            head.extend(["## Summary", "", result.overall_summary, ""])
        if result.quality_assessment:
            head.extend(["## Quality Assessment", "", result.quality_assessment, ""])

        tail: list[str] = []
        if result.recommendation:
            tail.extend(["## Recommendation", "", result.recommendation, ""])

        # Implementation Plan (capped; the report file holds the full plan)
        if result.implementation_plan:
            tail.extend(
                result.implementation_plan_markdown_lines(
                    max_chars=MAX_PLAN_CHARS, overflow_link=self.report_file_url(result)
                )
            )

        if result.total_cost > 0 or result.llm_calls > 0:
            tail.extend(
                [
                    "<details>",
                    "<summary>💰 Cost Summary</summary>",
                    "",
                    f"**Total:** ${result.total_cost:.4f} | **LLM Calls:** {result.llm_calls}",
                ]
            )
            if result.memory_pending:
                tail.append("- **Memory:** pending (retain, consolidation, mental models)")
            tail.extend(["", *result.cost_breakdown_markdown_lines("####"), "</details>", ""])
        tail.extend(
            ["---", f"📄 Full report: [{report_file(fp.path)}]({self.report_file_url(result)})", ""]
        )

        # Issues, highest severity first, until the budget runs out
        truncation_note_budget = 300
        budget = self.MAX_BODY_CHARS - self._size(head) - self._size(tail) - truncation_note_budget
        issue_lines: list[str] = []
        shown = 0
        if result.issues:
            issue_lines.extend(["## Issues", ""])
            stop = False
            for severity in SEVERITY_ORDER:
                sev_issues = [i for i in result.issues if i.severity == severity]
                if not sev_issues or stop:
                    continue
                section = [
                    f"### {SEVERITY_EMOJI[severity]} {severity.value.title()} ({len(sev_issues)})",
                    "",
                ]
                for issue in sev_issues:
                    block = self._issue_block(issue, result)
                    if self._size(issue_lines) + self._size(section) + self._size(block) > budget:
                        stop = True
                        break
                    section.extend(block)
                    shown += 1
                if len(section) > 2:
                    issue_lines.extend(section)
            if shown < result.total_issues:
                issue_lines.extend(
                    [
                        f"> {result.total_issues - shown} more issue(s) did not fit in this "
                        f"description. See the [full report]({self.report_file_url(result)}).",
                        "",
                    ]
                )

        lines = head + issue_lines + tail
        if result.memories:
            remaining = self.MAX_BODY_CHARS - self._size(lines) - 1
            memory_lines = result.memories_markdown_lines(
                summary="🧠 memories", max_chars=max(0, remaining)
            )
            insert_at = len(head)
            lines[insert_at:insert_at] = memory_lines
        return "\n".join(lines)
