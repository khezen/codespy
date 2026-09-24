"""Workflow-owned types."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from codespy.agents.review.models import Issue, IssueCategory, IssueSeverity


class SignatureStatsResult(BaseModel):
    """Statistics for a single signature's execution during review."""

    name: str = Field(
        description="Signature name (e.g., code_review, doc, scope, supply_chain, summary, audit)"
    )
    cost: float = Field(default=0.0, description="Cost in USD for this signature")
    tokens: int = Field(default=0, description="Tokens used by this signature")
    call_count: int = Field(default=0, description="Number of LLM calls made by this signature")
    duration_seconds: float = Field(default=0.0, description="Execution time in seconds")
    input_tokens: int = Field(default=0, description="Input/prompt tokens used")
    output_tokens: int = Field(default=0, description="Output/completion tokens used")
    input_cost: float = Field(default=0.0, description="Cost for input tokens")
    output_cost: float = Field(default=0.0, description="Cost for output tokens")

    @property
    def cost_per_call(self) -> float:
        """Get average cost per LLM call."""
        if self.call_count == 0:
            return 0.0
        return self.cost / self.call_count

    @property
    def tokens_per_call(self) -> float:
        """Get average tokens per LLM call."""
        if self.call_count == 0:
            return 0.0
        return self.tokens / self.call_count


class RemoteReviewConfig(BaseModel):
    """Configuration for reviewing a remote PR/MR from GitHub or GitLab."""

    url: str = Field(description="URL of the GitHub PR or GitLab MR to review")


class LocalReviewConfig(BaseModel):
    """Configuration for reviewing local git changes without a remote platform."""

    repo_path: Path = Field(description="Path to the git repository")
    base_ref: str = Field(
        default="main",
        description="Base git ref to compare against (e.g., 'main', 'develop', 'HEAD~5')",
    )
    uncommitted: bool = Field(
        default=False, description="If True, review uncommitted changes (working tree vs HEAD)"
    )


# Union type for review configuration
ReviewConfig = RemoteReviewConfig | LocalReviewConfig


class ReviewResult(BaseModel):
    """Complete review results for a pull request (GitHub PR or GitLab MR)."""

    pr_number: int = Field(description="PR number")
    pr_title: str = Field(description="PR title")
    pr_url: str = Field(description="PR URL")
    repo: str = Field(description="Repository name (owner/repo)")
    run_id: str = Field(
        default="",
        description="Identifier of the pipeline run that produced this result, "
        "shared with all Episode records persisted during this run",
    )
    reviewed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Review timestamp",
    )
    model_used: str = Field(description="LLM model used for review")
    issues: list["Issue"] = Field(default_factory=list, description="All issues found during review")
    overall_summary: str | None = Field(default=None, description="Overall summary of the PR")
    quality_assessment: str | None = Field(
        default=None, description="Overall assessment of code quality"
    )
    recommendation: str | None = Field(
        default=None, description="Overall recommendation (approve, request changes, etc.)"
    )
    total_cost: float = Field(default=0.0, description="Total cost in USD")
    total_tokens: int = Field(default=0, description="Total tokens used")
    llm_calls: int = Field(default=0, description="Number of LLM calls made")
    signature_stats: list[SignatureStatsResult] = Field(
        default_factory=list, description="Per-signature statistics (cost, tokens, time)"
    )

    @property
    def total_issues(self) -> int:
        """Get total number of issues."""
        return len(self.issues)

    @property
    def critical_issues(self) -> list["Issue"]:
        """Get all critical issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def security_issues(self) -> list["Issue"]:
        """Get all security issues."""
        return [i for i in self.issues if i.category == IssueCategory.SECURITY]

    @property
    def bug_issues(self) -> list["Issue"]:
        """Get all bug issues."""
        return [i for i in self.issues if i.category == IssueCategory.BUG]

    @property
    def documentation_issues(self) -> list["Issue"]:
        """Get all documentation issues."""
        return [i for i in self.issues if i.category == IssueCategory.DOCUMENTATION]

    @property
    def smell_issues(self) -> list["Issue"]:
        """Get all code smell issues."""
        return [i for i in self.issues if i.category == IssueCategory.SMELL]

    def issues_by_severity(self) -> dict["IssueSeverity", list["Issue"]]:
        """Group issues by severity."""
        result: dict[IssueSeverity, list[Issue]] = {s: [] for s in IssueSeverity}
        for issue in self.issues:
            result[issue.severity].append(issue)
        return result

    def to_markdown(self) -> str:
        """Format review results as Markdown."""
        lines = [
            f"# Code Review: {self.pr_title}",
            "",
            f"**PR:** [{self.repo}#{self.pr_number}]({self.pr_url})",
            f"**Reviewed at:** {self.reviewed_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Model:** {self.model_used}",
            "",
        ]

        # Overall summary
        if self.overall_summary:
            lines.extend(["## Summary", "", self.overall_summary, ""])

        # Quality assessment
        if self.quality_assessment:
            lines.extend(["## Quality Assessment", "", self.quality_assessment, ""])

        # Statistics
        lines.extend(
            [
                "## Statistics",
                "",
                f"- **Total Issues:** {self.total_issues}",
                f"- **Critical:** {len(self.critical_issues)}",
                f"- **Security:** {len(self.security_issues)}",
                f"- **Bugs:** {len(self.bug_issues)}",
                f"- **Documentation:** {len(self.documentation_issues)}",
                f"- **Smells:** {len(self.smell_issues)}",
                "",
            ]
        )

        # Cost information
        if self.total_cost > 0 or self.llm_calls > 0:
            lines.extend(
                [
                    "## Cost",
                    "",
                    f"- **LLM Calls:** {self.llm_calls}",
                    f"- **Total Cost:** ${self.total_cost:.4f}",
                    "",
                ]
            )

            # Per-signature breakdown
            if self.signature_stats:
                lines.extend(
                    [
                        "### Per-Signature Breakdown",
                        "",
                        "| Signature | In Tokens | Out Tokens | In Cost | Out Cost | Calls | Duration |",
                        "|-----------|-----------|------------|---------|----------|-------|----------|",
                    ]
                )
                for stats in sorted(self.signature_stats, key=lambda x: x.cost, reverse=True):
                    duration_str = f"{stats.duration_seconds:.1f}s"
                    lines.append(
                        f"| {stats.name} | {stats.input_tokens:,} | {stats.output_tokens:,} | "
                        f"${stats.input_cost:.4f} | ${stats.output_cost:.4f} | {stats.call_count} | {duration_str} |"
                    )
                lines.append("")

        # Issues by severity
        if self.issues:
            lines.extend(["## Issues", ""])

            for severity in [
                IssueSeverity.CRITICAL,
                IssueSeverity.HIGH,
                IssueSeverity.MEDIUM,
                IssueSeverity.LOW,
                IssueSeverity.INFO,
            ]:
                severity_issues = [i for i in self.issues if i.severity == severity]
                if severity_issues:
                    emoji = {
                        IssueSeverity.CRITICAL: "🔴",
                        IssueSeverity.HIGH: "🟠",
                        IssueSeverity.MEDIUM: "🟡",
                        IssueSeverity.LOW: "🔵",
                        IssueSeverity.INFO: "⚪",
                    }[severity]

                    lines.extend(
                        [f"### {emoji} {severity.value.title()} ({len(severity_issues)})", ""]
                    )

                    for issue in severity_issues:
                        lines.extend(
                            [
                                f"#### {issue.title}",
                                "",
                                f"**Location:** `{issue.location}`",
                                f"**Category:** {issue.category.value}",
                                "",
                                issue.description,
                                "",
                            ]
                        )

                        if issue.code_snippet:
                            lines.extend(
                                [
                                    "**Code:**",
                                    "```",
                                    issue.code_snippet,
                                    "```",
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
                            lines.append(
                                f"**Reference:** [{issue.cwe_id}](https://cwe.mitre.org/data/definitions/{issue.cwe_id.split('-')[1]}.html)"
                            )
                            lines.append("")

                        lines.append("---")
                        lines.append("")

        # Recommendation
        if self.recommendation:
            lines.extend(["## Recommendation", "", self.recommendation, ""])

        return "\n".join(lines)

    def to_json_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return self.model_dump(mode="json")
