"""Data models for code review results."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class IssueSeverity(str, Enum):
    """Severity level of an issue."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(str, Enum):
    """Category of an issue."""

    SECURITY = "security"
    BUG = "bug"
    DOCUMENTATION = "documentation"
    CONTEXT = "context"  # Issues found from codebase context analysis


class ScopeType(str, Enum):
    """Type of code scope in a repository."""

    LIBRARY = "library"  # Shared code that others import
    SERVICE = "service"  # Isolated microservice with explicit APIs
    APPLICATION = "application"  # Standalone app or frontend
    SCRIPT = "script"  # Build/deployment scripts, tooling


class PackageManifest(BaseModel):
    """Package management file information for a scope."""

    manifest_path: str = Field(description="Path to manifest file (e.g., package.json)")
    lock_file_path: str | None = Field(
        default=None, description="Path to lock file (e.g., package-lock.json)"
    )
    package_manager: str = Field(description="Package manager name (e.g., npm, go, pip)")
    dependencies_changed: bool = Field(
        default=False, description="Whether PR modified this manifest or lock file"
    )


from codespy.tools.git.models import ChangedFile


class ScopeResult(BaseModel):
    """A detected scope/subroot in the repository."""

    subroot: str = Field(description="Path relative to repo root (e.g., packages/auth)")
    scope_type: ScopeType = Field(description="Type of scope (library, service, etc.)")
    has_changes: bool = Field(
        default=False, description="Whether this scope has changed files from PR"
    )
    is_dependency: bool = Field(
        default=False, description="Whether this scope depends on a changed scope"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence score for scope identification"
    )
    language: str | None = Field(default=None, description="Primary language detected")
    package_manifest: PackageManifest | None = Field(
        default=None, description="Package manifest info if present"
    )
    changed_files: list[ChangedFile] = Field(
        default_factory=list, description="Changed files belonging to this scope"
    )
    reason: str = Field(description="Explanation for why this scope was identified")

    model_config = {"arbitrary_types_allowed": True}


class Issue(BaseModel):
    """Represents a single issue found during review."""

    category: IssueCategory = Field(description="Issue category")
    severity: IssueSeverity = Field(description="Issue severity")
    title: str = Field(description="Brief title of the issue")
    description: str = Field(description="Detailed description of the issue")
    filename: str = Field(description="File where the issue was found")
    line_start: int | None = Field(default=None, description="Starting line number")
    line_end: int | None = Field(default=None, description="Ending line number")
    code_snippet: str | None = Field(default=None, description="Relevant code snippet")
    suggestion: str | None = Field(default=None, description="Suggested fix or improvement")
    cwe_id: str | None = Field(
        default=None, description="CWE ID for security issues (e.g., CWE-79)"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence score (0-1)"
    )

    @property
    def location(self) -> str:
        """Get a human-readable location string."""
        if self.line_start:
            if self.line_end and self.line_end != self.line_start:
                return f"{self.filename}:{self.line_start}-{self.line_end}"
            return f"{self.filename}:{self.line_start}"
        return self.filename

class SignatureStatsResult(BaseModel):
    """Statistics for a single signature's execution during review."""

    name: str = Field(description="Signature name (e.g., bug_detection, code_security)")
    cost: float = Field(default=0.0, description="Cost in USD for this signature")
    tokens: int = Field(default=0, description="Tokens used by this signature")
    call_count: int = Field(default=0, description="Number of LLM calls made by this signature")
    duration_seconds: float = Field(default=0.0, description="Execution time in seconds")

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


class ReviewResult(BaseModel):
    """Complete review results for a merge request (GitHub PR or GitLab MR)."""

    mr_number: int = Field(description="MR number")
    mr_title: str = Field(description="MR title")
    mr_url: str = Field(description="MR URL")
    repo: str = Field(description="Repository name (owner/repo)")
    reviewed_at: datetime = Field(
        default_factory=datetime.utcnow, description="Review timestamp"
    )
    model_used: str = Field(description="LLM model used for review")
    issues: list[Issue] = Field(
        default_factory=list, description="All issues found during review"
    )
    overall_summary: str | None = Field(
        default=None, description="Overall summary of the PR"
    )
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
    def critical_issues(self) -> list[Issue]:
        """Get all critical issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def security_issues(self) -> list[Issue]:
        """Get all security issues."""
        return [i for i in self.issues if i.category == IssueCategory.SECURITY]

    @property
    def bug_issues(self) -> list[Issue]:
        """Get all bug issues."""
        return [i for i in self.issues if i.category == IssueCategory.BUG]

    @property
    def documentation_issues(self) -> list[Issue]:
        """Get all documentation issues."""
        return [i for i in self.issues if i.category == IssueCategory.DOCUMENTATION]

    @property
    def context_issues(self) -> list[Issue]:
        """Get all context issues."""
        return [i for i in self.issues if i.category == IssueCategory.CONTEXT]

    def issues_by_severity(self) -> dict[IssueSeverity, list[Issue]]:
        """Group issues by severity."""
        result: dict[IssueSeverity, list[Issue]] = {s: [] for s in IssueSeverity}
        for issue in self.issues:
            result[issue.severity].append(issue)
        return result

    def to_markdown(self) -> str:
        """Format review results as Markdown."""
        lines = [
            f"# Code Review: {self.mr_title}",
            "",
            f"**MR:** [{self.repo}#{self.mr_number}]({self.mr_url})",
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
        lines.extend([
            "## Statistics",
            "",
            f"- **Total Issues:** {self.total_issues}",
            f"- **Critical:** {len(self.critical_issues)}",
            f"- **Security:** {len(self.security_issues)}",
            f"- **Bugs:** {len(self.bug_issues)}",
            f"- **Documentation:** {len(self.documentation_issues)}",
            f"- **Context:** {len(self.context_issues)}",
            "",
        ])

        # Cost information
        if self.total_cost > 0 or self.llm_calls > 0:
            lines.extend([
                "## Cost",
                "",
                f"- **LLM Calls:** {self.llm_calls}",
                f"- **Total Tokens:** {self.total_tokens:,}",
                f"- **Total Cost:** ${self.total_cost:.4f}",
                "",
            ])

            # Per-signature breakdown
            if self.signature_stats:
                lines.extend([
                    "### Per-Signature Breakdown",
                    "",
                    "| Signature | Cost | Tokens | Calls | Duration |",
                    "|-----------|------|--------|-------|----------|",
                ])
                for stats in sorted(self.signature_stats, key=lambda x: x.cost, reverse=True):
                    duration_str = f"{stats.duration_seconds:.1f}s"
                    lines.append(
                        f"| {stats.name} | ${stats.cost:.4f} | {stats.tokens:,} | {stats.call_count} | {duration_str} |"
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

                    lines.extend([f"### {emoji} {severity.value.title()} ({len(severity_issues)})", ""])

                    for issue in severity_issues:
                        lines.extend([
                            f"#### {issue.title}",
                            "",
                            f"**Location:** `{issue.location}`",
                            f"**Category:** {issue.category.value}",
                            "",
                            issue.description,
                            "",
                        ])

                        if issue.code_snippet:
                            lines.extend([
                                "**Code:**",
                                "```",
                                issue.code_snippet,
                                "```",
                                "",
                            ])

                        if issue.suggestion:
                            lines.extend([
                                "**Suggestion:**",
                                issue.suggestion,
                                "",
                            ])

                        if issue.cwe_id:
                            lines.append(f"**Reference:** [{issue.cwe_id}](https://cwe.mitre.org/data/definitions/{issue.cwe_id.split('-')[1]}.html)")
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