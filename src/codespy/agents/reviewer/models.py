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


from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codespy.tools.github.models import ChangedFile


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
    changed_files: list[Any] = Field(
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
    file: str = Field(description="File where the issue was found")
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
                return f"{self.file}:{self.line_start}-{self.line_end}"
            return f"{self.file}:{self.line_start}"
        return self.file


class FileReview(BaseModel):
    """Review results for a single file."""

    filename: str = Field(description="Path to the reviewed file")
    issues: list[Issue] = Field(default_factory=list, description="Issues found in this file")
    summary: str | None = Field(default=None, description="Brief summary of file changes")
    reviewed: bool = Field(default=True, description="Whether the file was actually reviewed")
    skip_reason: str | None = Field(
        default=None, description="Reason if file was skipped"
    )

    @property
    def issue_count(self) -> int:
        """Get total number of issues."""
        return len(self.issues)

    @property
    def critical_count(self) -> int:
        """Get number of critical issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Get number of high severity issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.HIGH)


class ReviewResult(BaseModel):
    """Complete review results for a pull request."""

    pr_number: int = Field(description="PR number")
    pr_title: str = Field(description="PR title")
    pr_url: str = Field(description="PR URL")
    repo: str = Field(description="Repository name (owner/repo)")
    reviewed_at: datetime = Field(
        default_factory=datetime.utcnow, description="Review timestamp"
    )
    model_used: str = Field(description="LLM model used for review")
    file_reviews: list[FileReview] = Field(
        default_factory=list, description="Per-file review results"
    )
    overall_summary: str | None = Field(
        default=None, description="Overall summary of the PR"
    )
    recommendation: str | None = Field(
        default=None, description="Overall recommendation (approve, request changes, etc.)"
    )
    total_cost: float = Field(default=0.0, description="Total cost in USD")
    total_tokens: int = Field(default=0, description="Total tokens used")
    llm_calls: int = Field(default=0, description="Number of LLM calls made")

    @property
    def all_issues(self) -> list[Issue]:
        """Get all issues across all files."""
        issues = []
        for file_review in self.file_reviews:
            issues.extend(file_review.issues)
        return issues

    @property
    def total_issues(self) -> int:
        """Get total number of issues."""
        return len(self.all_issues)

    @property
    def critical_issues(self) -> list[Issue]:
        """Get all critical issues."""
        return [i for i in self.all_issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def security_issues(self) -> list[Issue]:
        """Get all security issues."""
        return [i for i in self.all_issues if i.category == IssueCategory.SECURITY]

    @property
    def bug_issues(self) -> list[Issue]:
        """Get all bug issues."""
        return [i for i in self.all_issues if i.category == IssueCategory.BUG]

    @property
    def documentation_issues(self) -> list[Issue]:
        """Get all documentation issues."""
        return [i for i in self.all_issues if i.category == IssueCategory.DOCUMENTATION]

    @property
    def context_issues(self) -> list[Issue]:
        """Get all context issues."""
        return [i for i in self.all_issues if i.category == IssueCategory.CONTEXT]

    def issues_by_severity(self) -> dict[IssueSeverity, list[Issue]]:
        """Group issues by severity."""
        result: dict[IssueSeverity, list[Issue]] = {s: [] for s in IssueSeverity}
        for issue in self.all_issues:
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

        # Issues by severity
        if self.all_issues:
            lines.extend(["## Issues", ""])

            for severity in [
                IssueSeverity.CRITICAL,
                IssueSeverity.HIGH,
                IssueSeverity.MEDIUM,
                IssueSeverity.LOW,
                IssueSeverity.INFO,
            ]:
                severity_issues = [i for i in self.all_issues if i.severity == severity]
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