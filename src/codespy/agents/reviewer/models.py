"""Data models for code review results."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from codespy.agents.memory.hippocampus import ContextMemory
from codespy.tools.git.models import ChangedFile, PullRequest

if TYPE_CHECKING:
    from codespy.agents.memory.hippocampus.context_memory import Topic


class PRContext(BaseModel):
    """Shared PR identity passed to all review modules after summarization.

    Built by the pipeline orchestrator after the Summarizer runs, then
    threaded through scope identification, review modules, and audit.
    Each module constructs its own Hippocampus question from these fields.
    """

    repo_slug: str = Field(
        description="Host-qualified repo identifier (e.g. github.com/owner/repo)"
    )
    pr_number: int = Field(description="PR number")
    pr_title: str = Field(description="PR title")
    pr_url: str = Field(description="Full PR URL (e.g. https://github.com/owner/repo/pull/123)")
    pr_description: str = Field(default="", description="PR body/description")
    summary: str = Field(description="2-3 sentence PR summary produced by Summarizer")

    def to_topic(self) -> "Topic":
        """Build a Topic representing this PR.

        Returns:
            Topic object with id as PR URL and description as "PR #N: Title"
        """
        from codespy.agents.memory.hippocampus.context_memory import Topic
        return Topic(
            id=self.pr_url,
            description=f"PR #{self.pr_number}: {self.pr_title}"[:500],
        )


class IssueSeverity(StrEnum):
    """Severity level of an issue."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(StrEnum):
    """Category of an issue."""

    SECURITY = "security"
    BUG = "bug"
    DOCUMENTATION = "documentation"
    SMELL = "smell"


class ScopeType(StrEnum):
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
    package_name: str | None = Field(default=None, description="Package identity from manifest")


class ReviewMetadata(BaseModel):
    """Runtime pipeline state, stable once constructed at pipeline start.

    Groups repo_path, run_id, pr, and is_local to reduce parameter
    proliferation across module method signatures.
    """

    repo_path: Path
    run_id: str | None = None
    pr: PullRequest | None = None
    is_local: bool = False


class ReviewContext(BaseModel):
    """Evolving pipeline state threaded through review stages.

    Carries the immutable PR identity and runtime pipeline metadata.
    Context memory is loaded independently by each module from its own prior episodes.
    """

    pr_context: PRContext = Field(
        description="Immutable PR identity (repo, number, title, summary)"
    )
    memory: ContextMemory | None = Field(
        default=None, description="Unused — each module loads its own prior episodes. Kept for API compatibility."
    )
    metadata: ReviewMetadata | None = Field(default=None, description="Runtime pipeline state")


class ScopeResult(BaseModel):
    """A detected scope/subroot in the repository."""

    repo: str = Field(
        default="", description="Repo identifier: 'owner/repo' (remote) or local dir name"
    )
    subroot: str = Field(description="Path relative to repo root (e.g., packages/auth)")
    scope_type: ScopeType = Field(description="Type of scope (library, service, etc.)")
    has_changes: bool = Field(
        default=False, description="Whether this scope has changed files from PR"
    )
    is_dependency: bool = Field(
        default=False, description="Whether this scope depends on a changed scope"
    )
    language: str | None = Field(default=None, description="Primary language detected")
    package_manifest: PackageManifest | None = Field(
        default=None, description="Package manifest info if present"
    )
    changed_files: list[ChangedFile] = Field(
        default_factory=list, description="Changed files belonging to this scope"
    )
    reason: str = Field(description="Explanation for why this scope was identified")
    skills: str | None = Field(
        default=None, description="Project/scope instructions inherited from ancestor directories"
    )
    description: str = Field(
        default="", description="Description of scope's role in the project (max 500 chars)"
    )

    model_config = {"arbitrary_types_allowed": True}

    def scope_path(self) -> str:
        """Return the storage-relative path for this scope.

        Used by Hippocampus memory as the base directory for episode files.
        ``subroot == "."`` (repo root) results in ``/{repo}/``.
        """
        if self.subroot in (".", ""):
            return f"/{self.repo}/"
        return f"/{self.repo}/{self.subroot.strip('/')}/"

    def topic(self, repo_full_name: str) -> Topic:
        """Build the Topic for this scope.

        Args:
            repo_full_name: Repository full name (owner/repo)

        Returns:
            Topic object with id and description
        """
        from codespy.agents.memory.hippocampus.context_memory import Topic, make_topic_id

        package_name = self.package_manifest.package_name if self.package_manifest else None
        topic_id = make_topic_id(repo_full_name, self.subroot, package_name)
        return Topic(id=topic_id, description=self.description)


class Issue(BaseModel):
    """Represents a single issue found during review."""

    category: IssueCategory = Field(description="Issue category")
    severity: IssueSeverity = Field(
        default=IssueSeverity.MEDIUM,
        description="Issue severity: critical, high, medium, low, or info",
    )
    title: str = Field(description="Brief title of the issue")
    description: str = Field(description="≤25 word imperative description. No filler.")
    filename: str = Field(description="File where the issue was found")
    line_start: int | None = Field(default=None, description="Starting line number")
    line_end: int | None = Field(default=None, description="Ending line number")
    code_snippet: str | None = Field(
        default=None, description="Deprecated—use line numbers. Leave None."
    )
    suggestion: str | None = Field(default=None, description="Suggested fix or improvement")
    cwe_id: str | None = Field(
        default=None, description="CWE ID for security issues (e.g., CWE-79)"
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score (0-1)")

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

    name: str = Field(
        description="Signature name (e.g., code_review, doc, scope, supply_chain, summary, audit)"
    )
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
    issues: list[Issue] = Field(default_factory=list, description="All issues found during review")
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
    def smell_issues(self) -> list[Issue]:
        """Get all code smell issues."""
        return [i for i in self.issues if i.category == IssueCategory.SMELL]

    def issues_by_severity(self) -> dict[IssueSeverity, list[Issue]]:
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
                    f"- **Total Tokens:** {self.total_tokens:,}",
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
                        "| Signature | Cost | Tokens | Calls | Duration |",
                        "|-----------|------|--------|-------|----------|",
                    ]
                )
                for stats in sorted(self.signature_stats, key=lambda x: x.cost, reverse=True):
                    duration_str = f"{stats.duration_seconds:.1f}s"
                    lines.append(
                        f"| {stats.name} | ${stats.cost:.4f} | {stats.tokens:,} | "
                        f"{stats.call_count} | {duration_str} |"
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
