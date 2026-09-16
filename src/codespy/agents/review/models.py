"""Cross-cutting review types."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from codespy.agents.memory.hippocampus import ContextMemory
from codespy.tools.git.models import PullRequest

if TYPE_CHECKING:
    from codespy.agents.memory.hippocampus.context_memory import Topic


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

    @property
    def repo_full_name(self) -> str:
        """Get owner/repo from host-qualified repo_slug."""
        # repo_slug is "github.com/owner/repo" or just "owner/repo"
        parts = self.repo_slug.split("/")
        if len(parts) >= 3:
            return "/".join(parts[1:])  # strip host
        return self.repo_slug

    def to_topic(self) -> "Topic":
        """Build a Topic representing this PR.

        Returns:
            Topic object with id as PR URL, type as "pull_request", and description as "PR #N: Title"
        """
        from codespy.agents.memory.hippocampus.context_memory import Topic
        return Topic(
            id=self.pr_url,
            type="pull_request",
            description=f"PR #{self.pr_number}: {self.pr_title}"[:500],
        )


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



