"""Abstract base class for Git platform clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codespy.config import Settings
    from codespy.tools.git.models import PullRequest


@dataclass
class SubmittedReview:
    """Reference to a submitted review for later editing.

    Attributes:
        id: The review/note ID from the platform
        body_suffix: Any content appended to the body (e.g., fallback inline comments)
    """
    id: int
    body_suffix: str = ""


@dataclass
class SubmittedPullRequest:
    """A report PR/MR opened or updated by codespy.

    Attributes:
        number: PR number (GitHub) or MR iid (GitLab)
        url: Web URL of the PR/MR
        created: True when this call opened it, False when it updated an open one
    """

    number: int
    url: str
    created: bool


class GitClient(ABC):
    """Abstract base class for Git platform clients (GitHub, GitLab)."""

    def __init__(self, settings: "Settings | None" = None) -> None:
        """Initialize the Git client.

        Args:
            settings: Application settings. Uses global settings if not provided.
        """
        from codespy.config import get_settings

        self.settings = settings or get_settings()

    @abstractmethod
    def parse_url(self, url: str) -> tuple[str, str, int]:
        """Parse a pull request URL into owner, repo, and PR number.

        Args:
            url: Pull request URL

        Returns:
            Tuple of (owner, repo, pr_number)

        Raises:
            ValueError: If URL is not valid for this platform
        """
        ...

    @abstractmethod
    def fetch_pull_request(self, url: str) -> "PullRequest":
        """Fetch pull request data from the platform.

        Args:
            url: Pull request URL

        Returns:
            PullRequest model with all data
        """
        ...

    @abstractmethod
    def clone_repository(
        self,
        owner: str,
        repo_name: str,
        ref: str,
        target_path: Path | None = None,
        depth: int | None = 1,
        sparse_paths: list[str] | None = None,
    ) -> Path:
        """Clone or update a repository.

        Args:
            owner: Repository owner
            repo_name: Repository name
            ref: Git ref (branch, tag, or commit) to checkout
            target_path: Target directory for clone
            depth: Shallow clone depth (None for full history)
            sparse_paths: List of paths for sparse checkout

        Returns:
            Path to the cloned repository
        """
        ...

    @abstractmethod
    def submit_review(
        self,
        url: str,
        body: str,
        comments: list[dict] | None = None,
        commit_sha: str | None = None,
    ) -> SubmittedReview | None:
        """Submit a review on a pull request.

        Args:
            url: Pull request URL
            body: Review body/summary text
            comments: List of inline comment dicts with keys:
                - path: File path
                - line: Line number
                - body: Comment text
            commit_sha: Commit SHA to review (defaults to head SHA)

        Returns:
            SubmittedReview with the review ID and any body suffix appended,
            or None if no review was created.
        """
        ...

    @abstractmethod
    def update_review(
        self,
        url: str,
        review: SubmittedReview,
        body: str,
    ) -> None:
        """Update an existing review with a new body.

        Args:
            url: Pull request URL
            review: The SubmittedReview returned by submit_review
            body: New review body (replaces the original)
        """
        ...

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform name (e.g., 'GitHub', 'GitLab')."""
        ...

    @staticmethod
    @abstractmethod
    def can_handle(url: str) -> bool:
        """Check if this client can handle the given URL.

        Args:
            url: URL to check

        Returns:
            True if this client can handle the URL
        """
        ...

    @abstractmethod
    def get_default_branch(self, owner: str, repo: str) -> str:
        """Get the default branch for a repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Default branch name (e.g., 'main', 'master')
        """
        ...

    @abstractmethod
    def resolve_branch(self, owner: str, repo: str, branch: str) -> str:
        """Resolve a branch name to a commit SHA.

        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name

        Returns:
            Commit SHA

        Raises:
            ValueError: If branch is not found or is not a branch
        """
        ...

    @abstractmethod
    def upsert_report_pull_request(
        self,
        owner: str,
        repo: str,
        base_branch: str,
        base_sha: str,
        head_branch: str,
        file_path: str,
        file_content: str,
        commit_message: str,
        title: str,
        body: str,
        update_comment: str,
    ) -> SubmittedPullRequest:
        """Create or update a report pull request.

        If an open PR exists with ``head_branch`` -> ``base_branch``, commits the
        file to that branch, replaces the body and posts ``update_comment``.
        Otherwise creates (or force-resets a stale) ``head_branch`` at
        ``base_sha``, commits the file and opens the PR.

        Returns:
            SubmittedPullRequest (number, url, created)

        Raises:
            Exception: On failure. A branch created by this call is deleted
                first; a branch that belongs to an open PR is never deleted.
        """
        ...

    @abstractmethod
    def update_pull_request_body(self, url: str, body: str) -> None:
        """Update an existing pull request's body/description.

        Args:
            url: Pull request URL
            body: New PR body
        """
        ...
