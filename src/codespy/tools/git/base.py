"""Abstract base class for Git platform clients."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codespy.config import Settings
    from codespy.tools.git.models import MergeRequest


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
        """Parse a merge request URL into owner, repo, and MR number.

        Args:
            url: Merge request URL

        Returns:
            Tuple of (owner, repo, mr_number)

        Raises:
            ValueError: If URL is not valid for this platform
        """
        ...

    @abstractmethod
    def fetch_merge_request(self, url: str) -> "MergeRequest":
        """Fetch merge request data from the platform.

        Args:
            url: Merge request URL

        Returns:
            MergeRequest model with all data
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
    ) -> None:
        """Submit a review on a merge request.

        Args:
            url: Merge request URL
            body: Review body/summary text
            comments: List of inline comment dicts with keys:
                - path: File path
                - line: Line number
                - body: Comment text
            commit_sha: Commit SHA to review (defaults to head SHA)
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