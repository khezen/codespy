"""GitHub API client for fetching PR data."""

import logging
import re
from pathlib import Path

from git import Repo
from github import Auth, Github
from github.PullRequest import PullRequest as GHPullRequest

from codespy.config import Settings, get_settings
from codespy.tools.github.models import ChangedFile, FileStatus, PullRequest

logger = logging.getLogger(__name__)


class GitHubClient:
    """Client for interacting with GitHub API."""

    PR_URL_PATTERN = re.compile(
        r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
    )

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the GitHub client.

        Args:
            settings: Application settings. Uses global settings if not provided.
        """
        self.settings = settings or get_settings()
        self._github: Github | None = None

    @property
    def github(self) -> Github:
        """Get or create GitHub client instance."""
        if self._github is None:
            if self.settings.github_token:
                auth = Auth.Token(self.settings.github_token)
                self._github = Github(auth=auth)
            else:
                self._github = Github()
        return self._github

    def parse_pr_url(self, url: str) -> tuple[str, str, int]:
        """Parse a GitHub PR URL into owner, repo, and PR number.

        Args:
            url: GitHub PR URL

        Returns:
            Tuple of (owner, repo, pr_number)

        Raises:
            ValueError: If URL is not a valid GitHub PR URL
        """
        match = self.PR_URL_PATTERN.match(url)
        if not match:
            raise ValueError(
                f"Invalid GitHub PR URL: {url}. "
                "Expected format: https://github.com/owner/repo/pull/123"
            )
        return match.group("owner"), match.group("repo"), int(match.group("number"))

    def fetch_pull_request(self, pr_url: str) -> PullRequest:
        """Fetch pull request data from GitHub.

        Args:
            pr_url: GitHub PR URL

        Returns:
            PullRequest model with all data
        """
        owner, repo_name, pr_number = self.parse_pr_url(pr_url)

        # Get repository and PR
        repo = self.github.get_repo(f"{owner}/{repo_name}")
        gh_pr: GHPullRequest = repo.get_pull(pr_number)

        # Build changed files list
        changed_files: list[ChangedFile] = []
        for file in gh_pr.get_files():
            status = FileStatus(file.status)
            changed_files.append(
                ChangedFile(
                    filename=file.filename,
                    status=status,
                    additions=file.additions,
                    deletions=file.deletions,
                    patch=file.patch,
                    previous_filename=file.previous_filename,
                )
            )

        return PullRequest(
            number=gh_pr.number,
            title=gh_pr.title,
            body=gh_pr.body,
            state=gh_pr.state,
            author=gh_pr.user.login,
            base_branch=gh_pr.base.ref,
            head_branch=gh_pr.head.ref,
            base_sha=gh_pr.base.sha,
            head_sha=gh_pr.head.sha,
            created_at=gh_pr.created_at,
            updated_at=gh_pr.updated_at,
            repo_owner=owner,
            repo_name=repo_name,
            changed_files=changed_files,
            labels=[label.name for label in gh_pr.labels],
        )

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
            target_path: Target directory for clone (defaults to cache_dir/owner/repo_name)
            depth: Shallow clone depth (None for full history, default 1)
            sparse_paths: List of paths for sparse checkout (None for full checkout)

        Returns:
            Path to the cloned repository
        """
        # Determine target directory
        if target_path is None:
            cache_dir = self.settings.cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)
            repo_dir = cache_dir / owner / repo_name
        else:
            repo_dir = target_path

        repo_url = f"https://github.com/{owner}/{repo_name}.git"
        if self.settings.github_token:
            repo_url = f"https://{self.settings.github_token}@github.com/{owner}/{repo_name}.git"

        if repo_dir.exists() and (repo_dir / ".git").exists():
            # Update existing clone
            repo = Repo(repo_dir)
            repo.remotes.origin.fetch()
            # Checkout the specific ref
            repo.git.checkout(ref)
        else:
            # Fresh clone with optimal settings
            repo_dir.mkdir(parents=True, exist_ok=True)

            if sparse_paths:
                # Sparse checkout: init repo, configure sparse, then fetch
                repo = Repo.init(repo_dir)
                repo.create_remote("origin", repo_url)

                # Enable sparse checkout
                repo.git.config("core.sparseCheckout", "true")

                # Write sparse paths
                sparse_file = repo_dir / ".git" / "info" / "sparse-checkout"
                sparse_file.parent.mkdir(parents=True, exist_ok=True)
                sparse_file.write_text("\n".join(sparse_paths) + "\n")

                # Fetch with depth and filter for efficiency
                fetch_args = ["origin", ref]
                if depth:
                    fetch_args.extend(["--depth", str(depth)])
                # Use treeless clone for sparse checkout efficiency
                fetch_args.extend(["--filter=tree:0"])
                repo.git.fetch(*fetch_args)

                # Checkout
                repo.git.checkout(ref)
            else:
                # Standard clone
                clone_kwargs: dict = {"no_single_branch": True}
                if depth:
                    clone_kwargs["depth"] = depth

                repo = Repo.clone_from(repo_url, repo_dir, **clone_kwargs)
                repo.git.checkout(ref)

        return repo_dir

    def submit_review(
        self,
        pr_url: str,
        body: str,
        comments: list[dict] | None = None,
        commit_sha: str | None = None,
    ) -> None:
        """Submit a review on a pull request.

        Args:
            pr_url: GitHub PR URL
            body: Review body/summary text
            comments: List of inline comment dicts with keys:
                - path: File path
                - line: Line number (in the diff, not the file)
                - body: Comment text
                - side: 'RIGHT' for additions, 'LEFT' for deletions (default: RIGHT)
            commit_sha: Commit SHA to review (defaults to PR head SHA)

        Note:
            Uses event='COMMENT' to avoid approving/requesting changes.
            The bot should not make approval decisions - that's for humans.
        """
        owner, repo_name, pr_number = self.parse_pr_url(pr_url)

        repo = self.github.get_repo(f"{owner}/{repo_name}")
        gh_pr: GHPullRequest = repo.get_pull(pr_number)

        # Use provided commit SHA or default to head
        review_commit = commit_sha or gh_pr.head.sha

        # Build comments list for the API
        review_comments = []
        if comments:
            for comment in comments:
                review_comment = {
                    "path": comment["path"],
                    "body": comment["body"],
                    "side": comment.get("side", "RIGHT"),
                }
                # Use line for single-line comments
                if "line" in comment:
                    review_comment["line"] = comment["line"]
                # Support multi-line comments
                if "start_line" in comment and "line" in comment:
                    review_comment["start_line"] = comment["start_line"]
                review_comments.append(review_comment)

        # Submit the review
        gh_pr.create_review(
            commit=repo.get_commit(review_commit),
            body=body,
            event="COMMENT",
            comments=review_comments,
        )

        logger.info(
            f"Submitted review on {owner}/{repo_name}#{pr_number} "
            f"with {len(review_comments)} inline comments"
        )
