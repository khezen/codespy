"""GitLab API client for fetching MR data."""

import logging
import re
from pathlib import Path

import gitlab
from git import Repo

from codespy.tools.git.base import GitClient
from codespy.tools.git.models import (
    ChangedFile,
    FileStatus,
    GitPlatform,
    MergeRequest,
)

logger = logging.getLogger(__name__)


class GitLabClient(GitClient):
    """Client for interacting with GitLab API."""

    # Pattern for gitlab.com URLs
    MR_URL_PATTERN = re.compile(
        r"https?://(?P<host>[^/]+)/(?P<path>.+?)/-/merge_requests/(?P<number>\d+)"
    )

    def __init__(self, settings=None) -> None:
        """Initialize the GitLab client."""
        super().__init__(settings)
        self._gitlab: gitlab.Gitlab | None = None

    @property
    def platform_name(self) -> str:
        """Return the platform name."""
        return "GitLab"

    @staticmethod
    def can_handle(url: str) -> bool:
        """Check if this client can handle the given URL."""
        return "/-/merge_requests/" in url

    @property
    def gitlab_url(self) -> str:
        """Get the GitLab instance URL."""
        return getattr(self.settings, "gitlab_url", None) or "https://gitlab.com"

    @property
    def gitlab_client(self) -> gitlab.Gitlab:
        """Get or create GitLab client instance."""
        if self._gitlab is None:
            token = getattr(self.settings, "gitlab_token", None)
            if token:
                self._gitlab = gitlab.Gitlab(self.gitlab_url, private_token=token)
            else:
                self._gitlab = gitlab.Gitlab(self.gitlab_url)
            self._gitlab.auth()
        return self._gitlab

    def parse_url(self, url: str) -> tuple[str, str, int]:
        """Parse a GitLab MR URL into namespace/project and MR number.

        Args:
            url: GitLab MR URL

        Returns:
            Tuple of (namespace, project, mr_number)
            Note: namespace may contain slashes for nested groups

        Raises:
            ValueError: If URL is not a valid GitLab MR URL
        """
        match = self.MR_URL_PATTERN.match(url)
        if not match:
            raise ValueError(
                f"Invalid GitLab MR URL: {url}. "
                "Expected format: https://gitlab.com/namespace/project/-/merge_requests/123"
            )

        path = match.group("path")
        mr_number = int(match.group("number"))

        # Split path into namespace and project
        # Handle nested namespaces (e.g., group/subgroup/project)
        parts = path.rsplit("/", 1)
        if len(parts) == 2:
            namespace, project = parts
        else:
            namespace = ""
            project = parts[0]

        return namespace, project, mr_number

    def _get_project_path(self, url: str) -> str:
        """Get the full project path from URL."""
        match = self.MR_URL_PATTERN.match(url)
        if not match:
            raise ValueError(f"Invalid GitLab MR URL: {url}")
        return match.group("path")

    def _map_status(self, diff_status: str) -> FileStatus:
        """Map GitLab diff status to FileStatus enum."""
        status_map = {
            "new": FileStatus.ADDED,
            "deleted": FileStatus.REMOVED,
            "renamed": FileStatus.RENAMED,
        }
        return status_map.get(diff_status, FileStatus.MODIFIED)

    def fetch_merge_request(self, url: str) -> MergeRequest:
        """Fetch merge request data from GitLab.

        Args:
            url: GitLab MR URL

        Returns:
            MergeRequest model with all data
        """
        namespace, project_name, mr_number = self.parse_url(url)
        project_path = self._get_project_path(url)

        # Get project and MR
        project = self.gitlab_client.projects.get(project_path)
        gl_mr = project.mergerequests.get(mr_number)

        # Get diff/changes
        changes = gl_mr.changes()

        # Build changed files list
        changed_files: list[ChangedFile] = []
        for change in changes.get("changes", []):
            # Determine status
            if change.get("new_file"):
                status = FileStatus.ADDED
            elif change.get("deleted_file"):
                status = FileStatus.REMOVED
            elif change.get("renamed_file"):
                status = FileStatus.RENAMED
            else:
                status = FileStatus.MODIFIED

            # Calculate additions/deletions from diff
            diff = change.get("diff", "")
            additions = sum(1 for line in diff.split("\n") if line.startswith("+") and not line.startswith("+++"))
            deletions = sum(1 for line in diff.split("\n") if line.startswith("-") and not line.startswith("---"))

            changed_files.append(
                ChangedFile(
                    filename=change.get("new_path", change.get("old_path")),
                    status=status,
                    additions=additions,
                    deletions=deletions,
                    patch=diff,
                    previous_filename=change.get("old_path") if change.get("renamed_file") else None,
                )
            )

        # Map GitLab state to common state
        state_map = {"opened": "open", "closed": "closed", "merged": "merged"}
        state = state_map.get(gl_mr.state, gl_mr.state)

        return MergeRequest(
            number=gl_mr.iid,
            title=gl_mr.title,
            body=gl_mr.description,
            state=state,
            author=gl_mr.author.get("username", "unknown"),
            base_branch=gl_mr.target_branch,
            head_branch=gl_mr.source_branch,
            base_sha=changes.get("diff_refs", {}).get("base_sha", ""),
            head_sha=changes.get("diff_refs", {}).get("head_sha", ""),
            created_at=gl_mr.created_at,
            updated_at=gl_mr.updated_at,
            repo_owner=namespace,
            repo_name=project_name,
            changed_files=changed_files,
            labels=gl_mr.labels,
            platform=GitPlatform.GITLAB,
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
            owner: Repository namespace (may include nested groups)
            repo_name: Repository name
            ref: Git ref (branch, tag, or commit) to checkout
            target_path: Target directory for clone
            depth: Shallow clone depth (None for full history)
            sparse_paths: List of paths for sparse checkout

        Returns:
            Path to the cloned repository
        """
        # Determine target directory
        if target_path is None:
            cache_dir = self.settings.cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Handle nested namespaces
            repo_dir = cache_dir / owner.replace("/", "_") / repo_name
        else:
            repo_dir = target_path

        # Build full project path
        project_path = f"{owner}/{repo_name}" if owner else repo_name

        # Get GitLab instance URL
        gitlab_host = self.gitlab_url.rstrip("/")

        # Build clone URL
        token = getattr(self.settings, "gitlab_token", None)
        if token:
            repo_url = f"https://oauth2:{token}@{gitlab_host.replace('https://', '').replace('http://', '')}/{project_path}.git"
        else:
            repo_url = f"{gitlab_host}/{project_path}.git"

        if repo_dir.exists() and (repo_dir / ".git").exists():
            # Update existing clone
            repo = Repo(repo_dir)
            repo.remotes.origin.fetch()
            repo.git.checkout(ref)
        else:
            # Fresh clone
            repo_dir.mkdir(parents=True, exist_ok=True)

            if sparse_paths:
                # Sparse checkout
                repo = Repo.init(repo_dir)
                repo.create_remote("origin", repo_url)
                repo.git.config("core.sparseCheckout", "true")

                sparse_file = repo_dir / ".git" / "info" / "sparse-checkout"
                sparse_file.parent.mkdir(parents=True, exist_ok=True)
                sparse_file.write_text("\n".join(sparse_paths) + "\n")

                fetch_args = ["origin", ref]
                if depth:
                    fetch_args.extend(["--depth", str(depth)])
                fetch_args.extend(["--filter=tree:0"])
                repo.git.fetch(*fetch_args)
                repo.git.checkout(ref)
            else:
                clone_kwargs: dict = {"no_single_branch": True}
                if depth:
                    clone_kwargs["depth"] = depth

                repo = Repo.clone_from(repo_url, repo_dir, **clone_kwargs)
                repo.git.checkout(ref)

        return repo_dir

    def submit_review(
        self,
        url: str,
        body: str,
        comments: list[dict] | None = None,
        commit_sha: str | None = None,
    ) -> None:
        """Submit a review on a merge request.

        GitLab doesn't have a "review" concept like GitHub. Instead:
        - The body becomes a general MR note/comment
        - Inline comments become discussions on specific lines

        Args:
            url: GitLab MR URL
            body: Review body/summary text
            comments: List of inline comment dicts with keys:
                - path: File path
                - line: Line number
                - body: Comment text
            commit_sha: Commit SHA to review (defaults to head SHA)
        """
        project_path = self._get_project_path(url)
        _, _, mr_number = self.parse_url(url)

        project = self.gitlab_client.projects.get(project_path)
        gl_mr = project.mergerequests.get(mr_number)

        # Get commit SHA for positioning
        if not commit_sha:
            changes = gl_mr.changes()
            commit_sha = changes.get("diff_refs", {}).get("head_sha")

        # Post main review body as a note
        if body:
            gl_mr.notes.create({"body": body})
            logger.info(f"Posted review summary on {project_path}!{mr_number}")

        # Post inline comments as discussions
        if comments:
            for comment in comments:
                # Create a discussion on the specific line
                position = {
                    "base_sha": gl_mr.diff_refs.get("base_sha"),
                    "head_sha": commit_sha,
                    "start_sha": gl_mr.diff_refs.get("start_sha"),
                    "new_path": comment["path"],
                    "new_line": comment["line"],
                    "position_type": "text",
                }

                # Handle multi-line comments
                if "start_line" in comment:
                    position["old_line"] = comment.get("start_line")

                try:
                    gl_mr.discussions.create({
                        "body": comment["body"],
                        "position": position,
                    })
                except gitlab.exceptions.GitlabCreateError as e:
                    # Fall back to regular note if position fails
                    logger.warning(f"Failed to create inline comment, falling back to note: {e}")
                    fallback_body = f"**{comment['path']}:{comment['line']}**\n\n{comment['body']}"
                    gl_mr.notes.create({"body": fallback_body})

            logger.info(
                f"Submitted {len(comments)} inline comments on {project_path}!{mr_number}"
            )