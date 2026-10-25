"""GitLab API client for fetching MR data."""

import logging
import re
from pathlib import Path
from typing import Any

import gitlab
from git import Repo

from codespy.config_utils import secret_value
from codespy.tools.git.base import GitClient, SubmittedPullRequest, SubmittedReview
from codespy.tools.git.conversation import (
    cap_comments,
    classify_author,
    count_reactions,
    has_self_signature,
    is_codespy_account,
)
from codespy.tools.git.models import (
    ChangedFile,
    FileStatus,
    GitPlatform,
    PRComment,
    PRCommit,
    PullRequest,
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
            token = secret_value(getattr(self.settings, "gitlab_token", None))
            if token:
                self._gitlab = gitlab.Gitlab(self.gitlab_url, private_token=token, timeout=30)
            else:
                self._gitlab = gitlab.Gitlab(self.gitlab_url, timeout=30)
            self._gitlab.auth()
        return self._gitlab

    def parse_url(self, url: str) -> tuple[str, str, int]:
        """Parse a GitLab MR URL into namespace/project and MR number.

        Args:
            url: GitLab MR URL

        Returns:
            Tuple of (namespace, project, pr_number)
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
        pr_number = int(match.group("number"))

        # Split path into namespace and project
        # Handle nested namespaces (e.g., group/subgroup/project)
        parts = path.rsplit("/", 1)
        if len(parts) == 2:
            namespace, project = parts
        else:
            namespace = ""
            project = parts[0]

        return namespace, project, pr_number

    def _get_project_path(self, url: str) -> str:
        """Get the full project path from URL."""
        match = self.MR_URL_PATTERN.match(url)
        if not match:
            raise ValueError(f"Invalid GitLab MR URL: {url}")
        return match.group("path")

    def _get_host(self, url: str) -> str:
        """Get the host (e.g. 'gitlab.com' or a self-hosted domain) from URL."""
        match = self.MR_URL_PATTERN.match(url)
        if not match:
            raise ValueError(f"Invalid GitLab MR URL: {url}")
        return match.group("host")

    def _map_status(self, diff_status: str) -> FileStatus:
        """Map GitLab diff status to FileStatus enum."""
        status_map = {
            "new": FileStatus.ADDED,
            "deleted": FileStatus.REMOVED,
            "renamed": FileStatus.RENAMED,
        }
        return status_map.get(diff_status, FileStatus.MODIFIED)

    def fetch_pull_request(self, url: str) -> PullRequest:
        """Fetch merge request data from GitLab.

        Args:
            url: GitLab MR URL

        Returns:
            PullRequest model with all data
        """
        namespace, project_name, pr_number = self.parse_url(url)
        project_path = self._get_project_path(url)
        host = self._get_host(url)

        # Get project and MR
        project = self.gitlab_client.projects.get(project_path)
        gl_mr = project.mergerequests.get(pr_number)

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
            additions = sum(
                1
                for line in diff.split("\n")
                if line.startswith("+") and not line.startswith("+++")
            )
            deletions = sum(
                1
                for line in diff.split("\n")
                if line.startswith("-") and not line.startswith("---")
            )

            changed_files.append(
                ChangedFile(
                    filename=change.get("new_path", change.get("old_path")),
                    status=status,
                    additions=additions,
                    deletions=deletions,
                    patch=diff,
                    previous_filename=change.get("old_path")
                    if change.get("renamed_file")
                    else None,
                )
            )

        # Map GitLab state to common state
        state_map = {"opened": "open", "closed": "closed", "merged": "merged"}
        state = state_map.get(gl_mr.state, gl_mr.state)

        pr = PullRequest(
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
            host=host,
            changed_files=changed_files,
            labels=gl_mr.labels,
            platform=GitPlatform.GITLAB,
        )
        self._fetch_conversation(gl_mr, pr)
        return pr

    # ------------------------------------------------------------------
    # Conversation (commits, discussions, award emojis)
    # ------------------------------------------------------------------

    def _self_login(self) -> str | None:
        """Username of the authenticated user, or None."""
        try:
            user = self.gitlab_client.user
            return getattr(user, "username", None) if user else None
        except Exception:
            return None

    def _fetch_conversation(self, gl_mr, pr: PullRequest) -> None:
        """Populate commits, comments and MR reactions. Soft-fails per part."""
        caps = self.settings.review.pr_context
        try:
            pr.commits = self._fetch_commits(gl_mr, caps.max_commits)
        except Exception as e:
            logger.warning(f"Failed to fetch MR commits: {e}")
        try:
            pr.comments = self._fetch_comments(gl_mr, caps.max_comments)
        except Exception as e:
            logger.warning(f"Failed to fetch MR discussions: {e}")
        try:
            pr.reactions = self._awards(gl_mr)
        except Exception as e:
            logger.debug(f"Failed to fetch MR award emojis: {e}")

    @staticmethod
    def _awards(obj) -> dict[str, int]:
        return count_reactions(
            (
                a.name,
                (a.user or {}).get("username"),
                bool((a.user or {}).get("bot")),
            )
            for a in obj.awardemojis.list(get_all=True)
        )

    @staticmethod
    def _fetch_commits(gl_mr, max_commits: int) -> list[PRCommit]:
        if max_commits <= 0:
            return []
        commits = [
            PRCommit(
                sha=getattr(c, "id", ""),
                author=getattr(c, "author_name", "") or "",
                message=getattr(c, "message", "") or "",
            )
            for c in gl_mr.commits()
        ]
        commits.reverse()  # GitLab returns newest first
        return commits[-max_commits:]

    def _fetch_comments(self, gl_mr, max_comments: int) -> list[PRComment]:
        self_login = self._self_login()
        trusted = list(getattr(self.settings.review.pr_context, "trusted_authors", []) or [])
        if not self_login and not trusted:
            logger.warning(
                "Cannot resolve the token's GitLab username and no "
                "review.pr_context.trusted_authors; feedback on earlier codespy "
                "reviews will be ignored (findings cannot be author-verified)"
            )
        comments: list[PRComment] = []
        for d in gl_mr.discussions.list(get_all=True):
            notes = [n for n in (d.attributes.get("notes") or []) if not n.get("system")]
            if not notes:
                continue
            root = notes[0]
            root_author = (root.get("author") or {}).get("username", "")
            root_origin = classify_author(
                root.get("body", ""), root_author, self_login=self_login,
                bot=bool((root.get("author") or {}).get("bot")),
            )
            thread_codespy = root_origin == "codespy"
            for i, n in enumerate(notes):
                author = n.get("author") or {}
                login = author.get("username", "")
                body = n.get("body", "") or ""
                origin = root_origin if i == 0 else classify_author(
                    body, login, self_login=self_login,
                    in_codespy_thread=thread_codespy, bot=bool(author.get("bot")),
                )
                position = n.get("position") or {}
                is_inline = bool(position) or n.get("type") == "DiffNote"
                kind = "inline" if is_inline else "issue"
                if i == 0 and not is_inline and origin == "codespy" and has_self_signature(body):
                    kind = "review"
                comments.append(
                    PRComment(
                        id=n.get("id"), kind=kind, author=login, body=body,
                        created_at=n.get("created_at"),
                        path=position.get("new_path") or position.get("old_path"),
                        line=position.get("new_line") or position.get("old_line"),
                        thread_id=str(d.id),
                        in_reply_to_id=root.get("id") if i > 0 else None,
                        resolved=n.get("resolved") if n.get("resolvable") else None,
                        origin=origin,
                        authored_by_self=is_codespy_account(login, self_login, trusted),
                    )
                )

        comments = cap_comments(comments, max_comments)

        # Award emojis: every codespy note (bounded) + up to max_comments others.
        own_budget = max_comments
        other_budget = max_comments
        for pc in reversed(comments):
            if pc.origin == "codespy":
                if pc.kind == "review" or own_budget <= 0:
                    continue
                own_budget -= 1
            else:
                if other_budget <= 0:
                    continue
                other_budget -= 1
            try:
                pc.reactions = self._awards(gl_mr.notes.get(pc.id, lazy=True))
            except Exception as e:
                logger.debug(f"Failed to fetch award emojis for note {pc.id}: {e}")
        return comments

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
            cache_dir = self.settings.review.cache_dir
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
        token = secret_value(getattr(self.settings, "gitlab_token", None))
        if token:
            host_clean = gitlab_host.replace("https://", "").replace("http://", "")
            repo_url = f"https://oauth2:{token}@{host_clean}/{project_path}.git"
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
    ) -> SubmittedReview | None:
        """Submit a review on a merge request.

        GitLab doesn't have a "review" concept like GitHub. Instead:
        - The body becomes a general MR note/comment
        - Inline comments become discussions on specific lines

        Uses optimistic approach with line validation fallback.

        Args:
            url: GitLab MR URL
            body: Review body/summary text
            comments: List of inline comment dicts with keys:
                - path: File path
                - line: Line number
                - body: Comment text
            commit_sha: Commit SHA to review (defaults to head SHA)

        Returns:
            SubmittedReview with the note ID and body suffix, or None if no note was created.
        """
        project_path = self._get_project_path(url)
        _, _, pr_number = self.parse_url(url)

        project = self.gitlab_client.projects.get(project_path)
        gl_mr = project.mergerequests.get(pr_number)

        # Get changes and commit SHA for positioning
        changes = gl_mr.changes()
        if not commit_sha:
            commit_sha = changes.get("diff_refs", {}).get("head_sha")

        # Build changed files map for line validation
        changed_files_map: dict[str, ChangedFile] = {}
        for change in changes.get("changes", []):
            filename = change.get("new_path", change.get("old_path"))
            diff = change.get("diff", "")

            # Determine status
            if change.get("new_file"):
                status = FileStatus.ADDED
            elif change.get("deleted_file"):
                status = FileStatus.REMOVED
            elif change.get("renamed_file"):
                status = FileStatus.RENAMED
            else:
                status = FileStatus.MODIFIED

            additions = sum(
                1
                for line in diff.split("\n")
                if line.startswith("+") and not line.startswith("+++")
            )
            deletions = sum(
                1
                for line in diff.split("\n")
                if line.startswith("-") and not line.startswith("---")
            )

            changed_files_map[filename] = ChangedFile(
                filename=filename,
                status=status,
                additions=additions,
                deletions=deletions,
                patch=diff,
                previous_filename=change.get("old_path") if change.get("renamed_file") else None,
            )

        # Track failed comments to append to body
        failed_comments: list[dict] = []
        successful_count = 0

        # Post inline comments as discussions
        if comments:
            for comment in comments:
                path = comment["path"]
                line = comment.get("line")
                changed_file = changed_files_map.get(path)

                # Skip if file not in MR
                if not changed_file:
                    logger.warning(f"Skipping comment for file not in MR: {path}")
                    failed_comments.append(comment)
                    continue

                # Pre-validate line is in diff
                if line and not changed_file.is_line_in_diff(line):
                    logger.debug(f"Comment on {path}:{line} not in diff, will add to body")
                    failed_comments.append(comment)
                    continue

                # Create a discussion on the specific line
                position = {
                    "base_sha": gl_mr.diff_refs.get("base_sha"),
                    "head_sha": commit_sha,
                    "start_sha": gl_mr.diff_refs.get("start_sha"),
                    "new_path": path,
                    "new_line": line,
                    "position_type": "text",
                }

                # Handle multi-line comments
                if "start_line" in comment:
                    position["old_line"] = comment.get("start_line")

                try:
                    gl_mr.discussions.create(
                        {
                            "body": comment["body"],
                            "position": position,
                        }
                    )
                    successful_count += 1
                except gitlab.exceptions.GitlabCreateError as e:
                    # Track failed comment to include in body
                    logger.warning(f"Failed to create inline comment on {path}:{line}: {e}")
                    failed_comments.append(comment)

        # Build final body with any failed comments
        final_body = body
        if failed_comments:
            final_body = self._append_comments_to_body(body, failed_comments)
            logger.info(f"Moving {len(failed_comments)} comments to body (could not post inline)")

        # Post main review body as a note (with any failed comments appended)
        note_id = 0
        body_suffix = ""
        if final_body:
            note = gl_mr.notes.create({"body": final_body})
            note_id = note.id
            body_suffix = final_body[len(body):]
            logger.info(f"Posted review on {project_path}!{pr_number}")

        if comments:
            logger.info(
                f"Submitted {successful_count}/{len(comments)} "
                f"inline comments on {project_path}!{pr_number}"
            )

        if note_id:
            return SubmittedReview(id=note_id, body_suffix=body_suffix)
        return None

    def update_review(
        self,
        url: str,
        review: SubmittedReview,
        body: str,
    ) -> None:
        """Update an existing GitLab MR note with a new body.

        Args:
            url: GitLab MR URL
            review: The SubmittedReview returned by submit_review
            body: New review body (replaces the original)
        """
        project_path = self._get_project_path(url)
        _, _, pr_number = self.parse_url(url)

        project = self.gitlab_client.projects.get(project_path)
        gl_mr = project.mergerequests.get(pr_number)

        # Get the note and update it
        note = gl_mr.notes.get(review.id)
        note.body = body
        note.save()
        logger.info(f"Updated note {review.id} on {project_path}!{pr_number}")

    def _append_comments_to_body(self, body: str, comments: list[dict]) -> str:
        """Append inline comments to the review body when they can't be posted inline.

        Args:
            body: Original review body
            comments: List of comment dicts that couldn't be posted inline

        Returns:
            Updated body with comments appended
        """
        if not comments:
            return body

        lines = [body, "", "---", "", "## ⚠️ Additional Issues (could not post inline)", ""]

        for comment in comments:
            path = comment.get("path", "unknown")
            line = comment.get("line", "?")
            comment_body = comment.get("body", "")

            lines.extend(
                [
                    f"### `{path}:{line}`",
                    "",
                    comment_body,
                    "",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _project_path(owner: str, repo: str) -> str:
        return f"{owner}/{repo}" if owner else repo

    def get_default_branch(self, owner: str, repo: str) -> str:
        """Get the default branch for a project."""
        return str(self.gitlab_client.projects.get(self._project_path(owner, repo)).default_branch)

    def resolve_branch(self, owner: str, repo: str, branch: str) -> str:
        """Resolve a branch name to its head commit SHA (ValueError if not a branch)."""
        project = self.gitlab_client.projects.get(self._project_path(owner, repo))
        try:
            return str(project.branches.get(branch).commit["id"])
        except gitlab.exceptions.GitlabGetError as e:
            raise ValueError(
                f"'{branch}' is not a branch of {self._project_path(owner, repo)} "
                "(full-path reviews need a branch as the report MR base)"
            ) from e

    @staticmethod
    def _put_file(project: Any, path: str, content: str, message: str, branch: str) -> None:
        """Create or update one file on ``branch`` in a single commit."""
        try:
            project.files.get(file_path=path, ref=branch)
            action = "update"
        except gitlab.exceptions.GitlabGetError:
            action = "create"
        project.commits.create(
            {
                "branch": branch,
                "commit_message": message,
                "actions": [{"action": action, "file_path": path, "content": content}],
            }
        )

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
        """Create or update the report MR for a full-path review."""
        project_path = self._project_path(owner, repo)
        project = self.gitlab_client.projects.get(project_path)

        open_mrs = project.mergerequests.list(
            state="opened", source_branch=head_branch, target_branch=base_branch
        )
        if open_mrs:
            # Re-run: update the existing report MR, never touch its history
            mr = project.mergerequests.get(open_mrs[0].iid)
            self._put_file(project, file_path, file_content, commit_message, head_branch)
            mr.description = body
            mr.save()
            mr.notes.create({"body": update_comment})
            logger.info(f"Updated report MR !{mr.iid} on {project_path}")
            return SubmittedPullRequest(number=mr.iid, url=mr.web_url, created=False)

        branch_created = False
        try:
            try:
                project.branches.get(head_branch)
            except gitlab.exceptions.GitlabGetError:
                pass
            else:
                # Stale branch from a closed/merged report MR: restart from base
                project.branches.delete(head_branch)
            project.branches.create({"branch": head_branch, "ref": base_sha})
            branch_created = True

            self._put_file(project, file_path, file_content, commit_message, head_branch)
            mr = project.mergerequests.create(
                {
                    "source_branch": head_branch,
                    "target_branch": base_branch,
                    "title": title,
                    "description": body,
                }
            )
        except Exception:
            if branch_created:
                try:
                    project.branches.delete(head_branch)
                    logger.warning(f"Deleted branch {head_branch} after a failed report MR")
                except Exception as cleanup_err:
                    logger.warning(f"Could not delete branch {head_branch}: {cleanup_err}")
            raise
        logger.info(f"Opened report MR !{mr.iid} on {project_path}")
        return SubmittedPullRequest(number=mr.iid, url=mr.web_url, created=True)

    def update_pull_request_body(self, url: str, body: str) -> None:
        """Replace a merge request's description."""
        project_path = self._get_project_path(url)
        _, _, mr_iid = self.parse_url(url)
        mr = self.gitlab_client.projects.get(project_path).mergerequests.get(mr_iid)
        mr.description = body
        mr.save()
        logger.info(f"Updated MR !{mr_iid} description on {project_path}")
