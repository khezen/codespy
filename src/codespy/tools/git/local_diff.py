"""Build MergeRequest objects from local git state (no GitHub/GitLab needed)."""

import logging
import subprocess
from datetime import datetime
from pathlib import Path

from codespy.tools.git.models import ChangedFile, FileStatus, GitPlatform, MergeRequest

logger = logging.getLogger(__name__)


def _run_git(repo_path: Path, *args: str) -> str:
    """Run a git command in the given repo and return stdout."""
    result = subprocess.run(
        ["git", "-C", str(repo_path)] + list(args),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _parse_status_char(char: str) -> FileStatus:
    """Map git diff --name-status letter to FileStatus."""
    mapping = {
        "A": FileStatus.ADDED,
        "M": FileStatus.MODIFIED,
        "D": FileStatus.REMOVED,
        "R": FileStatus.RENAMED,
        "C": FileStatus.ADDED,  # Copied → treat as added
        "T": FileStatus.MODIFIED,  # Type change
    }
    # Handle Rxxx (renamed with similarity %)
    return mapping.get(char[0], FileStatus.MODIFIED)


def _count_diff_lines(patch: str) -> tuple[int, int]:
    """Count additions and deletions from a unified diff patch."""
    additions = 0
    deletions = 0
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return additions, deletions


def _get_repo_info(repo_path: Path) -> tuple[str, str]:
    """Extract owner and repo name from git remote or directory name.

    Returns:
        Tuple of (owner, repo_name)
    """
    try:
        remote_url = _run_git(repo_path, "remote", "get-url", "origin")
        # Handle SSH: git@github.com:owner/repo.git
        if remote_url.startswith("git@"):
            path_part = remote_url.split(":", 1)[1]
        # Handle HTTPS: https://github.com/owner/repo.git
        elif "://" in remote_url:
            path_part = remote_url.split("://", 1)[1].split("/", 1)[1]
        else:
            path_part = remote_url

        # Remove .git suffix
        path_part = path_part.removesuffix(".git")
        parts = path_part.strip("/").split("/")
        if len(parts) >= 2:
            return parts[-2], parts[-1]
    except (RuntimeError, IndexError, ValueError):
        pass

    # Fallback to directory name
    return "local", repo_path.name


def _get_current_branch(repo_path: Path) -> str:
    """Get current branch name."""
    try:
        return _run_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
    except RuntimeError:
        return "unknown"


def _get_current_user(repo_path: Path) -> str:
    """Get git user name."""
    try:
        return _run_git(repo_path, "config", "user.name")
    except RuntimeError:
        return "local-user"


def build_mr_from_diff(
    repo_path: Path,
    base_ref: str = "main",
    include_uncommitted: bool = False,
) -> MergeRequest:
    """Build a MergeRequest from local git diff.

    Args:
        repo_path: Path to the local git repository
        base_ref: Base branch/ref to diff against (e.g., "main", "develop", "HEAD~3")
        include_uncommitted: If True, include staged + unstaged changes (diff against HEAD).
                           If False, diff current branch against base_ref.

    Returns:
        A MergeRequest object representing the local changes

    Raises:
        RuntimeError: If git commands fail
        FileNotFoundError: If repo_path doesn't exist or isn't a git repo
    """
    repo_path = Path(repo_path).resolve()
    if not (repo_path / ".git").exists():
        raise FileNotFoundError(f"Not a git repository: {repo_path}")

    owner, repo_name = _get_repo_info(repo_path)
    head_branch = _get_current_branch(repo_path)
    author = _get_current_user(repo_path)

    if include_uncommitted:
        # Diff working tree (staged + unstaged) against HEAD
        diff_ref = "HEAD"
        title = f"Uncommitted changes on {head_branch}"
    else:
        # Diff current branch against base_ref using merge-base for accurate comparison
        try:
            merge_base = _run_git(repo_path, "merge-base", base_ref, "HEAD")
            diff_ref = merge_base
        except RuntimeError:
            # If merge-base fails (e.g., no common ancestor), fall back to direct diff
            diff_ref = base_ref
        title = f"Changes on {head_branch} vs {base_ref}"

    # Get head and base SHAs
    head_sha = _run_git(repo_path, "rev-parse", "HEAD")
    try:
        base_sha = _run_git(repo_path, "rev-parse", diff_ref)
    except RuntimeError:
        base_sha = "0" * 40

    # Get changed files with status
    name_status_output = _run_git(repo_path, "diff", "--name-status", diff_ref)
    if not name_status_output:
        logger.info("No changes found")
        return MergeRequest(
            number=0,
            title=title,
            body=f"Local diff: {diff_ref}...HEAD",
            state="open",
            author=author,
            base_branch=base_ref if not include_uncommitted else head_branch,
            head_branch=head_branch,
            base_sha=base_sha,
            head_sha=head_sha,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            repo_owner=owner,
            repo_name=repo_name,
            changed_files=[],
            platform=GitPlatform.GITHUB,  # Doesn't matter for local review
        )

    # Parse each changed file and get its patch
    changed_files: list[ChangedFile] = []
    for line in name_status_output.split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        status_char = parts[0]
        file_status = _parse_status_char(status_char)

        if file_status == FileStatus.RENAMED and len(parts) >= 3:
            previous_filename = parts[1]
            filename = parts[2]
        else:
            filename = parts[1] if len(parts) > 1 else parts[0]
            previous_filename = None

        # Get the patch for this file
        try:
            patch = _run_git(repo_path, "diff", diff_ref, "--", filename)
        except RuntimeError:
            patch = None

        additions, deletions = _count_diff_lines(patch) if patch else (0, 0)

        changed_files.append(ChangedFile(
            filename=filename,
            status=file_status,
            additions=additions,
            deletions=deletions,
            patch=patch,
            previous_filename=previous_filename,
        ))

    logger.info(f"Built local MR with {len(changed_files)} changed files")

    return MergeRequest(
        number=0,
        title=title,
        body=f"Local diff: {diff_ref}...HEAD in {repo_path}",
        state="open",
        author=author,
        base_branch=base_ref if not include_uncommitted else head_branch,
        head_branch=head_branch,
        base_sha=base_sha,
        head_sha=head_sha,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        repo_owner=owner,
        repo_name=repo_name,
        changed_files=changed_files,
        platform=GitPlatform.GITHUB,  # Doesn't matter for local review
    )
