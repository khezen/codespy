"""Full-path review: present every file under a repo path as newly added."""

from __future__ import annotations

import logging
import os
import re
import stat
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from codespy.tools.git.models import (
    ChangedFile,
    FileStatus,
    GitPlatform,
    PullRequest,
    should_review_file,
)

logger = logging.getLogger(__name__)

# Report files committed by codespy itself: never reviewed as code
REPORT_DIR = ".codespy"
REPORT_BRANCH_PREFIX = "codespy/review/"
MAX_FILE_BYTES = 512 * 1024


def normalize_path(path: str) -> str:
    """Normalize a user-supplied repo path ('' / '/' / './' -> '.'). Rejects '..'."""
    p = PurePosixPath(path.strip().strip("/") or ".")
    if ".." in p.parts:
        raise ValueError(f"Path must stay inside the repository: {path}")
    s = str(p)
    return "." if s in ("", ".") else s


def path_slug(path: str) -> str:
    """Stable, branch/file-safe slug for a path ('.' -> 'root')."""
    if path == ".":
        return "root"
    slug = re.sub(r"[^A-Za-z0-9._-]", "", path.replace("/", "-")).strip(".-")
    return slug[:80] or "root"


def report_branch(path: str) -> str:
    return f"{REPORT_BRANCH_PREFIX}{path_slug(path)}"


def report_file(path: str) -> str:
    return f"{REPORT_DIR}/reviews/{path_slug(path)}.md"


def tree_url(host: str, owner: str, repo: str, sha: str, path: str, platform: GitPlatform) -> str:
    """Web URL of the reviewed tree at ``sha``."""
    base = f"https://{host}/{owner}/{repo}"
    sep = "/-/tree/" if platform == GitPlatform.GITLAB else "/tree/"
    return f"{base}{sep}{sha}" + ("" if path == "." else f"/{path}")


def blob_url(
    host: str,
    owner: str,
    repo: str,
    sha: str,
    filename: str,
    platform: GitPlatform | str,
    line_start: int | None = None,
    line_end: int | None = None,
) -> str:
    """Web URL of a file (optionally a line range) at ``sha``."""
    gitlab = str(platform) == GitPlatform.GITLAB.value
    url = f"https://{host}/{owner}/{repo}{'/-/blob/' if gitlab else '/blob/'}{sha}/{filename}"
    if line_start:
        url += f"#L{line_start}"
        if line_end and line_end != line_start:
            url += f"-{line_end}" if gitlab else f"-L{line_end}"
    return url


@dataclass
class TreeStats:
    """Counts collected while building the synthetic PR."""

    files_reviewed: int = 0
    bytes_reviewed: int = 0
    skipped: Counter[str] = field(default_factory=Counter)  # reason -> count
    files_by_dir: Counter[str] = field(default_factory=Counter)  # top-level dir -> files
    bytes_by_dir: Counter[str] = field(default_factory=Counter)  # top-level dir -> bytes

    @property
    def files_skipped(self) -> int:
        return sum(self.skipped.values())

    def breakdown(self) -> str:
        lines = ["Reviewable files per top-level directory:"]
        for d, n in self.files_by_dir.most_common():
            lines.append(f"  {d}: {n} files, {self.bytes_by_dir[d]:,} bytes")
        if self.skipped:
            lines.append("Skipped:")
            for reason, n in self.skipped.most_common():
                lines.append(f"  {reason}: {n}")
        return "\n".join(lines)


def _top_dir(rel: str) -> str:
    return rel.split("/", 1)[0] if "/" in rel else "."


def _list_tracked(repo_path: Path, path: str) -> list[str]:
    """Tracked paths under ``path`` that are present in the (sparse) working tree.

    Uses ``lexists`` so symlinks are listed (and later skipped) without being followed.
    """
    args = ["git", "-C", str(repo_path), "ls-files", "-z", "--"]
    args.append("." if path == "." else path)
    out = subprocess.run(args, capture_output=True, timeout=120, check=True)
    files = [f for f in out.stdout.decode("utf-8", "replace").split("\0") if f]
    return [f for f in files if os.path.lexists(repo_path / f)]


def _regular_file_inside(repo_root: Path, full: Path) -> os.stat_result | None:
    """``lstat`` of ``full`` when it is a regular file inside ``repo_root``, else None.

    The reviewed repository is untrusted: a tracked symlink (``x -> ~/.ssh/id_rsa``)
    must never be followed, or host files would be sent to the LLM and quoted in
    the report PR.
    """
    st = full.lstat()
    if not stat.S_ISREG(st.st_mode):
        return None
    if not full.resolve().is_relative_to(repo_root):
        return None  # a parent component escapes the repo
    return st


def _synthetic_patch(text: str) -> tuple[str, int]:
    """Unified diff hunk showing ``text`` as a newly added file."""
    lines = text.splitlines()
    body = "\n".join(f"+{line}" for line in lines)
    return f"@@ -0,0 +1,{len(lines)} @@\n{body}", len(lines)


def build_pr_from_tree(
    repo_path: Path,
    path: str,
    ref: str,
    sha: str,
    owner: str,
    repo: str,
    host: str,
    platform: GitPlatform,
    excluded_dirs: list[str],
) -> tuple[PullRequest, TreeStats]:
    """Build a synthetic PullRequest where every reviewable file under ``path`` is added.

    Raises:
        ValueError: If ``path`` has no tracked files.
    """
    stats = TreeStats()
    repo_root = repo_path.resolve()
    files = _list_tracked(repo_path, path)
    if not files:
        raise ValueError(f"No tracked files under '{path}' at {ref} ({sha[:8]})")

    changed: list[ChangedFile] = []
    for rel in sorted(files):
        if rel == REPORT_DIR or rel.startswith(REPORT_DIR + "/"):
            stats.skipped["codespy report"] += 1
            continue
        full = repo_path / rel
        st = _regular_file_inside(repo_root, full)
        if st is None:
            stats.skipped["symlink or not a regular file"] += 1
            continue
        if not should_review_file(
            ChangedFile(filename=rel, status=FileStatus.ADDED), excluded_dirs
        ):
            stats.skipped["excluded (binary, lock, minified, excluded dir)"] += 1
            continue
        size = st.st_size
        if size > MAX_FILE_BYTES:
            stats.skipped[f"larger than {MAX_FILE_BYTES // 1024} KB"] += 1
            continue
        fd = os.open(full, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as fh:
            raw = fh.read()
        if b"\0" in raw:
            stats.skipped["binary content"] += 1
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            stats.skipped["not UTF-8"] += 1
            continue
        if not text.strip():
            stats.skipped["empty"] += 1
            continue
        patch, n_lines = _synthetic_patch(text)
        changed.append(
            ChangedFile(filename=rel, status=FileStatus.ADDED, additions=n_lines, patch=patch)
        )
        stats.files_reviewed += 1
        stats.bytes_reviewed += size
        stats.files_by_dir[_top_dir(rel)] += 1
        stats.bytes_by_dir[_top_dir(rel)] += size

    shown = "the whole repository" if path == "." else f"`{path}`"
    body = (
        f"Full-path review of {shown} at `{ref}` (`{sha[:8]}`). There is no diff: every "
        f"file under the path is presented as newly added, and the code is reviewed as-is. "
        "Keep each independently built/deployed unit as its own scope."
    )
    now = datetime.now(UTC)
    pr = PullRequest(
        number=0,
        title=f"Full review of {'root' if path == '.' else path} @ {ref}",
        body=body,
        state="open",
        author="codespy",
        base_branch=ref,
        head_branch=report_branch(path),
        base_sha=sha,
        head_sha=sha,
        created_at=now,
        updated_at=now,
        repo_owner=owner,
        repo_name=repo,
        host=host,
        changed_files=changed,
        platform=platform,
        web_url=tree_url(host, owner, repo, sha, path, platform),
    )
    return pr, stats


def check_caps(stats: TreeStats, max_files: int, max_bytes: int) -> None:
    """Fail fast (before any LLM call) when the review is over a cap.

    Both caps apply to the files that will actually be reviewed, i.e. after
    every skip rule (excluded dirs, binary/lock/minified, symlinks, oversized,
    non-UTF-8, empty, ``.codespy/``). Untracked and .gitignore'd files are never
    listed in the first place (``git ls-files``).
    """
    problems = []
    if stats.files_reviewed > max_files:
        problems.append(
            f"{stats.files_reviewed} reviewable files > max_files={max_files} "
            "(--max-files / REVIEW_FULL_PATH_MAX_FILES)"
        )
    if stats.bytes_reviewed > max_bytes:
        problems.append(
            f"{stats.bytes_reviewed:,} bytes > max_bytes={max_bytes:,} (REVIEW_FULL_PATH_MAX_BYTES)"
        )
    if problems:
        raise ValueError(
            "Full-path review is over its cost cap: "
            + "; ".join(problems)
            + f" (counted after skipping {stats.files_skipped} ignored files)"
            + ". Narrow --path or raise the cap.\n"
            + stats.breakdown()
        )
