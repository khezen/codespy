"""Reporter that opens a fix PR/MR with the implementation agent's changes.

PR/MR review only: the fix PR targets the reviewed PR's head branch, starts
from the reviewed head commit, and references the reviewed PR. Re-runs reset
``codespy/fix/<n>`` to one fresh commit and update the open fix PR.
"""

import logging
from typing import TYPE_CHECKING

from codespy import __version__
from codespy.tools.git.base import SubmittedPullRequest
from codespy.tools.git.client import get_client
from codespy.tools.git.conversation import fix_marker
from codespy.tools.git.full_path import fix_branch
from codespy.tools.git.models import GitPlatform, PullRequest
from codespy.workflows.review.models import MAX_PLAN_CHARS, ReviewResult

if TYPE_CHECKING:
    from codespy.config import Settings

logger = logging.getLogger(__name__)


class FixPullRequestReporter:
    """Opens or updates ``codespy/fix/<n>`` against the reviewed PR's branch."""

    # Below GitHub's 65,536 char body limit
    MAX_BODY_CHARS = 60000

    def __init__(self, url: str, settings: "Settings | None" = None) -> None:
        self.url = url
        self.client = get_client(url, settings)

    @staticmethod
    def ref(pr: PullRequest) -> str:
        """``#n`` (GitHub) or ``!n`` (GitLab)."""
        return f"!{pr.number}" if pr.platform == GitPlatform.GITLAB else f"#{pr.number}"

    def title(self, pr: PullRequest) -> str:
        return f"codespy: fixes for {self.ref(pr)}"

    def build_body(self, result: ReviewResult, pr: PullRequest) -> str:
        """Description: marker, reference to the reviewed PR, steps, files, plan."""
        impl = result.implementation
        if impl is None:
            raise ValueError("implementation required")
        titles = {
            s.number: s.title
            for sp in (result.implementation_plan.scopes if result.implementation_plan else [])
            for s in sp.steps
        }
        lines = [
            f"# 🤖 Code[Spy](https://github.com/khezen/codespy) Fixes <sub>v{__version__}</sub>",
            fix_marker(result.run_id or ""),
            "",
            f"Implements the codespy implementation plan for {pr.url} ({self.ref(pr)}), "
            f"at `{pr.head_sha[:8]}`. Merge this into `{pr.head_branch}` to apply the fixes.",
            "",
            "> ⚠️ Written by an agent that cannot run code: nothing here was built or tested. "
            "Review it and let CI check it.",
            "",
        ]
        if impl.summary:
            lines.extend(["## Summary", "", impl.summary, ""])
        if impl.completed_steps:
            lines.extend(["## Completed steps", ""])
            lines.extend(f"- {n}. {titles.get(n, '')}".rstrip() for n in impl.completed_steps)
            lines.append("")
        if impl.skipped_steps:
            lines.extend(["## Skipped steps", ""])
            lines.extend(
                f"- {s.number}. {titles.get(s.number, '')} — {s.reason}".replace(".  —", ". —")
                for s in impl.skipped_steps
            )
            lines.append("")
        lines.extend(["## Changed files", ""])
        lines.extend(
            f"- `{c.path}`" + (" (deleted)" if c.content is None else "") for c in impl.changes
        )
        lines.append("")
        used = sum(len(line) + 1 for line in lines)
        lines.extend(
            result.implementation_plan_markdown_lines(
                max_chars=max(0, min(MAX_PLAN_CHARS, self.MAX_BODY_CHARS - used))
            )
        )
        return "\n".join(lines)

    def publish(self, result: ReviewResult, pr: PullRequest) -> SubmittedPullRequest | None:
        """Open/update the fix PR and set ``result.implementation.pull_request_url``.

        Returns None when there is nothing to publish. Raises on platform errors.
        """
        impl = result.implementation
        if impl is None or not impl.changes:
            return None
        handle = self.client.upsert_branch_pull_request(
            owner=pr.repo_owner,
            repo=pr.repo_name,
            base_branch=pr.head_branch,
            base_sha=pr.head_sha,
            head_branch=fix_branch(pr.number),
            files=impl.files_payload(),
            executable=impl.executable_paths(),
            commit_message=f"codespy: fixes for {self.ref(pr)} at {pr.head_sha[:8]}",
            title=self.title(pr),
            body=self.build_body(result, pr),
            update_comment=(
                f"Updated by codespy run `{(result.run_id or '')[:8]}` at `{pr.head_sha[:8]}`: "
                f"{len(impl.changes)} file(s) changed. The branch was reset to one fresh commit."
            ),
        )
        impl.pull_request_url = handle.url
        action = "Opened" if handle.created else "Updated"
        logger.info(f"{action} fix {self.client.platform_name} PR: {handle.url}")
        return handle
