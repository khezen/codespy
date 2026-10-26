"""Workflow-owned types."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from codespy.agents.review.models import Issue, IssueCategory, IssueSeverity
from codespy.agents.review.plan.models import ScopePlan

# Appended to a memory text cut to fit the review body size limit
TRUNCATED_MARKER = "…(truncated)"

# Maximum characters for the implementation plan in review body (leaves room for other sections)
MAX_PLAN_CHARS = 20000


class MemorySection(BaseModel):
    """One section of a recalled memory (e.g., 'Briefing: ...', 'Around this work')."""

    title: str = Field(description="Section title without markdown heading markers")
    text: str = Field(description="Section body text")


class RecalledMemory(BaseModel):
    """A recalled memory entry from Prefrontal."""

    task: str = Field(description="Task name that loaded the memory (e.g., 'scope', 'review')")
    sections: list[MemorySection] = Field(
        default_factory=list, description="Sections of the recalled memory"
    )
    nested: bool = Field(
        default=True,
        description="False = sections render directly under the memories section, without a per-recall block",
    )


class SignatureStatsResult(BaseModel):
    """Statistics for a single signature's execution during review."""

    name: str = Field(
        description="Signature name (e.g., code_review, doc, scope, supply_chain, summary, audit)"
    )
    cost: float = Field(default=0.0, description="Cost in USD for this signature")
    tokens: int = Field(default=0, description="Tokens used by this signature")
    call_count: int = Field(default=0, description="Number of LLM calls made by this signature")
    duration_seconds: float = Field(default=0.0, description="Execution time in seconds")
    input_tokens: int = Field(default=0, description="Input/prompt tokens used")
    output_tokens: int = Field(default=0, description="Output/completion tokens used")
    input_cost: float = Field(default=0.0, description="Cost for input tokens")
    output_cost: float = Field(default=0.0, description="Cost for output tokens")
    @property
    def is_memory(self) -> bool:
        """Check if this signature is a memory-related signature.

        Memory signatures have names starting with MEMORY_UNIT_PREFIX ("memory_").
        """
        from codespy.config_memory import MEMORY_UNIT_PREFIX

        return self.name.startswith(MEMORY_UNIT_PREFIX)

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


class FullPathInfo(BaseModel):
    """Information about a full-path review."""

    repo_url: str = Field(default="", description="Repository URL as given")
    path: str = Field(description="Reviewed path relative to repo root ('.' = whole repo)")
    ref: str = Field(description="Base branch")
    sha: str = Field(description="Resolved commit SHA")
    platform: str = Field(description="Git platform (github, gitlab)")
    host: str = Field(default="", description="Repo host (e.g. github.com)")
    owner: str = Field(description="Repository owner/namespace")
    repo: str = Field(description="Repository name")
    files_reviewed: int = Field(description="Number of files reviewed")
    files_skipped: int = Field(default=0, description="Number of files skipped")
    bytes_reviewed: int = Field(default=0, description="Total bytes reviewed")
    skipped: dict[str, int] = Field(default_factory=dict, description="Skip counts by reason")
    scopes: list[str] = Field(default_factory=list, description="Resolved scope subroots")

    @property
    def tree_url(self) -> str:
        from codespy.tools.git.full_path import tree_url
        from codespy.tools.git.models import GitPlatform

        platform = GitPlatform(self.platform)
        return tree_url(self.host, self.owner, self.repo, self.sha, self.path, platform)

    def blob_url(
        self, filename: str, line_start: int | None = None, line_end: int | None = None
    ) -> str:
        from codespy.tools.git.full_path import blob_url

        return blob_url(
            self.host, self.owner, self.repo, self.sha, filename, self.platform,
            line_start, line_end,
        )


class PathReviewConfig(BaseModel):
    """Configuration for reviewing a full path in a remote repository."""

    repo_url: str = Field(description="Repository URL (GitHub or GitLab)")
    path: str = Field(default=".", description="Path to review relative to repo root")
    ref: str | None = Field(
        default=None, description="Base branch (defaults to repo default branch)"
    )
    max_files: int | None = Field(
        default=None, description="Override review.full_path.max_files"
    )


# Union type for review configuration
ReviewConfig = RemoteReviewConfig | LocalReviewConfig | PathReviewConfig


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
    issues: list["Issue"] = Field(default_factory=list, description="All issues found during review")
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
    memories: list[RecalledMemory] = Field(
        default_factory=list,
        description="Prefrontal pre-call memory injected into agent inputs",
    )
    memory_pending: bool = Field(
        default=False,
        description="True when memory phase (saves, retain, consolidation) is still pending",
    )
    full_path: FullPathInfo | None = Field(
        default=None, description="Full-path review info (None for PR reviews)"
    )
    implementation_plan: list[ScopePlan] = Field(
        default_factory=list,
        description="Implementation plan generated from review findings",
    )

    @property
    def total_issues(self) -> int:
        """Get total number of issues."""
        return len(self.issues)

    @property
    def critical_issues(self) -> list["Issue"]:
        """Get all critical issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def security_issues(self) -> list["Issue"]:
        """Get all security issues."""
        return [i for i in self.issues if i.category == IssueCategory.SECURITY]

    @property
    def bug_issues(self) -> list["Issue"]:
        """Get all bug issues."""
        return [i for i in self.issues if i.category == IssueCategory.BUG]

    @property
    def documentation_issues(self) -> list["Issue"]:
        """Get all documentation issues."""
        return [i for i in self.issues if i.category == IssueCategory.DOCUMENTATION]

    @property
    def smell_issues(self) -> list["Issue"]:
        """Get all code smell issues."""
        return [i for i in self.issues if i.category == IssueCategory.SMELL]

    @property
    def seo_issues(self) -> list["Issue"]:
        """Get all SEO issues."""
        return [i for i in self.issues if i.category == IssueCategory.SEO]

    def issues_by_severity(self) -> dict["IssueSeverity", list["Issue"]]:
        """Group issues by severity."""
        result: dict[IssueSeverity, list[Issue]] = {s: [] for s in IssueSeverity}
        for issue in self.issues:
            result[issue.severity].append(issue)
        return result

    def issue_location_md(self, issue: Issue) -> str:
        """Issue location as markdown; a link to the reviewed SHA in path mode."""
        if self.full_path and issue.filename:
            url = self.full_path.blob_url(issue.filename, issue.line_start, issue.line_end)
            return f"[`{issue.location}`]({url})"
        return f"`{issue.location}`"

    def full_path_scope_lines(self, collapsible: bool = False) -> list[str]:
        """'Scope of this review' block for path mode ([] otherwise).

        ``collapsible`` wraps it in ``<details>`` instead of a ``##`` heading.
        """
        fp = self.full_path
        if not fp:
            return []
        shown = "whole repository" if fp.path == "." else f"`{fp.path}`"
        head = (
            ["<details>", "<summary>🗂️ Scope of this review</summary>", ""]
            if collapsible
            else ["## Scope of this review", ""]
        )
        lines = [
            *head,
            f"- **Path:** {shown} at `{fp.ref}` (`{fp.sha[:8]}`)",
            "- **Mode:** no diff; every file is reviewed as newly added code",
            f"- **Files reviewed:** {fp.files_reviewed} ({fp.bytes_reviewed:,} bytes)",
            f"- **Files skipped:** {fp.files_skipped}",
        ]
        for reason, n in sorted(fp.skipped.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {reason}: {n}")
        if fp.scopes:
            listed = ", ".join(f"`{s}`" for s in fp.scopes)
            lines.append(f"- **Scopes ({len(fp.scopes)}):** {listed}")
        lines.append("")
        if collapsible:
            lines.extend(["</details>", ""])
        return lines

    def to_markdown(self) -> str:
        """Format review results as Markdown."""
        fp = self.full_path
        if fp:
            # Full-path review mode: no PR, link the reviewed tree instead
            lines = [
                f"# Full Review: {fp.path} @ {fp.ref}",
                "",
                f"**Repository:** [{fp.owner}/{fp.repo}@{fp.sha[:8]}]({fp.tree_url})",
                f"**Reviewed at:** {self.reviewed_at.strftime('%Y-%m-%d %H:%M UTC')}",
                f"**Model:** {self.model_used}",
                "",
            ]
            lines.extend(self.full_path_scope_lines())
        else:
            # PR review mode
            lines = [
                f"# Code Review: {self.pr_title}",
                "",
                f"**PR:** [{self.repo}#{self.pr_number}]({self.pr_url})",
                f"**Reviewed at:** {self.reviewed_at.strftime('%Y-%m-%d %H:%M UTC')}",
                f"**Model:** {self.model_used}",
                "",
            ]

        # Memories section (collapsible) - placed before Summary
        lines.extend(self.memories_markdown_lines())

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
                f"- **SEO:** {len(self.seo_issues)}",
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
                    f"- **Total Cost:** ${self.total_cost:.4f}",
                ]
            )
            if self.memory_pending:
                lines.append("- **Memory:** pending (retain, consolidation, mental models)")
            lines.append("")

            # Per-signature breakdown
            lines.extend(self.cost_breakdown_markdown_lines("###"))

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
                                f"**Location:** {self.issue_location_md(issue)}",
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

        # Implementation Plan (uncapped for report file / stdout / MCP)
        lines.extend(self.implementation_plan_markdown_lines())

        return "\n".join(lines)

    def cost_breakdown_markdown_lines(self, heading: str = "###") -> list[str]:
        """Build cost breakdown tables for Review and Memory groups.

        Each non-empty group emits a table with columns:
        Signature | In Tokens | Out Tokens | In Cost | Out Cost | Calls | Duration

        Review signatures come first, then Memory. Each table has a subtotal line.
        Groups with no rows are omitted. If signature_stats is empty, returns [].

        Args:
            heading: Heading prefix (e.g., "###" or "####")

        Returns:
            Markdown lines for the cost breakdown section.
        """
        if not self.signature_stats:
            return []

        review_stats = [s for s in self.signature_stats if not s.is_memory]
        memory_stats = [s for s in self.signature_stats if s.is_memory]

        def table_lines(label: str, stats: list[SignatureStatsResult]) -> list[str]:
            if not stats:
                return []
            sorted_stats = sorted(stats, key=lambda x: x.cost, reverse=True)
            subtotal_cost = sum(s.cost for s in sorted_stats)
            subtotal_calls = sum(s.call_count for s in sorted_stats)
            lines = [
                f"{heading} {label}",
                "",
                "| Signature | In Tokens | Out Tokens | In Cost | Out Cost | Calls | Duration |",
                "|-----------|-----------|------------|---------|----------|-------|----------|",
            ]
            for s in sorted_stats:
                duration_str = f"{s.duration_seconds:.1f}s"
                lines.append(
                    f"| {s.name} | {s.input_tokens:,} | {s.output_tokens:,} | "
                    f"${s.input_cost:.4f} | ${s.output_cost:.4f} | {s.call_count} | {duration_str} |"
                )
            lines.extend(["", f"**Subtotal:** ${subtotal_cost:.4f} | **LLM Calls:** {subtotal_calls}", ""])
            return lines

        result: list[str] = []
        result.extend(table_lines("Review", review_stats))
        result.extend(table_lines("Memory", memory_stats))
        return result

    def memories_markdown_lines(
        self, summary: str = "memories", max_chars: int | None = None
    ) -> list[str]:
        """Build the collapsible memories section with nested per-recall and per-section details.

        Structure (flat run-level sections, nested scope recall):
          <details><summary>{summary}</summary>
            <blockquote>

              <!-- flat (nested=False): sections render directly -->
              <details><summary>{section_title}</summary>
                <blockquote>{body}</blockquote>
              </details>

              <!-- nested (nested=True): scope recall with own details wrapper -->
              <details><summary>{task}</summary>
                <blockquote>
                  <details><summary>{section_title}</summary>
                    <blockquote>{body}</blockquote>
                  </details>
                  ...
                </blockquote>
              </details>

            </blockquote>
          </details>

        GitReporter passes ``max_chars`` to stay within GitHub's 65,536 char
        review body limit: memory texts are truncated in order, other sections
        are never touched.

        Args:
            summary: Summary text of the outer collapsible section.
            max_chars: Character budget for the whole section (None = unlimited).

        Returns:
            Markdown lines, or ``[]`` when there is no memory or nothing fits.
        """
        if not self.memories:
            return []

        def size(block_lines: list[str]) -> int:
            # Lines are joined with "\n" by the caller
            return sum(len(line) + 1 for line in block_lines)

        def recall_head(task: str) -> list[str]:
            return ["<details>", f"<summary>{task}</summary>", "<blockquote>", ""]

        def recall_tail() -> list[str]:
            return ["</blockquote>", "</details>", ""]

        def section_block(title: str, body: str) -> list[str]:
            return [
                "<details>",
                f"<summary>{title}</summary>",
                "<blockquote>",
                "",
                body,
                "",
                "</blockquote>",
                "</details>",
                "",
            ]

        # Reserve room for outer head/tail and per-recall tails
        outer_head = ["<details>", f"<summary>{summary}</summary>", "<blockquote>", ""]
        outer_tail = ["</blockquote>", "</details>", ""]
        outer_overhead = size(outer_head) + size(outer_tail)

        lines: list[str] = []
        remaining = None if max_chars is None else max_chars - outer_overhead
        outer_stopped = False

        for memory in self.memories:
            if outer_stopped:
                break

            r_head = recall_head(memory.task)
            r_tail = recall_tail()
            recall_overhead = size(r_head) + size(r_tail)

            # For flat (nested=False), no recall head/tail overhead
            if memory.nested:
                # Budget check: need at least recall overhead + one section
                if remaining is not None and remaining < recall_overhead:
                    break

                if remaining is not None:
                    remaining -= recall_overhead
            else:
                # Flat: no recall overhead, sections go directly into lines
                recall_overhead = 0

            r_lines: list[str] = list(r_head) if memory.nested else []
            has_section = False
            recall_stopped = False

            for section in memory.sections:
                if recall_stopped:
                    break

                s_block = section_block(section.title, section.text)
                s_size = size(s_block)
                if remaining is None or s_size <= remaining:
                    r_lines.extend(s_block)
                    has_section = True
                    if remaining is not None:
                        remaining -= s_size
                else:
                    # Try to truncate the section body
                    overhead = size(section_block(section.title, "")) + len(TRUNCATED_MARKER)
                    if remaining is not None and remaining > overhead:
                        truncated_body = section.text[: remaining - overhead] + TRUNCATED_MARKER
                        r_lines.extend(section_block(section.title, truncated_body))
                        has_section = True
                        # Account for the truncated section size
                        remaining -= overhead + len(truncated_body)
                    # Close this recall - stop processing sections for this recall
                    recall_stopped = True
                    # Also stop the outer loop - no more recalls fit
                    outer_stopped = True

            if memory.nested:
                r_lines.extend(r_tail)

            if has_section:
                lines.extend(r_lines)
            elif memory.nested and remaining is not None:
                # No section rendered for this recall: give back its overhead
                remaining += recall_overhead

        if not lines:
            return []
        return outer_head + lines + outer_tail

    def implementation_plan_markdown_lines(
        self, max_chars: int | None = None, overflow_link: str | None = None
    ) -> list[str]:
        """Build the collapsible implementation plan section.

        Structure:
          <details>
          <summary>🛠️ Implementation Plan (N steps)</summary>
          <blockquote>
            [one scope] steps directly
            [several scopes] <details><summary>`subroot` (k steps)</summary>…</details>
          </blockquote>
          </details>

        Steps show title, files, change, verification and addressed issues.

        Args:
            max_chars: Character budget for the section (None = unlimited).
            overflow_link: URL to link to full plan when truncated.

        Returns:
            Markdown lines, or ``[]`` when there is no plan or nothing fits.
        """
        if not self.implementation_plan:
            return []
        plans = [sp for sp in self.implementation_plan if sp.steps]
        total_steps = sum(len(sp.steps) for sp in plans)
        if total_steps == 0:
            return []

        def size(block_lines: list[str]) -> int:
            return sum(len(line) + 1 for line in block_lines)

        def plural(n: int, word: str) -> str:
            return f"{n} {word}{'' if n == 1 else 's'}"

        def step_block(idx: int, step: Any) -> list[str]:
            block = [f"**{idx}. {step.title}**", ""]
            if step.files:
                block.extend([f"**Files:** {', '.join(f'`{f}`' for f in step.files)}", ""])
            if step.change:
                block.extend([step.change, ""])
            if step.verification:
                block.extend([f"**Verify:** {step.verification}", ""])
            if step.addresses:
                block.extend([f"**Addresses:** {'; '.join(step.addresses)}", ""])
            return block

        nested = len(plans) > 1
        summary = f"🛠️ Implementation Plan ({plural(total_steps, 'step')}"
        summary += f" in {plural(len(plans), 'scope')})" if nested else ")"
        outer_head = ["<details>", f"<summary>{summary}</summary>", "<blockquote>", ""]
        outer_tail = ["</blockquote>", "</details>", ""]

        def note_lines(omitted: int) -> list[str]:
            note = f"> {plural(omitted, 'more step')} omitted {TRUNCATED_MARKER}"
            if overflow_link:
                note += f" See the [full plan]({overflow_link})."
            return [note, ""]

        remaining: int | None = None
        if max_chars is not None:
            # Reserve the frame and the worst-case truncation note up front
            frame = size(outer_head) + size(outer_tail) + size(note_lines(total_steps))
            remaining = max_chars - frame

        body: list[str] = []
        shown = 0
        stopped = False
        for sp in plans:
            if stopped:
                break
            head: list[str] = []
            tail: list[str] = []
            if nested:
                head = [
                    "<details>",
                    f"<summary>`{sp.subroot}` ({plural(len(sp.steps), 'step')})</summary>",
                    "<blockquote>",
                    "",
                ]
                tail = ["</blockquote>", "</details>", ""]
            frame = size(head) + size(tail)
            scope_lines: list[str] = []
            for step in sp.steps:
                block = step_block(shown + 1, step)
                cost = size(block) + (frame if not scope_lines else 0)
                if remaining is not None and cost > remaining:
                    stopped = True
                    break
                scope_lines.extend(block)
                shown += 1
                if remaining is not None:
                    remaining -= cost
            if scope_lines:
                body.extend(head + scope_lines + tail)

        if shown == 0:
            return []
        if shown < total_steps:
            body.extend(note_lines(total_steps - shown))
        return outer_head + body + outer_tail

    def to_json_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return self.model_dump(mode="json")
