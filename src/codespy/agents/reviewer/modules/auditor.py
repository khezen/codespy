"""Auditor module — assesses code quality and provides recommendation after reviews."""

import logging
from typing import TYPE_CHECKING, Sequence

import dspy

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.memory.hippocampus import Hippocampus
from codespy.agents.reviewer.models import Issue, ReviewContext
from codespy.config import get_settings
from codespy.config_memory import get_memory_store
from codespy.tools.git.models import ChangedFile

if TYPE_CHECKING:
    from codespy.agents.reviewer.models import ScopeResult

logger = logging.getLogger(__name__)


class AuditSignature(dspy.Signature):
    """Assess code quality and provide a recommendation for a merge request.

    You are a busy Principal Engineer. Be extremely terse. State facts only.
    Based on the summary, changed files, and issues found during review, provide:
    - An overall assessment of the code quality
    - A recommendation (approve, request changes, or needs discussion)

    No polite filler. No conversational language.
    """

    mr_title: str = dspy.InputField(desc="Title of the merge request")
    summary: str = dspy.InputField(desc="Summary of what this MR accomplishes")
    changed_files: list[ChangedFile] = dspy.InputField(
        desc="In-scope reviewable files with status and line counts"
    )
    all_issues: list[Issue] = dspy.InputField(
        desc="All issues found during review"
    )

    quality_assessment: str = dspy.OutputField(
        desc="Overall assessment of code quality"
    )
    recommendation: str = dspy.OutputField(
        desc="One of: APPROVE, REQUEST_CHANGES, or NEEDS_DISCUSSION with brief justification"
    )


class Auditor(dspy.Module):
    """Assesses code quality and recommends action after all reviews complete."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    def forward(
        self,
        review_context: ReviewContext,
        changed_files: Sequence[ChangedFile],
        all_issues: Sequence[Issue],
        run_id: str | None = None,
        scopes: list["ScopeResult"] | None = None,
    ) -> tuple[str, str]:
        """Assess quality and recommend action.

        Args:
            review_context: ReviewContext containing PR identity and inherited memory
            changed_files: In-scope reviewable files
            all_issues: All issues found during review
            run_id: Pipeline run identifier
            scopes: List of resolved scopes for per-scope episode persistence

        Returns:
            Tuple of (quality_assessment, recommendation)
        """
        if not self._settings.is_signature_enabled("audit"):
            logger.debug("Skipping audit: disabled")
            return (
                "Audit disabled.",
                "NEEDS_DISCUSSION" if all_issues else "APPROVE",
            )

        auditor = dspy.ChainOfThought(AuditSignature)
        logger.info("Running audit...")

        question = (
            f"final audit of {review_context.pr_context.repo_slug}: "
            f"pull request {review_context.pr_context.mr_number} {review_context.pr_context.mr_title}: {review_context.pr_context.summary}"
        )

        with SignatureContext("audit", self._cost_tracker):
            mem: Hippocampus | None = None
            if self._settings.get_memory_enabled("audit"):
                mem = Hippocampus(
                    auditor,
                    budget=self._settings.get_memory_budget("audit"),
                    max_reflects=self._settings.get_memory_max_reflects("audit"),
                    question=question,
                    task_name="audit",
                    run_id=run_id,
                    initial_memory=review_context.memory,
                )
                result = mem(
                    mr_title=review_context.pr_context.mr_title,
                    summary=review_context.pr_context.summary,
                    changed_files=list(changed_files),
                    all_issues=list(all_issues),
                )
                mem.end_episode(
                    get_memory_store(self._settings),
                    f"/{review_context.pr_context.repo_slug}/",
                    artifacts={
                        "audit": (
                            f"## Quality Assessment\n\n{result.quality_assessment}\n\n"
                            f"## Recommendation\n\n{result.recommendation}\n"
                        )
                    },
                )
            else:
                result = auditor(
                    mr_title=review_context.pr_context.mr_title,
                    summary=review_context.pr_context.summary,
                    changed_files=list(changed_files),
                    all_issues=list(all_issues),
                )

        # Persist episode at each scope location if memory is enabled and scopes are provided
        if mem is not None and scopes:
            store = get_memory_store(self._settings)
            for scope in scopes:
                path = mem.episode_file_path(scope.scope_path())
                mem.save_episode(store, path)

        return result.quality_assessment, result.recommendation
