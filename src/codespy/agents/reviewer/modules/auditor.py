"""Auditor module — assesses code quality and provides recommendation after reviews."""

import logging
from codespy.agents.memory.hippocampus.episode import submit_episode_save
from collections.abc import Sequence
from typing import TYPE_CHECKING

import dspy

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.context_safe import ContextSafe
from codespy.agents.memory.hippocampus import ContextMemory, Hippocampus
from codespy.agents.reviewer.models import Issue, ReviewContext
from codespy.agents.reviewer.modules.scope_resolver import _deepest_common_folder
from codespy.config import get_settings
from codespy.config_memory import get_memory_store
from codespy.tools.git.models import ChangedFile

if TYPE_CHECKING:
    from codespy.agents.reviewer.models import ScopeResult

logger = logging.getLogger(__name__)


class AuditSignature(dspy.Signature):
    """Assess code quality and provide a recommendation for a pull request.

    You are a busy Principal Engineer. Be extremely terse. State facts only.
    Based on the summary, changed files, and issues found during review, provide:
    - An overall assessment of the code quality
    - A recommendation (approve, request changes, or needs discussion)

    No polite filler. No conversational language.
    """

    pr_title: str = dspy.InputField(desc="Title of the pull request")
    summary: str = dspy.InputField(desc="Summary of what this PR accomplishes")
    changed_files: list[ChangedFile] = dspy.InputField(
        desc="In-scope reviewable files with status and line counts"
    )
    all_issues: list[Issue] = dspy.InputField(desc="All issues found during review")

    quality_assessment: str = dspy.OutputField(desc="Overall assessment of code quality")
    recommendation: str = dspy.OutputField(
        desc="One of: APPROVE, REQUEST_CHANGES, or NEEDS_DISCUSSION with brief justification"
    )


class Auditor(dspy.Module):
    """Assesses code quality and recommends action after all reviews complete."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    def _call_auditor(
        self,
        auditor: dspy.ChainOfThought,
        review_context: ReviewContext,
        audit_files: list[ChangedFile],
        all_issues: list[Issue],
        run_id: str | None,
        scopes: list["ScopeResult"] | None,
        topic_ids: list[str] | None,
    ) -> dspy.Prediction:
        """Execute the auditor predictor (with or without Hippocampus memory)."""
        question = (
            f"final audit of {review_context.pr_context.repo_slug}: "
            f"pull request {review_context.pr_context.pr_number} "
            f"{review_context.pr_context.pr_title}: {review_context.pr_context.summary}"
        )

        # Load prior "audit" episodes per scope
        initial_memory: ContextMemory | None = None
        if self._settings.get_memory_enabled("audit") and scopes:
            from codespy.agents.memory.hippocampus.episode import find_latest_episode
            store = get_memory_store(self._settings)
            per_scope_memories: list[ContextMemory] = []
            for scope in scopes:
                ep = find_latest_episode(store, scope.scope_path(), task="audit", exclude_run_id=run_id)
                if ep is not None:
                    per_scope_memories.append(ep.context_memory)
            if per_scope_memories:
                initial_memory = ContextMemory.merge(*per_scope_memories)
                logger.info("Merged %d prior audit episode(s) into auditor memory", len(per_scope_memories))

        if self._settings.get_memory_enabled("audit"):
            mem = Hippocampus(
                auditor,
                budget=self._settings.get_memory_budget("audit"),
                max_reflects=self._settings.get_memory_max_reflects("audit"),
                question=question,
                task_name="audit",
                run_id=run_id,
                initial_memory=initial_memory,
                topic_ids=topic_ids,
            )
            result = mem(
                pr_title=review_context.pr_context.pr_title,
                summary=review_context.pr_context.summary,
                changed_files=audit_files,
                all_issues=all_issues,
            )
            # Fire-and-forget background episode save
            _store = get_memory_store(self._settings)
            _common_dir = (
                _deepest_common_folder(scopes, review_context.pr_context.repo_slug)
                if scopes else f"/{review_context.pr_context.repo_slug}/"
            )
            _artifacts = {
                "audit": (
                    f"## Quality Assessment\n\n{result.quality_assessment}\n\n"
                    f"## Recommendation\n\n{result.recommendation}\n"
                )
            }
            _scopes = scopes
            def _persist(m=mem, s=_store, d=_common_dir, a=_artifacts, sc=_scopes):
                try:
                    m.end_episode(s, d, artifacts=a)
                    if sc:
                        for scope in sc:
                            m.save_episode(s, m.episode_file_path(scope.scope_path()))
                except Exception:
                    logger.warning("Background audit episode save failed", exc_info=True)
            submit_episode_save(_persist, name="audit-episode-save")
        else:
            result = auditor(
                pr_title=review_context.pr_context.pr_title,
                summary=review_context.pr_context.summary,
                changed_files=audit_files,
                all_issues=all_issues,
            )

        return result

    def forward(
        self,
        review_context: ReviewContext,
        changed_files: Sequence[ChangedFile],
        all_issues: Sequence[Issue],
        run_id: str | None = None,
        scopes: list["ScopeResult"] | None = None,
        topic_ids: list[str] | None = None,
    ) -> tuple[str, str]:
        """Assess quality and recommend action.

        Args:
            review_context: ReviewContext containing PR identity (memory loaded per-scope from prior audit episodes)
            changed_files: In-scope reviewable files
            all_issues: All issues found during review
            run_id: Pipeline run identifier
            scopes: List of resolved scopes for per-scope episode persistence
            topic_ids: Optional list of topic IDs for auto-tagging

        Returns:
            Tuple of (quality_assessment, recommendation)
        """
        if not self._settings.is_signature_enabled("audit"):
            logger.debug("Skipping audit: disabled")
            return (
                "Audit disabled.",
                "NEEDS_DISCUSSION" if all_issues else "APPROVE",
            )

        auditor = ContextSafe(
            dspy.ChainOfThought(AuditSignature),
            AuditSignature,
            name="audit",
            max_llm_calls=self._settings.get_max_llm_calls("audit"),
            rlm_threshold=self._settings.get_rlm_threshold("chain_of_thought"),
        )
        logger.info("Running audit...")

        with SignatureContext("audit", self._cost_tracker):
            result = self._call_auditor(
                auditor,
                review_context,
                list(changed_files),
                list(all_issues),
                run_id,
                scopes,
                topic_ids,
            )

        return result.quality_assessment, result.recommendation
