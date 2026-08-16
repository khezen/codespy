"""Auditor module — assesses code quality and provides recommendation after reviews."""

import logging
from typing import TYPE_CHECKING, Sequence

import dspy
import litellm

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.memory.hippocampus import ContextMemory, Hippocampus
from codespy.agents.reviewer.models import Issue, ReviewContext
from codespy.config import get_settings
from codespy.config_memory import get_memory_store
from codespy.tools.git.models import ChangedFile

if TYPE_CHECKING:
    from codespy.agents.reviewer.models import ScopeResult

logger = logging.getLogger(__name__)


def _strip_patches(files: Sequence[ChangedFile]) -> list[ChangedFile]:
    """Remove patch content from files to reduce token usage."""
    return [f.model_copy(update={"patch": None}) for f in files]


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

    def _would_overflow_context(
        self,
        mr_title: str,
        summary: str,
        changed_files: list[ChangedFile],
        all_issues: list[Issue],
        context_memory: str | None = None,
    ) -> bool:
        """Estimate whether the input would overflow the model's context window.

        Uses litellm.token_counter for estimation with a safety margin to
        account for DSPy formatting overhead (system prompt, field descriptions,
        ChainOfThought instructions).
        """
        SAFETY_MARGIN = 4096  # DSPy formatting overhead + token counting imprecision

        try:
            llm_config = self._settings.get_llm_config("audit")
            model = llm_config.model
            max_tokens = llm_config.max_tokens or self._settings.default_max_tokens

            # Get model limits
            info = litellm.get_model_info(model)
            max_input = info.get("max_input_tokens") or 0
            max_output = info.get("max_output_tokens") or 0
            if not max_input:
                return False  # Unknown model, can't estimate

            # Use max_input as context window proxy (conservative).
            # For shared-budget models the true window is slightly larger,
            # but using max_input ensures we never overshoot.
            context_window = max_input

            # Estimate input tokens from a rough serialization
            input_text = f"{mr_title}\n{summary}\n{changed_files}\n{all_issues}"
            if context_memory:
                input_text += f"\n{context_memory}"
            estimated_input = litellm.token_counter(model=model, text=input_text)

            return (estimated_input + max_tokens + SAFETY_MARGIN) > context_window
        except Exception:
            return False  # Estimation failed; proceed with full input

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
            f"pull request {review_context.pr_context.mr_number} "
            f"{review_context.pr_context.mr_title}: {review_context.pr_context.summary}"
        )

        if self._settings.get_memory_enabled("audit"):
            mem = Hippocampus(
                auditor,
                budget=self._settings.get_memory_budget("audit"),
                max_reflects=self._settings.get_memory_max_reflects("audit"),
                question=question,
                task_name="audit",
                run_id=run_id,
                initial_memory=review_context.memory,
                topic_ids=topic_ids,
            )
            result = mem(
                mr_title=review_context.pr_context.mr_title,
                summary=review_context.pr_context.summary,
                changed_files=audit_files,
                all_issues=all_issues,
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
            # Persist episode at each scope location if scopes are provided
            if scopes:
                store = get_memory_store(self._settings)
                for scope in scopes:
                    path = mem.episode_file_path(scope.scope_path())
                    mem.save_episode(store, path)
        else:
            result = auditor(
                mr_title=review_context.pr_context.mr_title,
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
            review_context: ReviewContext containing PR identity and inherited memory
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

        auditor = dspy.ChainOfThought(AuditSignature)
        logger.info("Running audit...")

        # Pre-flight: check if full input would overflow
        context_memory_str = (
            review_context.memory.render() if review_context.memory else None
        )
        if self._would_overflow_context(
            mr_title=review_context.pr_context.mr_title,
            summary=review_context.pr_context.summary,
            changed_files=list(changed_files),
            all_issues=list(all_issues),
            context_memory=context_memory_str,
        ):
            logger.info(
                "Pre-flight: stripping patches from changed_files to fit context window"
            )
            audit_files = _strip_patches(changed_files)
        else:
            audit_files = list(changed_files)

        with SignatureContext("audit", self._cost_tracker):
            try:
                result = self._call_auditor(
                    auditor,
                    review_context,
                    audit_files,
                    list(all_issues),
                    run_id,
                    scopes,
                    topic_ids,
                )
            except dspy.ContextWindowExceededError:
                if audit_files is not _strip_patches(changed_files):
                    # Pre-flight didn't strip — try again without patches
                    logger.warning(
                        "Context window exceeded despite pre-flight check; "
                        "retrying without patches"
                    )
                    audit_files = _strip_patches(changed_files)
                    result = self._call_auditor(
                        auditor,
                        review_context,
                        audit_files,
                        list(all_issues),
                        run_id,
                        scopes,
                        topic_ids,
                    )
                else:
                    # Already stripped patches and still overflowing — re-raise
                    raise

        return result.quality_assessment, result.recommendation
