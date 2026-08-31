"""Documentation review module — detects stale or wrong documentation."""

import asyncio
import logging

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.context_safe import ContextSafe
from codespy.agents.memory.hippocampus import ContextMemory, Hippocampus
from codespy.agents.memory.hippocampus.episode import find_latest_episode, submit_episode_save
from codespy.agents.reviewer.models import Issue, IssueCategory, ReviewContext, ScopeResult
from codespy.agents.reviewer.modules.doc_extractor import extract_documentation
from codespy.agents.reviewer.modules.helpers import (
    build_patches,
    issues_to_markdown,
    make_scope_relative,
    resolve_scope_root,
    restore_repo_paths,
)
from codespy.config import get_settings
from codespy.config_memory import get_memory_store

logger = logging.getLogger(__name__)


class DocReviewSignature(dspy.Signature):
    """Detect stale or wrong documentation caused by code changes.

    You are a busy Principal Engineer. Be extremely terse.
    Use imperative mood ("Update X", not "You should update X").

    You are given:
    1. Code patches showing what changed
    2. Current documentation content (README, .env.example, docs/, etc.)

    Your job: identify documentation that is now WRONG or MISSING because of the
    code changes. Cross-reference the patches against the documentation.

    CHECK FOR:

    HTTP/API CHANGES:
    - Content-Type, status codes, response body changes → check documented examples
    - New/removed endpoints → update docs

    FUNCTION/METHOD SIGNATURE CHANGES:
    - Parameters added/removed/renamed → check if docs reference old signatures
    - Return type changes → update examples

    CONFIGURATION & ENVIRONMENT VARIABLES:
    - New config fields → check README Configuration section or .env.example
    - Removed/renamed fields → find old names in documentation
    - Default value changes → verify docs reflect new defaults

    CLI COMMANDS & FLAGS:
    - New commands/flags → add to CLI reference
    - Removed/renamed flags → find old names in documentation

    DATA MODELS:
    - New/removed fields in structs → update API examples

    DO NOT report:
    - Missing documentation for internal/private functions
    - Style preferences in documentation
    - Documentation that is correct but could be "better"
    - Issues unrelated to the code changes in the patches

    OUTPUT RULES:
    - Set category to "documentation"
    - Set severity: "low" for missing/incomplete docs, "medium" for wrong/stale docs
    - filename: the documentation file that needs updating (use path from === path === headers)
    - description: ≤25 words, imperative tone ("Update X section", "Add Y to README")
    - Empty list if documentation is up to date. No approval text ("LGTM", "looks good")
    - No polite or conversational language
    """

    patches: str = dspy.InputField(
        desc="Code patches (diffs) showing what changed in this scope. "
        "Each patch is prefixed with the filename."
    )
    documentation: str = dspy.InputField(
        desc="Current documentation content for this scope. "
        "Each file is prefixed with === filename ===."
    )
    categories: list[IssueCategory] = dspy.InputField(
        desc="Allowed issue categories. Use only these values."
    )

    issues: list[Issue] = dspy.OutputField(
        desc="Documentation issues. Category must be 'documentation'. "
        "Each issue MUST include 'filename' set to the documentation file that needs "
        "updating (from the === filename === headers in the documentation input). "
        "Titles <10 words. Descriptions ≤25 words, imperative. Empty list if none."
    )


class DocReviewer(dspy.Module):
    """Detects stale or wrong documentation caused by code changes.

    Two-step approach per scope:
    1. Deterministic extraction (single tree scan + file reads, no LLM)
    2. DocReview (ChainOfThought, no tools): compares patches vs doc content
    """

    def __init__(self) -> None:
        """Initialize the doc reviewer."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    def _build_patches(self, scope: ScopeResult) -> str:
        """Build patches representation for review."""
        return build_patches(scope.changed_files)

    async def aforward(
        self,
        scopes: Sequence[ScopeResult],
        review_context: ReviewContext,
    ) -> tuple[list[Issue], ContextMemory | None]:
        """Analyze scopes for documentation issues.

        Args:
            scopes: List of identified scopes with their changed files
            review_context: ReviewContext containing PR identity and
                runtime pipeline metadata (repo_path, run_id, mr)

        Returns:
            Tuple of (list of issues, None) - memory is not returned
        """
        # Local bindings from review_context metadata
        repo_path = review_context.metadata.repo_path
        run_id = review_context.metadata.run_id
        pr = review_context.metadata.pr

        if not self._settings.is_signature_enabled("doc"):
            logger.debug("Skipping doc: disabled")
            return [], None
        changed_scopes = [s for s in scopes if s.has_changes and s.changed_files]
        if not changed_scopes:
            logger.info("No scopes with changes for doc review")
            return [], None
        all_issues: list[Issue] = []
        max_iters = self._settings.get_max_iters("doc")
        total_files = sum(len(s.changed_files) for s in changed_scopes)
        logger.info(f"Doc review for {len(changed_scopes)} scopes ({total_files} changed files)...")

        # Parallelize scope processing
        scope_tasks = [
            self._review_scope(scope, repo_path, run_id, pr, review_context, max_iters)
            for scope in changed_scopes
        ]
        results = await asyncio.gather(*scope_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Doc review failed for scope {changed_scopes[i].subroot}: {result}",
                    exc_info=result,
                )
            else:
                all_issues.extend(result)

        logger.info(f"Doc review found {len(all_issues)} issues")
        return all_issues, None

    async def _review_scope(
        self,
        scope: ScopeResult,
        repo_path: Path,
        run_id: str,
        pr: Any,
        review_context: ReviewContext,
        max_iters: int,
    ) -> list[Issue]:
        """Review documentation for a single scope. Returns list of issues (empty on error)."""
        scope_root = resolve_scope_root(repo_path, scope.subroot)
        if not scope_root.exists():
            logger.debug(
                f"  Scope directory does not exist (deleted/moved files): {scope.subroot}"
            )
            return []
        # Step 1: Extract documentation (deterministic — no LLM)
        logger.info(f"  Doc extraction: scope {scope.subroot}")
        try:
            documentation = extract_documentation(scope_root)
        except Exception as e:
            logger.error(f"Doc extraction failed for scope {scope.subroot}: {e}", exc_info=True)
            documentation = ""
        if not documentation.strip():
            logger.debug(f"  No documentation found in {scope.subroot}, skipping doc review")
            return []
        # Step 2: Review docs vs patches (ChainOfThought, no tools)
        scoped = make_scope_relative(scope)
        patches = self._build_patches(scoped)
        if not patches:
            logger.debug(f"  No patches in {scope.subroot}, skipping doc review")
            return []
        try:
            reviewer = ContextSafe(
                dspy.ChainOfThought(DocReviewSignature),
                DocReviewSignature,
                name="doc",
                max_iters=max_iters,
                max_llm_calls=self._settings.get_max_llm_calls("doc"),
                rlm_threshold=self._settings.get_rlm_threshold("chain_of_thought"),
            )
            logger.info(
                f"  Doc review: scope {scope.subroot} ({len(scope.changed_files)} files)"
            )
            mem: Hippocampus | None = None
            async with SignatureContext("doc", self._cost_tracker):
                # Load own prior "doc" episode for this scope
                scope_initial_memory: ContextMemory | None = None
                if self._settings.get_memory_enabled("doc"):
                    ep = find_latest_episode(
                        get_memory_store(self._settings),
                        scope.scope_path(),
                        task="doc",
                        exclude_run_id=run_id,
                    )
                    if ep is not None:
                        scope_initial_memory = ep.context_memory
                if self._settings.get_memory_enabled("doc"):
                    question = (
                        f"review documentation of {scope.repo}: {scope.subroot}: "
                        f"pull request {review_context.pr_context.pr_number} "
                        f"{review_context.pr_context.pr_title}: "
                        f"{review_context.pr_context.summary}"
                    )
                    topic_ids = [scope.topic(pr.repo_full_name).id] if pr else []
                    mem = Hippocampus(
                        reviewer,
                        budget=self._settings.get_memory_budget("doc"),
                        max_reflects=self._settings.get_memory_max_reflects("doc"),
                        question=question,
                        task_name="doc",
                        run_id=run_id,
                        initial_memory=scope_initial_memory,
                        topic_ids=topic_ids,
                    )
                    result = await mem.aforward(
                        patches=patches,
                        documentation=documentation,
                        categories=[IssueCategory.DOCUMENTATION],
                    )
                    issues = [
                        issue
                        for issue in (result.issues or [])
                        if issue.confidence >= self._settings.min_confidence
                    ]
                    # Fire-and-forget background episode save
                    _store = get_memory_store(self._settings)
                    _scope_path = scope.scope_path()
                    _artifacts = {"review": issues_to_markdown(issues)}
                    def _persist(m=mem, s=_store, p=_scope_path, a=_artifacts):
                        try:
                            m.end_episode(s, p, artifacts=a)
                        except Exception:
                            logger.warning("Background doc episode save failed", exc_info=True)
                    submit_episode_save(_persist, name="doc-episode-save")
                else:
                    result = await asyncio.to_thread(
                        reviewer,
                        patches=patches,
                        documentation=documentation,
                        categories=[IssueCategory.DOCUMENTATION],
                    )
                    issues = [
                        issue
                        for issue in (result.issues or [])
                        if issue.confidence >= self._settings.min_confidence
                    ]
            restore_repo_paths(issues, scope.subroot)
            logger.debug(f"  Scope {scope.subroot}: {len(issues)} doc issues")
            return issues
        except Exception as e:
            logger.error(f"Doc review failed for scope {scope.subroot}: {e}", exc_info=True)
            return []

    def forward(
        self,
        scopes: Sequence[ScopeResult],
        review_context: ReviewContext,
    ) -> tuple[list[Issue], ContextMemory | None]:
        """Analyze scopes for documentation issues (sync wrapper).

        Args:
            scopes: List of identified scopes with their changed files
            review_context: ReviewContext containing PR identity and
                runtime pipeline metadata (repo_path, run_id, mr)

        Returns:
            Tuple of (list of issues, None) - memory is not returned
        """
        return asyncio.run(self.aforward(scopes, review_context))
