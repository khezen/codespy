"""Documentation review module — detects stale or wrong documentation."""

import asyncio
import logging
from pathlib import Path
from typing import Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.memory.hippocampus import Hypocampus
from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.agents.reviewer.modules.doc_extractor import extract_documentation
from codespy.agents.reviewer.modules.helpers import (
    MIN_CONFIDENCE,
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
        """Build compact patches representation."""
        parts: list[str] = []
        for f in scope.changed_files:
            if f.patch:
                parts.append(f"--- {f.filename} ---\n{f.patch}")
        return "\n\n".join(parts)

    async def aforward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for documentation issues.

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository

        Returns:
            List of documentation issues found across all scopes
        """
        if not self._settings.is_signature_enabled("doc"):
            logger.debug("Skipping doc: disabled")
            return []
        changed_scopes = [s for s in scopes if s.has_changes and s.changed_files]
        if not changed_scopes:
            logger.info("No scopes with changes for doc review")
            return []
        all_issues: list[Issue] = []
        total_files = sum(len(s.changed_files) for s in changed_scopes)
        logger.info(
            f"Doc review for {len(changed_scopes)} scopes "
            f"({total_files} changed files)..."
        )
        for scope in changed_scopes:
            scope_root = resolve_scope_root(repo_path, scope.subroot)
            # Step 1: Extract documentation (deterministic — no LLM)
            logger.info(f"  Doc extraction: scope {scope.subroot}")
            try:
                documentation = extract_documentation(scope_root)
            except Exception as e:
                logger.error(f"Doc extraction failed for scope {scope.subroot}: {e}")
                documentation = ""
            if not documentation.strip():
                logger.debug(
                    f"  No documentation found in {scope.subroot}, skipping doc review"
                )
                continue
            # Step 2: Review docs vs patches (ChainOfThought, no tools)
            scoped = make_scope_relative(scope)
            patches = self._build_patches(scoped)
            if not patches:
                logger.debug(f"  No patches in {scope.subroot}, skipping doc review")
                continue
            try:
                reviewer = dspy.ChainOfThought(DocReviewSignature)
                logger.info(
                    f"  Doc review: scope {scope.subroot} "
                    f"({len(scope.changed_files)} files)"
                )
                async with SignatureContext("doc", self._cost_tracker):
                    if self._settings.get_memory_enabled("doc"):
                        mem = Hypocampus(
                            reviewer,
                            token_budget=self._settings.get_memory_token_budget("doc"),
                            max_trajectory_tokens=self._settings.get_memory_max_trajectory_tokens(
                                "doc"
                            ),
                            max_reflects=self._settings.get_memory_max_reflects("doc"),
                        )
                        result = await mem.aforward(
                            patches=patches,
                            documentation=documentation,
                            categories=[IssueCategory.DOCUMENTATION],
                        )
                        await mem.aend_episode(get_memory_store(self._settings), scope.scope_path())
                    else:
                        result = await asyncio.to_thread(
                            reviewer,
                            patches=patches,
                            documentation=documentation,
                            categories=[IssueCategory.DOCUMENTATION],
                        )
                issues = [
                    issue for issue in (result.issues or [])
                    if issue.confidence >= MIN_CONFIDENCE
                ]
                restore_repo_paths(issues, scope.subroot)
                all_issues.extend(issues)
                logger.debug(
                    f"  Scope {scope.subroot}: {len(issues)} doc issues"
                )
            except Exception as e:
                logger.error(f"Doc review failed for scope {scope.subroot}: {e}")

        logger.info(f"Doc review found {len(all_issues)} issues")
        return all_issues

    def forward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for documentation issues (sync wrapper).

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository

        Returns:
            List of documentation issues found across all scopes
        """
        return asyncio.run(self.aforward(scopes, repo_path))
