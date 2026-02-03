"""Domain expert module for codebase-aware review."""

import logging
from typing import Sequence

import dspy  # type: ignore[import-untyped]

from codespy.tools.github.models import ChangedFile
from codespy.agents.reviewer.models import Issue, IssueCategory
from codespy.agents.reviewer.modules.helpers import is_speculative, MIN_CONFIDENCE

logger = logging.getLogger(__name__)


class DomainExpertSignature(dspy.Signature):
    """Analyze code changes for cross-file consistency and architectural issues.

    You are a domain expert reviewing code changes for consistency and architecture.

    CRITICAL RULES:
    - ONLY report issues you can DIRECTLY SEE in the provided diffs
    - DO NOT speculate about code you cannot see
    - Focus on cross-file consistency within the provided changes
    - Quality over quantity: prefer 0 reports over 1 speculative report

    What to check:
    - Inconsistent patterns across the changed files
    - API contract changes that affect other changed files
    - Naming conventions that differ from other files in the PR
    - Architecture decisions that conflict with patterns in other changes
    - Missing updates in related files that are part of this PR

    DO NOT REPORT:
    - Issues about files not included in the changes
    - "Callers may need updating" without seeing those callers
    - Speculative breaking changes
    - Any issue where your evidence is not in the provided diffs
    """

    all_diffs: str = dspy.InputField(
        desc="Combined diffs of all changed files in the PR"
    )
    file_list: str = dspy.InputField(
        desc="List of all changed files with their status"
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="VERIFIED contextual issues based on cross-file analysis. Empty list if none."
    )


class DomainExpert(dspy.Module):
    """Analyzes code changes for cross-file consistency using DSPy."""

    category = IssueCategory.CONTEXT

    def __init__(self) -> None:
        """Initialize the domain expert with chain-of-thought reasoning."""
        super().__init__()
        self.predictor = dspy.ChainOfThought(DomainExpertSignature)

    def forward(self, files: Sequence[ChangedFile]) -> list[Issue]:
        """Analyze files for cross-file consistency and architectural issues.

        Args:
            files: The changed files to analyze

        Returns:
            List of contextual issues found across all files
        """
        # Filter to files with patches
        files_with_patches = [f for f in files if f.patch]
        
        if not files_with_patches:
            logger.debug("No files with patches to analyze for context")
            return []

        # Build combined diff and file list
        all_diffs_parts = []
        file_list_parts = []
        
        for file in files_with_patches:
            all_diffs_parts.append(f"=== {file.filename} ===\n{file.patch}")
            file_list_parts.append(f"- {file.filename} ({file.status.value}): +{file.additions}/-{file.deletions}")
        
        all_diffs = "\n\n".join(all_diffs_parts)
        file_list = "\n".join(file_list_parts)

        try:
            result = self.predictor(
                all_diffs=all_diffs,
                file_list=file_list,
                category=self.category,
            )
            issues = [
                issue for issue in result.issues
                if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
            ]
            logger.debug(f"  Domain expert: {len(issues)} cross-file issues")
            return issues
        except Exception as e:
            logger.error(f"Error in contextual analysis: {e}")
            return []
