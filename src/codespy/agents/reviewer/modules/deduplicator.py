"""Issue deduplication module for removing redundant issues across reviewers."""

import logging
from typing import Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import ModuleContext, get_cost_tracker
from codespy.agents.reviewer.models import Issue, IssueSeverity

logger = logging.getLogger(__name__)

# Severity priority (higher number = higher priority)
SEVERITY_PRIORITY = {
    IssueSeverity.INFO: 1,
    IssueSeverity.LOW: 2,
    IssueSeverity.MEDIUM: 3,
    IssueSeverity.HIGH: 4,
    IssueSeverity.CRITICAL: 5,
}


class IssueDeduplicationSignature(dspy.Signature):
    """Identify and deduplicate similar issues from code review.

    You are analyzing a list of issues found by multiple code reviewers (security auditor,
    bug detector, documentation reviewer, domain expert). Different reviewers may identify
    the same underlying problem but categorize or describe it differently.

    Two issues are considered DUPLICATES if they:
    1. Reference the same file AND overlapping line ranges (or same general location)
    2. Describe the same underlying problem, even with different wording
    3. Would require the same fix to resolve

    For each group of duplicates found:
    - Return ONLY the issue with highest severity (critical > high > medium > low > info)
    - If same severity, keep the one with higher confidence score
    - Preserve important details: if one has a CWE ID or other reference and the other doesn't, keep the CWE ID or reference
    - Merge suggestions if they provide complementary guidance

    IMPORTANT: Return the deduplicated list of issues. Do NOT return duplicate groups,
    just the final list of unique issues that should remain.
    """

    issues: list[Issue] = dspy.InputField(
        desc="List of issues from all reviewers that may contain duplicates"
    )

    deduplicated_issues: list[Issue] = dspy.OutputField(
        desc="List of unique issues after removing duplicates, keeping highest severity/confidence"
    )


class IssueDeduplicator(dspy.Module):
    """Deduplicates issues found by multiple review modules.

    Uses semantic understanding to identify when different reviewers
    (security auditor, bug detector, etc.) have found the same underlying issue.
    Prioritizes keeping issues with higher severity, then higher confidence.
    """

    MODULE_NAME = "deduplicator"

    def __init__(self) -> None:
        """Initialize the issue deduplicator with chain-of-thought reasoning."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()

    def forward(self, issues: Sequence[Issue]) -> list[Issue]:
        """Deduplicate issues and return the unique list.

        Args:
            issues: List of issues from all review modules

        Returns:
            Deduplicated list of issues, prioritizing higher severity and confidence
        """
        if not issues:
            return []

        if len(issues) == 1:
            return list(issues)

        logger.info(f"Deduplicating {len(issues)} issues...")

        try:
            deduplicator_agent = dspy.ChainOfThought(IssueDeduplicationSignature)
            # Use ModuleContext to track costs and timing for this module
            with ModuleContext(self.MODULE_NAME, self._cost_tracker):
                result = deduplicator_agent(issues=list(issues))
                deduplicated = result.deduplicated_issues
            removed_count = len(issues) - len(deduplicated)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} duplicate issues")
            return deduplicated
        except Exception as e:
            logger.error(f"Error during deduplication: {e}")
            # Fall back to returning original issues
            return list(issues)
  