"""Domain expert module for codebase-aware review."""

import logging

import dspy  # type: ignore[import-untyped]

from codespy.tools.github.models import ChangedFile
from codespy.agents.reviewer.models import Issue, IssueCategory
from codespy.agents.reviewer.modules.helpers import is_speculative, MIN_CONFIDENCE

logger = logging.getLogger(__name__)


class DomainExpertSignature(dspy.Signature):
    """Analyze code changes using VERIFIED caller information and related files.

    CRITICAL RULES - READ CAREFULLY:
    - The related_files input includes VERIFIED caller information from codebase search
    - If "Verified Callers of Changed Functions" section exists, USE IT to report concrete issues
    - ONLY report issues where you can cite SPECIFIC file:line references
    - NEVER say "cannot be verified" - if you can't verify, don't report
    - NEVER speculate about callers that might exist - only report about callers you can see

    USING VERIFIED CALLER INFORMATION:
    - Look for "=== Verified Callers of Changed Functions ===" section in related_files
    - This shows REAL callers found via code search - use these for your analysis
    - Report issues like: "Caller at api/handler.go:45 needs to be updated..."
    - Include the actual caller file and line in your issue description

    What to check:
    - Breaking changes where verified callers need updating (cite the file:line)
    - Signature changes that affect callers shown in the verified list
    - Renamed/removed functions that have callers in the verified list
    - Pattern inconsistencies you can SHOW in related_files content

    DO NOT REPORT:
    - "Callers may need updating" without citing specific callers from verified list
    - "Unknown callers might be affected" - only report what you can see
    - "This could break X" without showing X in the context
    - Any issue where your evidence is speculative
    """

    diff: str = dspy.InputField(
        desc="The code diff showing changes"
    )
    filename: str = dspy.InputField(
        desc="Path to the file being analyzed"
    )
    related_files: str = dspy.InputField(
        desc="Content of related files AND verified callers - includes 'Verified Callers of Changed Functions' section with file:line references when available"
    )
    repo_structure: str = dspy.InputField(
        desc="Overview of the repository structure"
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="VERIFIED contextual issues. Only report issues with concrete evidence from verified callers. Empty list if none."
    )


class DomainExpert(dspy.Module):
    """Analyzes code changes in the context of the broader codebase using DSPy."""

    category = IssueCategory.CONTEXT

    def __init__(self) -> None:
        """Initialize the domain expert with chain-of-thought reasoning."""
        super().__init__()
        self.predictor = dspy.ChainOfThought(DomainExpertSignature)

    def forward(self, file: ChangedFile, context: str = "", repo_structure: str = "") -> list[Issue]:
        """Analyze a file with codebase context and return issues.

        Args:
            file: The changed file to analyze
            context: Related files content as formatted string
            repo_structure: Repository structure overview

        Returns:
            List of contextual issues found
        """
        if not file.patch:
            logger.debug(f"Skipping {file.filename}: no patch available")
            return []

        try:
            result = self.predictor(
                diff=file.patch or "",
                filename=file.filename,
                related_files=context or "No related files available.",
                repo_structure=repo_structure or "Repository structure not available.",
                category=self.category,
            )
            return [
                issue for issue in result.issues
                if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
            ]
        except Exception as e:
            logger.error(f"Error in contextual analysis of {file.filename}: {e}")
            return []