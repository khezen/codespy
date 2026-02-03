"""Documentation review module."""

import logging
from typing import Sequence

import dspy  # type: ignore[import-untyped]

from codespy.tools.github.models import ChangedFile
from codespy.agents.reviewer.models import Issue, IssueCategory
from codespy.agents.reviewer.modules.helpers import is_markdown_file, is_speculative, MIN_CONFIDENCE

logger = logging.getLogger(__name__)


class DocumentationReviewSignature(dspy.Signature):
    """Review markdown documentation files for accuracy and completeness.

    You are reviewing markdown documentation files (README.md, docs/*.md, etc.).
    This is NOT about code comments or docstrings - focus ONLY on:

    - Is the documentation accurate and up-to-date?
    - Are there factual errors or outdated information?
    - Is important information missing that users/developers need?
    - Are code examples correct and working?
    - Are links valid and pointing to the right resources?
    - Is the documentation clear and well-organized?

    Only report issues for MARKDOWN documentation files.
    Do NOT report issues about missing docstrings in code files.
    """

    diff: str = dspy.InputField(
        desc="The markdown documentation diff showing changes"
    )
    full_content: str = dspy.InputField(
        desc="The full markdown file content after changes"
    )
    filename: str = dspy.InputField(
        desc="Path to the markdown file being analyzed"
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="Documentation issues found. Empty list if documentation is adequate."
    )


class DocumentationReviewer(dspy.Module):
    """Reviews markdown documentation files for accuracy and completeness."""

    category = IssueCategory.DOCUMENTATION

    def __init__(self) -> None:
        """Initialize the documentation reviewer with chain-of-thought reasoning."""
        super().__init__()
        self.predictor = dspy.ChainOfThought(DocumentationReviewSignature)

    def forward(self, files: Sequence[ChangedFile]) -> list[Issue]:
        """Analyze markdown files for documentation issues.

        Only analyzes markdown files. Skips non-markdown files automatically.

        Args:
            files: The changed files to analyze

        Returns:
            List of documentation issues found across all markdown files
        """
        all_issues: list[Issue] = []
        
        for file in files:
            # Skip non-markdown files
            if not is_markdown_file(file.filename):
                continue

            if not file.patch:
                logger.debug(f"Skipping {file.filename}: no patch available")
                continue

            try:
                result = self.predictor(
                    diff=file.patch or "",
                    full_content=file.content or "",
                    filename=file.filename,
                    category=self.category,
                )
                issues = [
                    issue for issue in result.issues
                    if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
                ]
                all_issues.extend(issues)
                logger.debug(f"  Documentation in {file.filename}: {len(issues)} issues")
            except Exception as e:
                logger.error(f"Error analyzing {file.filename}: {e}")
        
        return all_issues
