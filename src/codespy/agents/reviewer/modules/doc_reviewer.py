"""Documentation review module."""

import logging

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
    file_path: str = dspy.InputField(
        desc="Path to the markdown file being analyzed"
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="Documentation issues found. Use file_path for 'file' field and category for 'category' field. Empty list if documentation is adequate."
    )


class DocumentationReviewer(dspy.Module):
    """Reviews markdown documentation files for accuracy and completeness."""

    category = IssueCategory.DOCUMENTATION

    def __init__(self) -> None:
        """Initialize the documentation reviewer with chain-of-thought reasoning."""
        super().__init__()
        self.predictor = dspy.ChainOfThought(DocumentationReviewSignature)

    def forward(self, file: ChangedFile, context: str = "") -> list[Issue]:
        """Analyze a markdown file for documentation issues.

        Only analyzes markdown files. Returns empty list for non-markdown files.

        Args:
            file: The changed file to analyze
            context: Not used for documentation review

        Returns:
            List of documentation issues found
        """
        # Skip non-markdown files
        if not is_markdown_file(file.filename):
            return []

        if not file.patch:
            logger.debug(f"Skipping {file.filename}: no patch available")
            return []

        try:
            result = self.predictor(
                diff=file.patch or "",
                full_content=file.content or "",
                file_path=file.filename,
                category=self.category,
            )
            return [
                issue for issue in result.issues
                if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
            ]
        except Exception as e:
            logger.error(f"Error analyzing {file.filename}: {e}")
            return []