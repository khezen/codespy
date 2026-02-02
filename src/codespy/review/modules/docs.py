"""Documentation review module."""

from typing import Any

import dspy

from codespy.tools.github.models import ChangedFile
from codespy.review.models import Issue, IssueCategory
from codespy.review.modules.base import BaseReviewModule
from codespy.review.signatures import DocumentationReview

# Markdown file extensions to review
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx", ".rst", ".txt"}


class DocumentationReviewer(BaseReviewModule):
    """Reviews markdown documentation files for accuracy and completeness.

    Only analyzes markdown files (*.md, *.markdown, *.mdx, *.rst).
    Skips code files - use other modules for code documentation like docstrings.
    """

    category = IssueCategory.DOCUMENTATION

    def _create_predictor(self) -> dspy.Module:
        """Create the documentation review predictor."""
        return dspy.ChainOfThought(DocumentationReview)

    def _is_markdown_file(self, filename: str) -> bool:
        """Check if the file is a markdown documentation file."""
        import os
        _, ext = os.path.splitext(filename.lower())
        return ext in MARKDOWN_EXTENSIONS

    def analyze(self, file: ChangedFile, context: str = "") -> list[Issue]:
        """Analyze a file for documentation issues.

        Only analyzes markdown files. Returns empty list for non-markdown files.

        Args:
            file: The changed file to analyze
            context: Not used for documentation review

        Returns:
            List of documentation issues found
        """
        # Skip non-markdown files
        if not self._is_markdown_file(file.filename):
            return []

        # Call parent's analyze method for markdown files
        return super().analyze(file, context)

    def _prepare_inputs(
        self,
        file: ChangedFile,
        context: str = "",
    ) -> dict[str, Any]:
        """Prepare inputs for documentation review.

        Args:
            file: The changed markdown file to analyze
            context: Not used for documentation review

        Returns:
            Dictionary of inputs for the predictor
        """
        return {
            "diff": file.patch or "",
            "full_content": file.content or "",
            "file_path": file.filename,
        }
