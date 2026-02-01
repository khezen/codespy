"""Documentation review module."""

from typing import Any

import dspy

from codespy.github.models import ChangedFile
from codespy.review.models import IssueCategory
from codespy.review.modules.base import BaseReviewModule
from codespy.review.signatures import DocumentationReview


class DocumentationReviewer(BaseReviewModule):
    """Reviews code for documentation completeness using DSPy."""

    category = IssueCategory.DOCUMENTATION

    def _create_predictor(self) -> dspy.Module:
        """Create the documentation review predictor."""
        return dspy.ChainOfThought(DocumentationReview)

    def _prepare_inputs(
        self,
        file: ChangedFile,
        context: str = "",
    ) -> dict[str, Any]:
        """Prepare inputs for documentation review.

        Args:
            file: The changed file to analyze
            context: Not used for documentation review

        Returns:
            Dictionary of inputs for the predictor
        """
        return {
            "diff": file.patch or "",
            "full_content": file.content or "",
            "file_path": file.filename,
            "language": self.get_language(file),
        }