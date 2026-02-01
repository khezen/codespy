"""Security vulnerability analyzer module."""

from typing import Any

import dspy

from codespy.github.models import ChangedFile
from codespy.review.models import IssueCategory
from codespy.review.modules.base import BaseReviewModule
from codespy.review.signatures import SecurityAnalysis


class SecurityAnalyzer(BaseReviewModule):
    """Analyzes code for security vulnerabilities using DSPy."""

    category = IssueCategory.SECURITY

    def _create_predictor(self) -> dspy.Module:
        """Create the security analysis predictor with chain-of-thought reasoning."""
        return dspy.ChainOfThought(SecurityAnalysis)

    def _prepare_inputs(
        self,
        file: ChangedFile,
        context: str = "",
    ) -> dict[str, Any]:
        """Prepare inputs for security analysis.

        Args:
            file: The changed file to analyze
            context: Additional context from related files

        Returns:
            Dictionary of inputs for the predictor
        """
        return {
            "diff": file.patch or "",
            "full_content": file.content or "",
            "file_path": file.filename,
            "language": self.get_language(file),
            "context": context or "No additional context available.",
        }
