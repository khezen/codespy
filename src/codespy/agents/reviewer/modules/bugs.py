"""Bug detection module."""

from typing import Any

import dspy

from codespy.tools.github.models import ChangedFile
from codespy.agents.reviewer.models import IssueCategory
from codespy.agents.reviewer.modules.base import BaseReviewModule
from codespy.agents.reviewer.signatures import BugDetection


class BugDetector(BaseReviewModule):
    """Detects potential bugs and logic errors using DSPy."""

    category = IssueCategory.BUG

    def _create_predictor(self) -> dspy.Module:
        """Create the bug detection predictor with chain-of-thought reasoning."""
        return dspy.ChainOfThought(BugDetection)

    def _prepare_inputs(
        self,
        file: ChangedFile,
        context: str = "",
    ) -> dict[str, Any]:
        """Prepare inputs for bug detection.

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