"""Contextual analysis module for codebase-aware review."""

import logging
from typing import Any

import dspy

from codespy.tools.github.models import ChangedFile, ReviewContext
from codespy.agents.reviewer.models import Issue, IssueCategory
from codespy.agents.reviewer.modules.base import BaseReviewModule
from codespy.agents.reviewer.signatures import ContextualAnalysis

logger = logging.getLogger(__name__)


class ContextAnalyzer(BaseReviewModule):
    """Analyzes code changes in the context of the broader codebase using DSPy."""

    category = IssueCategory.CONTEXT

    def _create_predictor(self) -> dspy.Module:
        """Create the contextual analysis predictor."""
        return dspy.ChainOfThought(ContextualAnalysis)

    def _prepare_inputs(
        self,
        file: ChangedFile,
        context: str = "",
    ) -> dict[str, Any]:
        """Prepare inputs for contextual analysis.

        Args:
            file: The changed file to analyze
            context: Related files content as formatted string

        Returns:
            Dictionary of inputs for the predictor
        """
        return {
            "diff": file.patch or "",
            "file_path": file.filename,
            "related_files": context or "No related files available.",
            "repo_structure": "Repository structure not available.",
        }

    def analyze_with_context(
        self,
        file: ChangedFile,
        review_context: ReviewContext,
    ) -> list[Issue]:
        """Analyze a file with full review context.

        Args:
            file: The changed file to analyze
            review_context: Full review context including related files

        Returns:
            List of contextual issues found
        """
        if not file.patch:
            logger.debug(f"Skipping {file.filename}: no patch available")
            return []

        # Build context string from related files
        context_parts = []
        for filename, content in review_context.related_files.items():
            # Truncate large files
            if len(content) > 5000:
                content = content[:5000] + "\n... (truncated)"
            context_parts.append(f"=== {filename} ===\n{content}")

        # Add verified caller information if available
        callers_str = review_context.get_callers_for_file(file.filename)
        if callers_str and "No callers found" not in callers_str:
            context_parts.append(callers_str)
            logger.debug(f"Including caller information for {file.filename}")

        related_files_str = "\n\n".join(context_parts) if context_parts else "No related files."
        repo_structure = review_context.repository_structure or "Structure not available."

        try:
            inputs = {
                "diff": file.patch or "",
                "file_path": file.filename,
                "related_files": related_files_str,
                "repo_structure": repo_structure,
            }
            result = self._predictor(**inputs)
            return self._parse_issues(result, file)
        except Exception as e:
            logger.error(f"Error in contextual analysis of {file.filename}: {e}")
            return []