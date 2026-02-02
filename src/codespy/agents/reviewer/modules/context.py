"""Contextual analysis module for codebase-aware review."""

import logging
from typing import Any

import dspy

from codespy.tools.github.models import ChangedFile, ReviewContext
from codespy.agents.reviewer.models import Issue, IssueCategory
from codespy.agents.reviewer.modules.base import BaseReviewModule

logger = logging.getLogger(__name__)


class ContextualAnalysisSignature(dspy.Signature):
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
    file_path: str = dspy.InputField(
        desc="Path to the file being analyzed"
    )
    related_files: str = dspy.InputField(
        desc="Content of related files AND verified callers - includes 'Verified Callers of Changed Functions' section with file:line references when available"
    )
    repo_structure: str = dspy.InputField(
        desc="Overview of the repository structure"
    )

    issues_json: str = dspy.OutputField(
        desc="""JSON array of VERIFIED contextual issues. Each issue should have:
        {
            "title": "Brief title - be specific about what caller/file is affected",
            "severity": "critical|high|medium|low|info",
            "description": "MUST cite specific file:line from verified callers or related_files. Example: 'The caller at api/handler.go:45 calls parse() with 2 args but signature changed to 3 args'",
            "line_start": <number or null>,
            "line_end": <number or null>,
            "code_snippet": "The changed code AND the caller code that needs updating",
            "suggestion": "Specific fix with file:line references",
            "confidence": <0.0-1.0> - set to 0.9+ if you have verified caller info
        }
        Return empty array [] if no verified callers need updating and no issues found in related_files.
        Quality over quantity - only report issues with concrete evidence."""
    )


class ContextAnalyzer(BaseReviewModule):
    """Analyzes code changes in the context of the broader codebase using DSPy.
    
    This module uses chain-of-thought reasoning to identify issues that require
    understanding of how the changed code relates to other parts of the codebase.
    It focuses on verified callers and related files to find breaking changes.
    """

    category = IssueCategory.CONTEXT

    def __init__(self) -> None:
        """Initialize the context analyzer with chain-of-thought reasoning."""
        super().__init__()
        self.predictor = dspy.ChainOfThought(ContextualAnalysisSignature)

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
        
        This is an enhanced version of analyze() that takes the full ReviewContext
        object and extracts related files and caller information automatically.

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
            issues_json = self.forward(**inputs)
            return self._parse_and_filter_issues(issues_json, file)
        except Exception as e:
            logger.error(f"Error in contextual analysis of {file.filename}: {e}")
            return []