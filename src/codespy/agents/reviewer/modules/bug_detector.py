"""Bug detection module."""

import logging

import dspy  # type: ignore[import-untyped]

from codespy.tools.github.models import ChangedFile
from codespy.agents.reviewer.models import Issue, IssueCategory
from codespy.agents.reviewer.modules.helpers import get_language, is_speculative, MIN_CONFIDENCE

logger = logging.getLogger(__name__)


class BugDetectionSignature(dspy.Signature):
    """Detect VERIFIED bugs and logic errors in code changes.

    You are an expert software engineer reviewing code for bugs.

    CRITICAL RULES:
    - ONLY report bugs you can DIRECTLY SEE in the code diff or full content
    - DO NOT speculate about potential issues you cannot verify
    - DO NOT report "might be", "could be", "possibly", "may cause" issues
    - If you cannot point to the EXACT buggy code, do NOT report it
    - Quality over quantity: prefer 0 reports over 1 speculative report

    Look for CONCRETE bugs:
    - Logic errors with clear incorrect conditions visible in code
    - Null/undefined references where you can see the missing check
    - Resource leaks where you can see open without close
    - Error handling where you can see the missing try/catch or error check
    - Type mismatches visible in the code
    - Off-by-one errors with clear evidence

    DO NOT report:
    - Style issues or minor improvements
    - Hypothetical edge cases you cannot see evidence for
    - "This might cause problems" without concrete evidence
    """

    diff: str = dspy.InputField(
        desc="The code diff showing changes"
    )
    full_content: str = dspy.InputField(
        desc="The full file content after changes"
    )
    file_path: str = dspy.InputField(
        desc="Path to the file being analyzed"
    )
    language: str = dspy.InputField(
        desc="Programming language of the file"
    )
    context: str = dspy.InputField(
        desc="Additional context from related files"
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="VERIFIED bugs found. Use file_path for 'file' field and category for 'category' field. Empty list if none."
    )


class BugDetector(dspy.Module):
    """Detects potential bugs and logic errors using DSPy."""

    category = IssueCategory.BUG

    def __init__(self) -> None:
        """Initialize the bug detector with chain-of-thought reasoning."""
        super().__init__()

    def forward(self, file: ChangedFile, context: str = "") -> list[Issue]:
        """Analyze a file for bugs and return issues.

        Args:
            file: The changed file to analyze
            context: Additional context from related files

        Returns:
            List of bug issues found
        """
        if not file.patch:
            logger.debug(f"Skipping {file.filename}: no patch available")
            return []
        
        try:
            agent = dspy.ChainOfThought(BugDetectionSignature)
            result = agent(
                diff=file.patch or "",
                full_content=file.content or "",
                file_path=file.filename,
                language=get_language(file),
                context=context or "No additional context available.",
                category=self.category,
            )
            return [
                issue for issue in result.issues
                if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
            ]
        except Exception as e:
            logger.error(f"Error analyzing {file.filename}: {e}")
            return []