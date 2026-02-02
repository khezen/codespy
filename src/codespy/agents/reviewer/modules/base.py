"""Base module for DSPy review modules."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import dspy

from codespy.tools.github.models import ChangedFile
from codespy.agents.reviewer.models import Issue, IssueCategory

logger = logging.getLogger(__name__)


# Language detection based on file extension
EXTENSION_TO_LANGUAGE = {
    "py": "Python",
    "js": "JavaScript",
    "ts": "TypeScript",
    "jsx": "JavaScript (React)",
    "tsx": "TypeScript (React)",
    "go": "Go",
    "rs": "Rust",
    "java": "Java",
    "kt": "Kotlin",
    "c": "C",
    "cpp": "C++",
    "h": "C/C++ Header",
    "hpp": "C++ Header",
    "cs": "C#",
    "rb": "Ruby",
    "php": "PHP",
    "swift": "Swift",
    "scala": "Scala",
    "sh": "Shell",
    "bash": "Bash",
    "sql": "SQL",
    "vue": "Vue",
    "svelte": "Svelte",
}

# Phrases that indicate speculative/unverified issues - filter these out
SPECULATIVE_PHRASES = [
    "cannot be verified",
    "without access to",
    "may need",
    "might need",
    "could cause",
    "could break",
    "might break",
    "may cause",
    "possibly",
    "potentially",
    "i assume",
    "there might be",
    "there may be",
    "likely",
    "probably",
    "it's possible",
    "it is possible",
    "should verify",
    "need to verify",
    "needs verification",
    "cannot confirm",
    "unable to verify",
    "ensure that",
    "make sure that",
    "verify that",
]

# Minimum confidence threshold
MIN_CONFIDENCE = 0.7


class BaseReviewModule(ABC):
    """Base class for all review modules."""

    category: IssueCategory

    def __init__(self) -> None:
        """Initialize the module."""
        self._predictor = self._create_predictor()

    @abstractmethod
    def _create_predictor(self) -> dspy.Module:
        """Create the DSPy predictor for this module."""
        pass

    @abstractmethod
    def _prepare_inputs(
        self,
        file: ChangedFile,
        context: str = "",
    ) -> dict[str, Any]:
        """Prepare inputs for the predictor."""
        pass

    def get_language(self, file: ChangedFile) -> str:
        """Get the programming language for a file."""
        return EXTENSION_TO_LANGUAGE.get(file.extension, "Unknown")

    def analyze(
        self,
        file: ChangedFile,
        context: str = "",
    ) -> list[Issue]:
        """Analyze a file and return issues.

        Args:
            file: The changed file to analyze
            context: Additional context from related files

        Returns:
            List of issues found
        """
        if not file.patch:
            logger.debug(f"Skipping {file.filename}: no patch available")
            return []

        try:
            inputs = self._prepare_inputs(file, context)
            result = self._predictor(**inputs)
            return self._parse_issues(result, file)
        except Exception as e:
            logger.error(f"Error analyzing {file.filename}: {e}")
            return []

    def _is_speculative(self, issue: Issue) -> bool:
        """Check if an issue contains speculative/unverified language.

        Args:
            issue: The issue to check

        Returns:
            True if the issue appears speculative
        """
        # Check description and suggestion for speculative phrases
        text_to_check = (
            (issue.description or "").lower() +
            " " +
            (issue.suggestion or "").lower() +
            " " +
            (issue.title or "").lower()
        )

        for phrase in SPECULATIVE_PHRASES:
            if phrase in text_to_check:
                logger.debug(
                    f"Filtering speculative issue '{issue.title}': contains '{phrase}'"
                )
                return True

        return False

    def _parse_issues(self, result: Any, file: ChangedFile) -> list[Issue]:
        """Parse the predictor result into Issue objects.

        Args:
            result: The DSPy predictor result
            file: The file being analyzed

        Returns:
            List of parsed issues (filtered for quality)
        """
        issues: list[Issue] = []

        try:
            # Get the issues_json field from the result
            issues_json = getattr(result, "issues_json", "[]")

            # Clean up the JSON string (handle markdown code blocks)
            issues_json = issues_json.strip()
            if issues_json.startswith("```"):
                # Remove markdown code block
                lines = issues_json.split("\n")
                issues_json = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            # Parse JSON
            raw_issues = json.loads(issues_json)

            if not isinstance(raw_issues, list):
                logger.warning(f"Expected list of issues, got {type(raw_issues)}")
                return []

            for raw in raw_issues:
                if not isinstance(raw, dict):
                    continue

                try:
                    confidence = raw.get("confidence", 0.8)

                    # Filter out low-confidence issues
                    if confidence < MIN_CONFIDENCE:
                        logger.debug(
                            f"Filtering low-confidence issue '{raw.get('title', 'Untitled')}': "
                            f"confidence {confidence} < {MIN_CONFIDENCE}"
                        )
                        continue

                    issue = Issue(
                        category=self.category,
                        severity=raw.get("severity", "medium"),
                        title=raw.get("title", "Untitled Issue"),
                        description=raw.get("description", ""),
                        file=file.filename,
                        line_start=raw.get("line_start"),
                        line_end=raw.get("line_end"),
                        code_snippet=raw.get("code_snippet"),
                        suggestion=raw.get("suggestion"),
                        cwe_id=raw.get("cwe_id"),
                        confidence=confidence,
                    )

                    # Filter out speculative issues
                    if self._is_speculative(issue):
                        continue

                    issues.append(issue)
                except Exception as e:
                    logger.warning(f"Failed to parse issue: {e}")
                    continue

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse issues JSON: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing issues: {e}")

        return issues