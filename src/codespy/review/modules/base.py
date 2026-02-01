"""Base module for DSPy review modules."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import dspy

from codespy.github.models import ChangedFile
from codespy.review.models import Issue, IssueCategory

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

    def _parse_issues(self, result: Any, file: ChangedFile) -> list[Issue]:
        """Parse the predictor result into Issue objects.

        Args:
            result: The DSPy predictor result
            file: The file being analyzed

        Returns:
            List of parsed issues
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
                        confidence=raw.get("confidence", 0.8),
                    )
                    issues.append(issue)
                except Exception as e:
                    logger.warning(f"Failed to parse issue: {e}")
                    continue

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse issues JSON: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing issues: {e}")

        return issues