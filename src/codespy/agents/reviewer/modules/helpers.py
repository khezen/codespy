"""Helper functions for DSPy review modules."""

import json
import logging
import os

from codespy.tools.github.models import ChangedFile, ReviewContext
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

# Markdown file extensions to review
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx", ".rst", ".txt"}


def is_markdown_file(filename: str) -> bool:
    """Check if the file is a markdown documentation file."""
    _, ext = os.path.splitext(filename.lower())
    return ext in MARKDOWN_EXTENSIONS


def get_language(file: ChangedFile) -> str:
    """Get the programming language for a file based on extension.
    
    Args:
        file: The changed file
        
    Returns:
        Language name or "Unknown"
    """
    return EXTENSION_TO_LANGUAGE.get(file.extension, "Unknown")


def is_speculative(issue: Issue) -> bool:
    """Check if an issue contains speculative/unverified language.

    Args:
        issue: The issue to check

    Returns:
        True if the issue appears speculative
    """
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


def parse_issues_json(
    issues_json: str,
    file: ChangedFile,
    category: IssueCategory,
) -> list[Issue]:
    """Parse the issues JSON and filter for quality.

    Args:
        issues_json: Raw JSON string from the LLM
        file: The file being analyzed
        category: Issue category for all parsed issues

    Returns:
        List of parsed issues (filtered for quality)
    """
    issues: list[Issue] = []

    try:
        # Clean up the JSON string (handle markdown code blocks)
        issues_json = issues_json.strip()
        if issues_json.startswith("```"):
            lines = issues_json.split("\n")
            issues_json = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        raw_issues = json.loads(issues_json)

        if not isinstance(raw_issues, list):
            logger.warning(f"Expected list of issues, got {type(raw_issues)}")
            return []

        for raw in raw_issues:
            if not isinstance(raw, dict):
                continue

            try:
                confidence = raw.get("confidence", 0.8)

                if confidence < MIN_CONFIDENCE:
                    logger.debug(
                        f"Filtering low-confidence issue '{raw.get('title', 'Untitled')}': "
                        f"confidence {confidence} < {MIN_CONFIDENCE}"
                    )
                    continue

                issue = Issue(
                    category=category,
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

                if is_speculative(issue):
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


def build_context_string(file: ChangedFile, review_context: ReviewContext) -> tuple[str, str]:
    """Build context strings from ReviewContext for a specific file.
    
    Args:
        file: The changed file being analyzed
        review_context: Full review context including related files
        
    Returns:
        Tuple of (related_files_str, repo_structure_str)
    """
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
    
    return related_files_str, repo_structure
