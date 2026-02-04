"""Helper functions for DSPy review modules."""

import logging
import os

from codespy.tools.github.models import ChangedFile
from codespy.agents.reviewer.models import Issue

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
MIN_CONFIDENCE = 0.6

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


