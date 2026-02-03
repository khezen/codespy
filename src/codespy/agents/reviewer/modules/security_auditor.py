"""Security vulnerability analyzer module."""

import logging

import dspy  # type: ignore[import-untyped]

from codespy.tools.github.models import ChangedFile
from codespy.agents.reviewer.models import Issue, IssueCategory
from codespy.agents.reviewer.modules.helpers import get_language, is_speculative, MIN_CONFIDENCE

logger = logging.getLogger(__name__)


class SecurityAnalysisSignature(dspy.Signature):
    """Analyze code changes for security vulnerabilities.

    You are a security expert reviewing code changes. Identify potential security
    vulnerabilities including but not limited to:
    - Injection attacks (SQL, command, XSS, etc.)
    - Authentication and authorization issues
    - Sensitive data exposure
    - Insecure cryptographic practices
    - Security misconfigurations
    - Input validation issues
    - Path traversal vulnerabilities
    - Race conditions
    - Memory safety issues

    For each issue, provide:
    - A clear title
    - Severity (critical, high, medium, low, info)
    - Detailed description of the vulnerability
    - The affected code location
    - A suggested fix
    - CWE ID if applicable
    """

    diff: str = dspy.InputField(
        desc="The code diff showing changes (unified diff format)"
    )
    full_content: str = dspy.InputField(
        desc="The full file content after changes"
    )
    filename: str = dspy.InputField(
        desc="Path to the file being analyzed"
    )
    language: str = dspy.InputField(
        desc="Programming language of the file"
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="Security issues found. Empty list if none."
    )


class SecurityAuditor(dspy.Module):
    """Analyzes code for security vulnerabilities using DSPy."""

    category = IssueCategory.SECURITY

    def __init__(self) -> None:
        """Initialize the security auditor with chain-of-thought reasoning."""
        super().__init__()

    def forward(self, file: ChangedFile) -> list[Issue]:
        """Analyze a file for security vulnerabilities and return issues.

        Args:
            file: The changed file to analyze

        Returns:
            List of security issues found
        """
        if not file.patch:
            logger.debug(f"Skipping {file.filename}: no patch available")
            return []

        try:
            agent = dspy.ChainOfThought(SecurityAnalysisSignature)
            result = agent(
                diff=file.patch or "",
                full_content=file.content or "",
                filename=file.filename,
                language=get_language(file),
                category=self.category,
            )
            return [
                issue for issue in result.issues
                if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
            ]
        except Exception as e:
            logger.error(f"Error analyzing {file.filename}: {e}")
            return []