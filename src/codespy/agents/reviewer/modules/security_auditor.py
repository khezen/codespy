"""Security vulnerability analyzer module."""

import logging

import dspy  # type: ignore[import-untyped]

from codespy.tools.github.models import ChangedFile
from codespy.agents.reviewer.models import Issue, IssueCategory
from codespy.agents.reviewer.modules.helpers import get_language, parse_issues_json

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
    file_path: str = dspy.InputField(
        desc="Path to the file being analyzed"
    )
    language: str = dspy.InputField(
        desc="Programming language of the file"
    )
    context: str = dspy.InputField(
        desc="Additional context from related files in the codebase"
    )

    issues_json: str = dspy.OutputField(
        desc="""JSON array of security issues found. Each issue should have:
        {
            "title": "Brief title",
            "severity": "critical|high|medium|low|info",
            "description": "Detailed explanation",
            "line_start": <number or null>,
            "line_end": <number or null>,
            "code_snippet": "Affected code",
            "suggestion": "How to fix",
            "cwe_id": "CWE-XXX or null",
            "confidence": <0.0-1.0>
        }
        Return empty array [] if no issues found."""
    )


class SecurityAuditor(dspy.Module):
    """Analyzes code for security vulnerabilities using DSPy."""

    category = IssueCategory.SECURITY

    def __init__(self) -> None:
        """Initialize the security auditor with chain-of-thought reasoning."""
        super().__init__()
        self.predictor = dspy.ChainOfThought(SecurityAnalysisSignature)

    def forward(self, file: ChangedFile, context: str = "") -> list[Issue]:
        """Analyze a file for security vulnerabilities and return issues.

        Args:
            file: The changed file to analyze
            context: Additional context from related files

        Returns:
            List of security issues found
        """
        if not file.patch:
            logger.debug(f"Skipping {file.filename}: no patch available")
            return []

        try:
            result = self.predictor(
                diff=file.patch or "",
                full_content=file.content or "",
                file_path=file.filename,
                language=get_language(file),
                context=context or "No additional context available.",
            )
            return parse_issues_json(result.issues_json, file, self.category)
        except Exception as e:
            logger.error(f"Error analyzing {file.filename}: {e}")
            return []