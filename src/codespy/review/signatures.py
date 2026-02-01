"""DSPy signatures for code review tasks."""

import dspy


class SecurityAnalysis(dspy.Signature):
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


class BugDetection(dspy.Signature):
    """Detect potential bugs and logic errors in code changes.

    You are an expert software engineer reviewing code for bugs. Look for:
    - Logic errors and incorrect conditions
    - Off-by-one errors
    - Null/undefined reference issues
    - Resource leaks (memory, file handles, connections)
    - Incorrect error handling
    - Race conditions and concurrency issues
    - Type mismatches
    - Incorrect API usage
    - Edge cases not handled
    - Incorrect algorithm implementation

    Focus on actual bugs, not style issues or minor improvements.
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

    issues_json: str = dspy.OutputField(
        desc="""JSON array of bugs found. Each bug should have:
        {
            "title": "Brief title",
            "severity": "critical|high|medium|low|info",
            "description": "What the bug is and why it's problematic",
            "line_start": <number or null>,
            "line_end": <number or null>,
            "code_snippet": "Buggy code",
            "suggestion": "How to fix the bug",
            "confidence": <0.0-1.0>
        }
        Return empty array [] if no bugs found."""
    )


class DocumentationReview(dspy.Signature):
    """Review code changes for documentation completeness.

    You are reviewing code for documentation quality. Check for:
    - Missing function/method docstrings
    - Missing class docstrings
    - Incomplete parameter documentation
    - Missing return value documentation
    - Outdated comments that don't match the code
    - Missing inline comments for complex logic
    - Missing README updates for public API changes
    - Missing type hints (for typed languages)

    Focus on documentation that helps understand and maintain the code.
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

    issues_json: str = dspy.OutputField(
        desc="""JSON array of documentation issues. Each issue should have:
        {
            "title": "Brief title",
            "severity": "medium|low|info",
            "description": "What documentation is missing or incorrect",
            "line_start": <number or null>,
            "line_end": <number or null>,
            "code_snippet": "Code needing documentation",
            "suggestion": "Suggested documentation to add",
            "confidence": <0.0-1.0>
        }
        Return empty array [] if documentation is adequate."""
    )


class ContextualAnalysis(dspy.Signature):
    """Analyze if code changes make sense within the broader codebase context.

    You are reviewing code changes considering the broader codebase. Check for:
    - Breaking changes to existing interfaces
    - Inconsistencies with existing patterns in the codebase
    - Missing updates to related files
    - Duplicate functionality that already exists
    - Incorrect assumptions about how other parts of the code work
    - Missing tests for new functionality
    - API contract violations

    This analysis requires understanding how the changed code interacts with
    the rest of the codebase.
    """

    diff: str = dspy.InputField(
        desc="The code diff showing changes"
    )
    file_path: str = dspy.InputField(
        desc="Path to the file being analyzed"
    )
    related_files: str = dspy.InputField(
        desc="Content of related files (imports, dependencies)"
    )
    repo_structure: str = dspy.InputField(
        desc="Overview of the repository structure"
    )

    issues_json: str = dspy.OutputField(
        desc="""JSON array of contextual issues. Each issue should have:
        {
            "title": "Brief title",
            "severity": "critical|high|medium|low|info",
            "description": "How the change conflicts with or misses context",
            "line_start": <number or null>,
            "line_end": <number or null>,
            "code_snippet": "Relevant code",
            "suggestion": "How to align with codebase",
            "confidence": <0.0-1.0>
        }
        Return empty array [] if changes align well with codebase."""
    )


class PRSummary(dspy.Signature):
    """Generate an overall summary and recommendation for a pull request.

    Based on all the issues found during review, provide:
    - A concise summary of what the PR does
    - An overall assessment of the code quality
    - A recommendation (approve, request changes, or needs discussion)
    """

    pr_title: str = dspy.InputField(desc="Title of the pull request")
    pr_description: str = dspy.InputField(desc="Description/body of the PR")
    changed_files: str = dspy.InputField(
        desc="List of changed files with change counts"
    )
    all_issues: str = dspy.InputField(
        desc="JSON array of all issues found during review"
    )

    summary: str = dspy.OutputField(
        desc="2-3 sentence summary of what this PR accomplishes"
    )
    recommendation: str = dspy.OutputField(
        desc="One of: APPROVE, REQUEST_CHANGES, or NEEDS_DISCUSSION with brief justification"
    )