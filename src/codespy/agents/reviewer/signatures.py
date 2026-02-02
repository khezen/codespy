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

    issues_json: str = dspy.OutputField(
        desc="""JSON array of VERIFIED bugs found. Each bug should have:
        {
            "title": "Brief title",
            "severity": "critical|high|medium|low|info",
            "description": "What the bug is and why it's problematic - must include SPECIFIC code evidence",
            "line_start": <number or null>,
            "line_end": <number or null>,
            "code_snippet": "The EXACT buggy code",
            "suggestion": "How to fix the bug",
            "confidence": <0.0-1.0> - set low if not 100% sure
        }
        Return empty array [] if no VERIFIED bugs found. Do NOT include speculative issues."""
    )


class DocumentationReview(dspy.Signature):
    """Review markdown documentation files for accuracy and completeness.

    You are reviewing markdown documentation files (README.md, docs/*.md, etc.).
    This is NOT about code comments or docstrings - focus ONLY on:

    - Is the documentation accurate and up-to-date?
    - Are there factual errors or outdated information?
    - Is important information missing that users/developers need?
    - Are code examples correct and working?
    - Are links valid and pointing to the right resources?
    - Is the documentation clear and well-organized?

    Only report issues for MARKDOWN documentation files.
    Do NOT report issues about missing docstrings in code files.
    """

    diff: str = dspy.InputField(
        desc="The markdown documentation diff showing changes"
    )
    full_content: str = dspy.InputField(
        desc="The full markdown file content after changes"
    )
    file_path: str = dspy.InputField(
        desc="Path to the markdown file being analyzed"
    )

    issues_json: str = dspy.OutputField(
        desc="""JSON array of documentation issues. Each issue should have:
        {
            "title": "Brief title",
            "severity": "medium|low|info",
            "description": "What is inaccurate, outdated, or missing in the docs",
            "line_start": <number or null>,
            "line_end": <number or null>,
            "code_snippet": "Relevant documentation text",
            "suggestion": "How to improve the documentation",
            "confidence": <0.0-1.0>
        }
        Return empty array [] if documentation is adequate."""
    )


class ContextualAnalysis(dspy.Signature):
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