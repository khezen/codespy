"""Code defect and security vulnerability detection module."""

import asyncio
import logging
from pathlib import Path
from typing import Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.agents.reviewer.modules.helpers import (
    MIN_CONFIDENCE,
    create_mcp_tools,
    make_scope_relative,
    resolve_scope_root,
    restore_repo_paths,
)
from codespy.config import get_settings
from codespy.tools.mcp_utils import cleanup_mcp_contexts

logger = logging.getLogger(__name__)


class BugDetectorSignature(dspy.Signature):
    """Detect VERIFIED code defects and security vulnerabilities in a scope.

    You are a busy Principal Engineer with very little time.
    Be extremely terse. Use imperative mood ("Fix X", not "You should fix X").
    You have tools to explore the scope's filesystem, search for text, and analyze code.
    All file paths are relative to the scope root directory (the current tool root).
    Tools are restricted to this scope — you cannot access files outside it.

    ═══════════════════════════════════════════════════════════════════════════════
    PHASE 1 — READ DOCUMENTATION (MANDATORY FIRST STEP)
    ═══════════════════════════════════════════════════════════════════════════════

    Use read_file to read the README at the scope root:
    - Path: readme.md OR README.md (paths are relative to scope root)
    - This file provides essential context about the scope's purpose, API contracts,
      configuration, and expected behavior. Use it to inform defect detection.

    If README doesn't exist at scope root, search for alternative documentation files
    (e.g., docs/, documentation/, README.rst) using get_tree or file_exists.

    ═══════════════════════════════════════════════════════════════════════════════
    PHASE 2 — ANALYZE CHANGES FOR DEFECTS
    ═══════════════════════════════════════════════════════════════════════════════

    Using the README context and the patches in scope.changed_files, look for:

    CRITICAL RULES:
    - ONLY report defects you can VERIFY using the available tools
    - Before reporting any defect, USE the tools to verify your assumptions
    - DO NOT speculate about potential issues you cannot verify
    - DO NOT report "might be", "could be", "possibly", "may cause" issues
    - If you cannot point to the EXACT defective code with evidence, do NOT report it
    - Quality over quantity: prefer 0 reports over 1 speculative report

    VERIFICATION WORKFLOW:
    1. Review each changed file's patch — the diff shows what changed
    2. For suspected defects that need verification beyond the patch:
       - Use find_function_definitions to check function signatures and implementations
       - Use find_function_calls to understand how functions are called and trace data flow
       - Use find_function_usages/find_callers to trace usage patterns
       - Use search_literal to find related patterns (sanitization, encoding, validation)
       - Use read_file ONLY if you need broader context not visible in the patch
    3. Only report issues that are CONFIRMED by your verification

    BUGS & LOGIC ERRORS (category = "bug"):
    - Logic errors: verify the condition is actually incorrect by checking related code
    - Null/undefined references: verify the check is actually missing
    - Resource leaks: verify there's no cleanup in finally/defer/close methods
    - Error handling: verify errors aren't handled elsewhere in the call chain
    - Type mismatches: verify types by checking definitions
    - Off-by-one errors: verify by understanding the data structure bounds

    SECURITY VULNERABILITIES (category = "security"):
    - Injection attacks (SQL, command, XSS): verify input reaches dangerous sink without sanitization
    - Authentication/authorization issues: verify auth check is actually missing
    - Sensitive data exposure: verify data is actually exposed, not just accessed
    - Insecure cryptographic practices: verify the actual algorithm used
    - Security misconfigurations: verify by checking actual config values
    - Input validation issues: verify input is not validated
    - Path traversal: verify path input is not sanitized
    - Race conditions: verify shared state access without synchronization
    - Memory safety issues: verify unsafe memory operations

    DO NOT report:
    - Hypothetical vulnerabilities without evidence
    - "Could be vulnerable if..." scenarios

    ═══════════════════════════════════════════════════════════════════════════════
    OUTPUT RULES
    ═══════════════════════════════════════════════════════════════════════════════

    - Set category to one of the values provided in the categories input
    - For security issues, include cwe_id where applicable
    - Reference files by name and line number only — never copy source code into issues
    - Do not repeat patch content in reasoning steps. Keep each reasoning step to 1-2 sentences
    - Empty list if no issues found. No approval text ("LGTM", "looks good")
    - description: ≤25 words, imperative tone, no filler ("Fix X", "Update Y section")
    - No polite or conversational language ("I suggest", "Please consider", "Great")
    - Do not populate code_snippet — use line numbers instead
    """

    scope: ScopeResult = dspy.InputField(
        desc="Scope with changed files. Has: subroot, scope_type, "
        "changed_files (filename + patch - analyze patch first), language, package_manifest. "
        "File paths in changed_files are relative to the scope root (tool root)."
    )
    categories: list[IssueCategory] = dspy.InputField(
        desc="Allowed issue categories. Use only these values for the 'category' field on each issue."
    )

    issues: list[Issue] = dspy.OutputField(
        desc="Verified defects. Category must be one of the provided categories. "
        "Titles <10 words. Descriptions ≤25 words, imperative. Empty list if none. "
        "File paths must be relative to scope root."
    )


class BugDetector(dspy.Module):
    """Detects code defects and security vulnerabilities.

    Focuses exclusively on bugs and security issues, using MCP tools
    to verify each finding before reporting.

    MCP tools are scope-restricted: for each scope, tools are rooted at
    repo_path/scope.subroot so the agent cannot access files outside the scope.
    """

    def __init__(self) -> None:
        """Initialize the bug detector."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    async def aforward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for code defects and security vulnerabilities.

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository

        Returns:
            List of bug and security issues found across all scopes
        """
        if not self._settings.is_signature_enabled("bug_review"):
            logger.debug("Skipping bug_review: disabled")
            return []

        changed_scopes = [s for s in scopes if s.has_changes and s.changed_files]
        if not changed_scopes:
            logger.info("No scopes with changes for bug detection")
            return []

        all_issues: list[Issue] = []
        max_iters = self._settings.get_max_iters("bug_review")

        total_files = sum(len(s.changed_files) for s in changed_scopes)
        logger.info(
            f"Bug detection for {len(changed_scopes)} scopes "
            f"({total_files} changed files)..."
        )

        for scope in changed_scopes:
            scope_root = resolve_scope_root(repo_path, scope.subroot)
            tools, contexts = await create_mcp_tools(scope_root, "bug_detector")
            try:
                agent = dspy.ReAct(
                    signature=BugDetectorSignature,
                    tools=tools,
                    max_iters=max_iters,
                )
                scoped = make_scope_relative(scope)
                logger.info(
                    f"  Bug detection: scope {scope.subroot} "
                    f"({len(scope.changed_files)} files)"
                )
                async with SignatureContext("bug_review", self._cost_tracker):
                    result = await agent.acall(
                        scope=scoped,
                        categories=[IssueCategory.BUG, IssueCategory.SECURITY],
                    )

                issues = [
                    issue for issue in (result.issues or [])
                    if issue.confidence >= MIN_CONFIDENCE
                ]
                restore_repo_paths(issues, scope.subroot)
                all_issues.extend(issues)
                logger.debug(
                    f"  Scope {scope.subroot}: {len(issues)} bug/security issues"
                )
            except Exception as e:
                logger.error(f"Bug detection failed for scope {scope.subroot}: {e}")
            finally:
                await cleanup_mcp_contexts(contexts)

        logger.info(f"Bug detection found {len(all_issues)} issues")
        return all_issues

    def forward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for defects (sync wrapper).

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository

        Returns:
            List of bug and security issues found across all scopes
        """
        return asyncio.run(self.aforward(scopes, repo_path))