"""Unified code review module — defects, security, and code smells."""

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


class CodeReviewSignature(dspy.Signature):
    """Detect VERIFIED code defects, security vulnerabilities, and code smells.

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
      configuration, and expected behavior.

    If README doesn't exist, search for alternative documentation files
    (e.g., docs/, README.rst) using get_tree or file_exists.

    ═══════════════════════════════════════════════════════════════════════════════
    PHASE 2 — ANALYZE CHANGES (PRIORITY ORDER)
    ═══════════════════════════════════════════════════════════════════════════════

    Review each changed file's patch. Prioritize bugs and security first, then smells.

    CRITICAL RULES:
    - ONLY report issues you can VERIFY using the available tools
    - Before reporting, USE the tools to verify your assumptions
    - DO NOT speculate — no "might be", "could be", "possibly" issues
    - If you cannot point to the EXACT defective code with evidence, do NOT report it
    - Quality over quantity: prefer 0 reports over 1 speculative report

    VERIFICATION WORKFLOW (shared across all categories):
    1. Review each changed file's patch — the diff shows what changed
    2. For suspected issues that need verification beyond the patch:
       - Use find_function_definitions to check function signatures and implementations
       - Use find_function_calls to understand how functions are called and trace data flow
       - Use find_function_usages/find_callers to trace usage patterns
       - Use search_literal to find related patterns (sanitization, validation, naming)
       - Use read_file ONLY if you need broader context not visible in the patch
    3. Only report issues that are CONFIRMED by your verification

    ─── A. BUGS & LOGIC ERRORS (category = "bug") ───────────────────────────────

    - Logic errors: verify the condition is actually incorrect by checking related code
    - Null/undefined references: verify the check is actually missing
    - Resource leaks: verify there's no cleanup in finally/defer/close methods
    - Error handling: verify errors aren't handled elsewhere in the call chain
    - Type mismatches: verify types by checking definitions
    - Off-by-one errors: verify by understanding the data structure bounds

    REMOVED DEFENSIVE CODE (always report as "bug"):
    If a guard (bounds check, null check, Math.max/min, default value, try/catch,
    sanitization) existed in removed lines (-) and is absent from added lines (+),
    report it. Use read_file or search_literal to confirm the guard was not relocated
    elsewhere in the same function/file. If it was moved, do not report.

    ─── B. SECURITY VULNERABILITIES (category = "security") ─────────────────────

    - Injection (SQL, command, XSS): verify input reaches dangerous sink unsanitized
    - Authentication/authorization: verify auth check is actually missing
    - Sensitive data exposure: verify data is actually exposed, not just accessed
    - Insecure crypto: verify the actual algorithm used
    - Path traversal: verify path input is not sanitized
    - Race conditions: verify shared state access without synchronization

    ─── C. CODE SMELLS (category = "smell") ──────────────────────────────────────

    UNCOMMUNICATIVE NAMES:
    - Variables not describing data they hold (data, info, item, temp, val)
    - Functions not starting with a verb (user(), process())
    - Booleans not reading as predicates (valid, flag → isValid, hasPermission)
    - Side-effect mismatch: getName() that writes to DB

    PRIMITIVE OBSESSION:
    - 3+ related primitive params traveling together → suggest struct/class

    COMPLEXITY:
    - Double negatives, nested ternaries, >2 logical operators per condition
    - Magic numbers in logic → suggest named constants
    - 3+ nesting levels → suggest guard clauses/early returns
    - 5+ switch/if-elif branches → suggest polymorphism

    YAGNI:
    - Abstract class/interface with single implementation (verify with find_callers)
    - Unused parameters accepted "for future use"

    DO NOT report as smells:
    - Idiomatic short names (i, j, k, err, ctx, db, tx)
    - Test file naming conventions

    ═══════════════════════════════════════════════════════════════════════════════
    OUTPUT RULES
    ═══════════════════════════════════════════════════════════════════════════════

    - Set category to one of the values provided in the categories input
    - For security issues, include cwe_id where applicable
    - Reference files by name and line number only — never copy source code into issues
    - Do not repeat patch content in reasoning steps. Keep each step to 1-2 sentences
    - Empty list if no issues found. No approval text ("LGTM", "looks good")
    - description: ≤25 words, imperative tone, no filler ("Fix X", "Rename Y to Z")
    - No polite or conversational language
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
        desc="Verified issues. Category must be one of the provided categories. "
        "Titles <10 words. Descriptions ≤25 words, imperative. Empty list if none. "
        "File paths must be relative to scope root."
    )


class CodeReviewer(dspy.Module):
    """Unified code reviewer — defects, security, and smells in a single pass.

    Merges DefectDetector and SmellDetector into one agent to avoid redundant
    README reads, tool sessions, and input token costs per scope.

    MCP tools are scope-restricted: for each scope, tools are rooted at
    repo_path/scope.subroot so the agent cannot access files outside the scope.
    """

    def __init__(self) -> None:
        """Initialize the code reviewer."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    async def aforward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for defects, security issues, and code smells.

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository

        Returns:
            List of bug, security, and smell issues found across all scopes
        """
        if not self._settings.is_signature_enabled("code_review"):
            logger.debug("Skipping code_review: disabled")
            return []

        # Determine which categories are active
        categories: list[IssueCategory] = []
        categories.append(IssueCategory.BUG)
        categories.append(IssueCategory.SECURITY)
        categories.append(IssueCategory.SMELL)

        changed_scopes = [s for s in scopes if s.has_changes and s.changed_files]
        if not changed_scopes:
            logger.info("No scopes with changes for code review")
            return []

        all_issues: list[Issue] = []
        max_iters = self._settings.get_max_iters("code_review")

        total_files = sum(len(s.changed_files) for s in changed_scopes)
        logger.info(
            f"Code review for {len(changed_scopes)} scopes "
            f"({total_files} changed files)..."
        )

        for scope in changed_scopes:
            scope_root = resolve_scope_root(repo_path, scope.subroot)
            tools, contexts = await create_mcp_tools(scope_root, "code_reviewer")
            try:
                agent = dspy.ReAct(
                    signature=CodeReviewSignature,
                    tools=tools,
                    max_iters=max_iters,
                )
                scoped = make_scope_relative(scope)
                logger.info(
                    f"  Code review: scope {scope.subroot} "
                    f"({len(scope.changed_files)} files)"
                )
                async with SignatureContext("code_review", self._cost_tracker):
                    result = await agent.acall(
                        scope=scoped,
                        categories=categories,
                    )

                issues = [
                    issue for issue in (result.issues or [])
                    if issue.confidence >= MIN_CONFIDENCE
                ]
                restore_repo_paths(issues, scope.subroot)
                all_issues.extend(issues)
                logger.debug(
                    f"  Scope {scope.subroot}: {len(issues)} code review issues"
                )
            except Exception as e:
                logger.error(f"Code review failed for scope {scope.subroot}: {e}")
            finally:
                await cleanup_mcp_contexts(contexts)

        logger.info(f"Code review found {len(all_issues)} issues")
        return all_issues

    def forward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for code issues (sync wrapper).

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository

        Returns:
            List of bug, security, and smell issues found across all scopes
        """
        return asyncio.run(self.aforward(scopes, repo_path))
