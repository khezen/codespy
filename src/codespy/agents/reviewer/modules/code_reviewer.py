"""Unified code review module — defects, security, and code smells."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.dspy_config import AsyncReActV2
from codespy.agents.memory.hippocampus import Hippocampus
from codespy.agents.memory.hippocampus import ContextMap
from codespy.agents.reviewer.models import Issue, IssueCategory, ReviewContext, ScopeResult
from codespy.agents.reviewer.modules.helpers import (
    MIN_CONFIDENCE,
    issues_to_markdown,
    make_scope_relative,
    resolve_scope_root,
    restore_repo_paths,
)

from codespy.config import get_settings
from codespy.config_memory import get_memory_store
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class CodeReviewSignature(dspy.Signature):
    """Detect VERIFIED code defects, security vulnerabilities, and code smells.

    You are a busy Principal Engineer with very little time.
    Be extremely terse. Use imperative mood ("Fix X", not "You should fix X").
    You have tools to explore the scope's filesystem, search for text, and analyze code.
    All file paths are relative to the scope root directory (the current tool root).
    Tools are restricted to this scope — you cannot access files outside it.

    ═══════════════════════════════════════════════════════════════════════════════
    ANALYZE CHANGES
    ═══════════════════════════════════════════════════════════════════════════════

    Review each changed file's patch. For each file, check ALL categories (A, B, C)
    before moving to the next file.

    TOOLS AVAILABLE:
    - find_function_definitions: check function signatures and implementations
    - find_function_calls: understand how functions are called, trace data flow
    - find_function_usages / find_callers: trace usage patterns
    - search_literal: find related patterns (sanitization, validation, naming)
    - read_file:broader context not visible in the patch (use sparingly)

    ─── A. BUGS & LOGIC ERRORS (category = "bug") ───────────────────────────────

    EVIDENCE STANDARD: STRICT — you must VERIFY with tools before reporting.
    DO NOT speculate. No "might be", "could be", "possibly" issues.
    If you cannot point to the EXACT defective code with evidence, do NOT report.

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

    EVIDENCE STANDARD: STRICT — you must VERIFY with tools before reporting.
    DO NOT speculate. Trace data flow from source to sink.

    - Injection (SQL, command, XSS): verify input reaches dangerous sink unsanitized
    - Authentication/authorization: verify auth check is actually missing
    - Sensitive data exposure: verify data is actually exposed, not just accessed
    - Insecure crypto: verify the actual algorithm used
    - Path traversal: verify path input is not sanitized
    - Race conditions: verify shared state access without synchronization

    ─── C. CODE SMELLS (category = "smell") ──────────────────────────────────────

    EVIDENCE STANDARD: OBSERVATIONAL — the patch or function signature IS the evidence.
    A function with 5 primitive parameters IS a smell — no further verification needed.
    A variable named "data" IS a smell — the name itself is the evidence.
    Use tools to confirm if needed (e.g., find_function_definitions for param lists),
    but the structural pattern visible in the patch is sufficient to report.

    UNCOMMUNICATIVE NAMES:
    - Variables not describing data they hold (data, info, item, temp, val)
    - Functions not starting with a verb (user(), process())
    - Booleans not reading as predicates (valid, flag → isValid, hasPermission)
    - Side-effect mismatch: getName() that writes to DB

    PRIMITIVE OBSESSION:
    - Functions taking 3+ related primitive parameters that travel together
      Example: (string zip, string city, string street) → suggest Address struct/class
    - Use find_function_definitions to inspect parameter lists if not visible in patch

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

    MCP tools are scope-restricted: for each scope, tools are rooted at
    repo_path/scope.subroot so the agent cannot access files outside the scope.
    """

    def __init__(self) -> None:
        """Initialize the code reviewer."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    async def _create_tools(
        self, scope_root: Path
    ) -> tuple[list[Any], list[Any]]:
        """Create scope-restricted tools: filesystem + ripgrep + treesitter."""
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        scope_root_str = str(scope_root)
        caller = "code_reviewer"

        tools.extend(await connect_mcp_server(
            tools_dir / "storage" / "filesystem" / "server.py",
            [scope_root_str], contexts, caller,
        ))
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "ripgrep" / "server.py",
            [scope_root_str], contexts, caller,
        ))
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "treesitter" / "server.py",
            [scope_root_str], contexts, caller,
        ))
        return tools, contexts

    async def aforward(
        self,
        scopes: Sequence[ScopeResult],
        repo_path: Path,
        run_id: str | None = None,
        review_context: ReviewContext | None = None,
    ) -> tuple[list[Issue], ContextMap | None]:
        """Analyze scopes for defects, security issues, and code smells.

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository
            run_id: Identifier of the pipeline run, shared across all agents
                invoked within the same review run (see ``Hippocampus.run_id``)
            review_context: ReviewContext containing PR identity and inherited memory

        Returns:
            Tuple of (list of issues, merged context map or None)
        """
        if not self._settings.is_signature_enabled("code_review"):
            logger.debug("Skipping code_review: disabled")
            return [], review_context.memory if review_context else None

        # Determine which categories are active
        categories: list[IssueCategory] = []
        categories.append(IssueCategory.BUG)
        categories.append(IssueCategory.SECURITY)
        categories.append(IssueCategory.SMELL)

        changed_scopes = [s for s in scopes if s.has_changes and s.changed_files]
        if not changed_scopes:
            logger.info("No scopes with changes for code review")
            return [], review_context.memory if review_context else None

        all_issues: list[Issue] = []
        scope_memories: list[ContextMap] = []
        max_iters = self._settings.get_max_iters("code_review")

        total_files = sum(len(s.changed_files) for s in changed_scopes)
        logger.info(
            f"Code review for {len(changed_scopes)} scopes "
            f"({total_files} changed files)..."
        )

        for scope in changed_scopes:
            scope_root = resolve_scope_root(repo_path, scope.subroot)
            tools, contexts = await self._create_tools(scope_root)
            try:
                agent = AsyncReActV2(
                    signature=CodeReviewSignature,
                    tools=tools,
                    max_iters=max_iters,
                )
                scoped = make_scope_relative(scope)
                logger.info(
                    f"  Code review: scope {scope.subroot} "
                    f"({len(scope.changed_files)} files)"
                )
                mem: Hippocampus | None = None
                async with SignatureContext("code_review", self._cost_tracker):
                    if self._settings.get_memory_enabled("code_review"):
                        question = (
                            f"review code change of {scope.repo}: {scope.subroot}: "
                            f"pull request {review_context.pr_context.mr_number} {review_context.pr_context.mr_title}: {review_context.pr_context.summary}"
                        ) if review_context else None
                        mem = Hippocampus(
                            agent,
                            budget=self._settings.get_memory_budget("code_review"),
                            max_reflects=self._settings.get_memory_max_reflects("code_review"),
                            question=question,
                            task_name="code_review",
                            run_id=run_id,
                            initial_memory=review_context.memory if review_context else None,
                        )
                        result = await mem.aforward(
                            scope=scoped,
                            categories=categories,
                        )
                        issues = [
                            issue for issue in (result.issues or [])
                            if issue.confidence >= MIN_CONFIDENCE
                        ]
                        await mem.aend_episode(
                            get_memory_store(self._settings),
                            scope.scope_path(),
                            artifacts={"review": issues_to_markdown(issues)},
                        )
                        # Collect scope's context map
                        if mem:
                            scope_memories.append(mem.cmap.model_copy(deep=True))
                    else:
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
                logger.error(f"Code review failed for scope {scope.subroot}: {e}", exc_info=True)
            finally:
                await cleanup_mcp_contexts(contexts)

        logger.info(f"Code review found {len(all_issues)} issues")
        # Merge all scope context maps into one module-level map
        merged_memory = (
            ContextMap.merge(*scope_memories)
            if scope_memories
            else (review_context.memory if review_context else None)
        )
        return all_issues, merged_memory

    def forward(
        self,
        scopes: Sequence[ScopeResult],
        repo_path: Path,
        run_id: str | None = None,
        review_context: ReviewContext | None = None,
    ) -> tuple[list[Issue], ContextMap | None]:
        """Analyze scopes for code issues (sync wrapper).

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository
            run_id: Identifier of the pipeline run, shared across all agents
                invoked within the same review run
            review_context: ReviewContext containing PR identity and inherited memory

        Returns:
            Tuple of (list of issues, merged context map or None)
        """
        return asyncio.run(self.aforward(scopes, repo_path, run_id=run_id, review_context=review_context))
