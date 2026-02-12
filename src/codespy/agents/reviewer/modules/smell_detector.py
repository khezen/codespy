"""Code smell detection module."""

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


class SmellDetectorSignature(dspy.Signature):
    """Detect code smells in changed code within a scope.

    You are a busy Principal Engineer with very little time.
    Be extremely terse. Use imperative mood ("Rename X", not "You should rename X").
    You have tools to explore the scope's filesystem, search for text, and analyze code.
    All file paths are relative to the scope root directory (the current tool root).
    Tools are restricted to this scope — you cannot access files outside it.

    ═══════════════════════════════════════════════════════════════════════════════
    PHASE 1 — READ DOCUMENTATION (MANDATORY FIRST STEP)
    ═══════════════════════════════════════════════════════════════════════════════

    Use read_file to read the README at the scope root:
    - Path: readme.md OR README.md (paths are relative to scope root)
    - This file provides context about the scope's conventions, patterns,
      and domain terminology. Use it to calibrate your smell detection.

    If README doesn't exist at scope root, search for alternative documentation files
    (e.g., docs/, documentation/, README.rst) using get_tree or file_exists.

    ═══════════════════════════════════════════════════════════════════════════════
    PHASE 2 — ANALYZE CHANGES FOR CODE SMELLS
    ═══════════════════════════════════════════════════════════════════════════════

    Code smells signal deeper structural rot even when tests pass. Focus on
    semantics, intent clarity, and whether the code "makes sense" to a reader.

    VERIFICATION WORKFLOW:
    1. Review the patch for suspicious names, parameter lists, or control flow
    2. Use find_function_definitions to check function signatures and body length
    3. Use find_function_calls / find_callers to verify usage patterns
    4. Use search_literal to find related constants, sibling implementations, etc.
    5. Only report smells you can CONFIRM with evidence from the tools

    ── UNCOMMUNICATIVE NAMES (Semantic Naming) ───────────────────────────────────

    - Variables MUST be nouns describing the data they hold.
      Flag: data, info, item, rs, list1, temp, val, x in non-trivial contexts.
      Compare the variable name against its type or source to verify mismatch.
    - Functions MUST start with a verb indicating the action.
      Flag: user(), process(), handle() with no object. Suggest: calculateUserTax(), fetchActiveUsers().
    - Booleans MUST read as predicates (answer a yes/no question).
      Flag: valid, flag, status, ready. Suggest: isEmailFormatValid, hasPermission, shouldRetry.
    - Side-effect mismatch: a function named getName() that also writes to a database,
      or getData() that deletes records. Use find_function_definitions to read the body.

    ── PRIMITIVE OBSESSION (Data Clumps) ─────────────────────────────────────────

    - Flag functions taking 3+ related primitive parameters that travel together.
      Example: (string zip, string city, string street) → suggest an Address struct/class.
    - Use find_function_definitions to inspect parameter lists.
    - Look for groups of variables that always appear together in function signatures.
    - Suggest creating a Data Class, Struct, or named type.

    ── MENTAL GYMNASTICS (Complexity) ────────────────────────────────────────────

    - Double negatives: if (!isNotReady) → suggest isReady = !isNotReady as named boolean.
    - Nested ternary operators: flag any ternary inside another ternary.
    - Lines with >2 logical operators (&&, ||, !): suggest extracting to a named boolean.
    - Magic numbers: raw numeric literals (86400, 3600, 1024) in logic → suggest named constants
      (SECONDS_IN_A_DAY, SECONDS_IN_AN_HOUR, BYTES_PER_KB).
    - Deeply nested if/else (3+ levels, "Pyramid of Doom") → suggest guard clauses / early returns.
    - Large switch/if-elif chains (5+ branches) → suggest polymorphism or strategy pattern.
    - Mutually exclusive booleans (is_red, is_blue, is_green on same entity) → suggest Enum or Union type.
    - Functions longer than 25 lines → suggest extracting sub-functions.

    ── SPECULATIVE GENERALITY (YAGNI) ────────────────────────────────────────────

    - Abstract classes or interfaces with only a single implementation.
      Use find_function_usages or find_callers to verify single-implementation.
    - Names referencing future features: processDataForFutureV2Api(), handleLegacyAndNewFormat().
    - Unused parameters accepted "for future use."
    - Suggest stripping back to make current intent clear and simple.

    DO NOT report:
    - Language-idiomatic patterns (e.g., single-letter loop vars i, j, k are fine)
    - Conventional short names in tight scopes (e.g., err, ctx, db, tx in Go)
    - Test file naming or test helper naming

    ═══════════════════════════════════════════════════════════════════════════════
    OUTPUT RULES
    ═══════════════════════════════════════════════════════════════════════════════

    - Set category to "smell" (the only allowed category)
    - Reference files by name and line number only — never copy source code into issues
    - Do not repeat patch content in reasoning steps. Keep each reasoning step to 1-2 sentences
    - Empty list if no issues found. No approval text ("LGTM", "looks good")
    - description: ≤25 words, imperative tone, no filler ("Rename X to Y", "Extract Z to constant")
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
        desc="Code smell issues. Category must be 'smell'. "
        "Titles <10 words. Descriptions ≤25 words, imperative. Empty list if none. "
        "File paths must be relative to scope root."
    )


class SmellDetector(dspy.Module):
    """Detects code smells — naming, complexity, data clumps, YAGNI.

    Focuses exclusively on structural quality issues that indicate
    deeper design problems, even when the code is functionally correct.

    MCP tools are scope-restricted: for each scope, tools are rooted at
    repo_path/scope.subroot so the agent cannot access files outside the scope.
    """

    def __init__(self) -> None:
        """Initialize the smell detector."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    async def aforward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for code smells.

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository

        Returns:
            List of code smell issues found across all scopes
        """
        if not self._settings.is_signature_enabled("smell"):
            logger.debug("Skipping smell: disabled")
            return []

        changed_scopes = [s for s in scopes if s.has_changes and s.changed_files]
        if not changed_scopes:
            logger.info("No scopes with changes for smell detection")
            return []

        all_issues: list[Issue] = []
        max_iters = self._settings.get_max_iters("smell")

        total_files = sum(len(s.changed_files) for s in changed_scopes)
        logger.info(
            f"Smell detection for {len(changed_scopes)} scopes "
            f"({total_files} changed files)..."
        )

        for scope in changed_scopes:
            scope_root = resolve_scope_root(repo_path, scope.subroot)
            tools, contexts = await create_mcp_tools(scope_root, "smell_detector")
            try:
                agent = dspy.ReAct(
                    signature=SmellDetectorSignature,
                    tools=tools,
                    max_iters=max_iters,
                )
                scoped = make_scope_relative(scope)
                logger.info(
                    f"  Smell detection: scope {scope.subroot} "
                    f"({len(scope.changed_files)} files)"
                )
                async with SignatureContext("smell", self._cost_tracker):
                    result = await agent.acall(
                        scope=scoped,
                        categories=[IssueCategory.SMELL],
                    )

                issues = [
                    issue for issue in (result.issues or [])
                    if issue.confidence >= MIN_CONFIDENCE
                ]
                restore_repo_paths(issues, scope.subroot)
                all_issues.extend(issues)
                logger.debug(
                    f"  Scope {scope.subroot}: {len(issues)} smell issues"
                )
            except Exception as e:
                logger.error(f"Smell detection failed for scope {scope.subroot}: {e}")
            finally:
                await cleanup_mcp_contexts(contexts)

        logger.info(f"Smell detection found {len(all_issues)} issues")
        return all_issues

    def forward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for code smells (sync wrapper).

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository

        Returns:
            List of code smell issues found across all scopes
        """
        return asyncio.run(self.aforward(scopes, repo_path))