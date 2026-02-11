"""Merged code defect detection and documentation review module."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.agents.reviewer.modules.helpers import MIN_CONFIDENCE, make_scope_relative, resolve_scope_root, restore_repo_paths
from codespy.config import get_settings
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class CodeAndDocReviewSignature(dspy.Signature):
    """Detect VERIFIED code defects AND stale/wrong documentation in a scope.

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
      configuration, and expected behavior. Use it to inform BOTH documentation
      review AND defect detection.

    If README doesn't exist at scope root, search for alternative documentation files
    (e.g., docs/, documentation/, README.rst) using get_tree or file_exists.

    ═══════════════════════════════════════════════════════════════════════════════
    PHASE 2 — ANALYZE CHANGES FOR DEFECTS AND DOCUMENTATION ISSUES
    ═══════════════════════════════════════════════════════════════════════════════

    Using the README context and the patches in scope.changed_files, look for:

    ── A. CODE DEFECTS (category = "bug" or "security") ──────────────────────────

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

    ── B. DOCUMENTATION ISSUES (category = "documentation") ──────────────────────

    Verify documentation against the specific changes in this scope:

    HTTP/API CHANGES (CRITICAL — high miss rate):
    - Content-Type changes → check documented examples
    - HTTP status code changes → update all references
    - Response body structure changes → verify documented examples match
    - New response fields → ensure documented
    - Search docs for endpoint paths to find all examples

    FUNCTION/METHOD SIGNATURE CHANGES:
    - Parameters added/removed/renamed → check if docs reference old signatures
    - Return type changes → update all examples
    - New public functions → ensure documented if scope has doc conventions

    CONFIGURATION & ENVIRONMENT VARIABLES:
    - New config fields → check README Configuration section or .env.example
    - Removed/renamed fields → search docs for old field names
    - Default value changes → verify docs reflect new defaults

    ERROR TYPES & CODES:
    - New/removed error types → check error documentation
    - Error behavior changes (error → success) → BREAKING, must document
    - HTTP status code semantics change → update API docs

    DATA MODELS & STRUCTS:
    - New/removed fields in request/response structs → update API examples
    - Field type changes → update examples

    CLI COMMANDS & FLAGS:
    - New commands/flags → add to CLI reference
    - Removed/renamed flags → search docs for old names

    ── C. CODE SMELLS (category = "smell") ───────────────────────────────────────

    Code smells signal deeper structural rot even when tests pass. Focus on
    semantics, intent clarity, and whether the code "makes sense" to a reader.

    VERIFICATION WORKFLOW for smells:
    1. Review the patch for suspicious names, parameter lists, or control flow
    2. Use find_function_definitions to check function signatures and body length
    3. Use find_function_calls / find_callers to verify usage patterns
    4. Use search_literal to find related constants, sibling implementations, etc.
    5. Only report smells you can CONFIRM with evidence from the tools

    ── C1. UNCOMMUNICATIVE NAMES (Semantic Naming) ──

    - Variables MUST be nouns describing the data they hold.
      Flag: data, info, item, rs, list1, temp, val, x in non-trivial contexts.
      Compare the variable name against its type or source to verify mismatch.
    - Functions MUST start with a verb indicating the action.
      Flag: user(), process(), handle() with no object. Suggest: calculateUserTax(), fetchActiveUsers().
    - Booleans MUST read as predicates (answer a yes/no question).
      Flag: valid, flag, status, ready. Suggest: isEmailFormatValid, hasPermission, shouldRetry.
    - Side-effect mismatch: a function named getName() that also writes to a database,
      or getData() that deletes records. Use find_function_definitions to read the body.

    ── C2. PRIMITIVE OBSESSION (Data Clumps) ──

    - Flag functions taking 3+ related primitive parameters that travel together.
      Example: (string zip, string city, string street) → suggest an Address struct/class.
    - Use find_function_definitions to inspect parameter lists.
    - Look for groups of variables that always appear together in function signatures.
    - Suggest creating a Data Class, Struct, or named type.

    ── C3. MENTAL GYMNASTICS (Complexity) ──

    - Double negatives: if (!isNotReady) → suggest isReady = !isNotReady as named boolean.
    - Nested ternary operators: flag any ternary inside another ternary.
    - Lines with >2 logical operators (&&, ||, !): suggest extracting to a named boolean.
    - Magic numbers: raw numeric literals (86400, 3600, 1024) in logic → suggest named constants
      (SECONDS_IN_A_DAY, SECONDS_IN_AN_HOUR, BYTES_PER_KB).
    - Deeply nested if/else (3+ levels, "Pyramid of Doom") → suggest guard clauses / early returns.
    - Large switch/if-elif chains (5+ branches) → suggest polymorphism or strategy pattern.
    - Mutually exclusive booleans (is_red, is_blue, is_green on same entity) → suggest Enum or Union type.
    - Functions longer than 25 lines → suggest extracting sub-functions.

    ── C4. SPECULATIVE GENERALITY (YAGNI) ──

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
        desc="Verified defects and documentation issues. Category must be one of the provided categories. "
        "Titles <10 words. Descriptions ≤25 words, imperative. Empty list if none. "
        "File paths must be relative to scope root."
    )


class CodeAndDocReviewer(dspy.Module):
    """Detects code defects and documentation issues in a single agentic pass.

    Merges the responsibilities of DefectDetector and DocumentationReviewer
    to share MCP tool overhead and README context across both concerns.

    MCP tools are scope-restricted: for each scope, tools are rooted at
    repo_path/scope.subroot so the agent cannot access files outside the scope.
    """

    def __init__(self) -> None:
        """Initialize the code and doc reviewer."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    async def _create_mcp_tools(self, scope_root: Path) -> tuple[list[Any], list[Any]]:
        """Create DSPy tools from MCP servers, rooted at scope directory.

        Args:
            scope_root: Path to the scope root directory (repo_path / scope.subroot)

        Returns:
            Tuple of (tools list, contexts list for cleanup)
        """
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        scope_root_str = str(scope_root)
        caller = "code_and_doc_reviewer"
        # Filesystem tools: read_file, list_directory, get_tree, file_exists, get_file_info
        tools.extend(
            await connect_mcp_server(
                tools_dir / "filesystem" / "server.py", [scope_root_str], contexts, caller
            )
        )
        # Ripgrep tools: search_literal, find_function_usages, find_type_usages, etc.
        tools.extend(
            await connect_mcp_server(
                tools_dir / "parsers" / "ripgrep" / "server.py", [scope_root_str], contexts, caller
            )
        )
        # Treesitter tools: find_function_definitions, find_function_calls, etc.
        tools.extend(
            await connect_mcp_server(
                tools_dir / "parsers" / "treesitter" / "server.py", [scope_root_str], contexts, caller
            )
        )
        return tools, contexts

    async def aforward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for code defects and documentation issues.

        For each scope, MCP tools are created rooted at repo_path/scope.subroot
        so the agent can only access files within the scope boundary. This prevents
        unnecessary out-of-scope tool calls that waste tokens and cost.

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository

        Returns:
            List of issues (bugs, security, documentation) found across all scopes
        """
        if not self._settings.is_signature_enabled("code_and_doc_review"):
            logger.debug("Skipping code_and_doc_review: disabled")
            return []

        changed_scopes = [s for s in scopes if s.has_changes and s.changed_files]
        if not changed_scopes:
            logger.info("No scopes with changes to review")
            return []

        all_issues: list[Issue] = []
        max_iters = self._settings.get_max_iters("code_and_doc_review")

        total_files = sum(len(s.changed_files) for s in changed_scopes)
        logger.info(
            f"Reviewing code and docs for {len(changed_scopes)} scopes "
            f"({total_files} changed files)..."
        )

        for scope in changed_scopes:
            # Scope-restrict MCP tools to the scope's subroot directory
            scope_root = resolve_scope_root(repo_path, scope.subroot)
            tools, contexts = await self._create_mcp_tools(scope_root)
            try:
                agent = dspy.ReAct(
                    signature=CodeAndDocReviewSignature,
                    tools=tools,
                    max_iters=max_iters,
                )
                # Create scope-relative copy so file paths match the scoped tool root
                scoped = make_scope_relative(scope)
                logger.info(
                    f"  Reviewing scope {scope.subroot} "
                    f"({len(scope.changed_files)} files)"
                )
                async with SignatureContext("code_and_doc_review", self._cost_tracker):
                    result = await agent.acall(
                        scope=scoped,
                        categories=[IssueCategory.BUG, IssueCategory.SECURITY, IssueCategory.DOCUMENTATION, IssueCategory.SMELL],
                    )

                issues = [
                    issue for issue in (result.issues or [])
                    if issue.confidence >= MIN_CONFIDENCE
                ]
                # Restore repo-root-relative paths in reported issues
                restore_repo_paths(issues, scope.subroot)
                all_issues.extend(issues)
                logger.debug(
                    f"  Scope {scope.subroot}: {len(issues)} issues"
                )
            except Exception as e:
                logger.error(f"Review failed for scope {scope.subroot}: {e}")
            finally:
                await cleanup_mcp_contexts(contexts)

        logger.info(f"Code and doc review found {len(all_issues)} issues")
        return all_issues

    def forward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for defects and documentation issues (sync wrapper).

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository

        Returns:
            List of issues found across all scopes
        """
        return asyncio.run(self.aforward(scopes, repo_path))