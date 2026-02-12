"""Documentation review module — detects stale or wrong documentation."""

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


class DocReviewSignature(dspy.Signature):
    """Detect stale or wrong documentation caused by code changes in a scope.

    You are a busy Principal Engineer with very little time.
    Be extremely terse. Use imperative mood ("Update X", not "You should update X").
    You have tools to explore the scope's filesystem, search for text, and analyze code.
    All file paths are relative to the scope root directory (the current tool root).
    Tools are restricted to this scope — you cannot access files outside it.

    ═══════════════════════════════════════════════════════════════════════════════
    PHASE 1 — READ DOCUMENTATION (MANDATORY FIRST STEP)
    ═══════════════════════════════════════════════════════════════════════════════

    Use read_file to read the README at the scope root:
    - Path: readme.md OR README.md (paths are relative to scope root)
    - This file is the primary documentation artifact. You MUST read it in full
      before reviewing any changes so you understand what is documented.

    Also search for additional documentation:
    - Use get_tree or file_exists to check for docs/, documentation/, CHANGELOG,
      API docs, .env.example, or any other prose files.
    - Read any doc files that are relevant to the changed code.

    ═══════════════════════════════════════════════════════════════════════════════
    PHASE 2 — CROSS-REFERENCE CHANGES AGAINST DOCUMENTATION
    ═══════════════════════════════════════════════════════════════════════════════

    For each changed file's patch, check whether the change invalidates or
    requires updates to existing documentation. Use search_literal to find
    occurrences of old names, values, or patterns in doc files.

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

    VERIFICATION WORKFLOW:
    1. Read the patch to understand what changed in code
    2. Use search_literal to find references to changed identifiers in doc files
    3. Use read_file to verify the doc content is actually stale
    4. Only report when you have CONFIRMED the documentation is wrong or missing

    DO NOT report:
    - Missing documentation for internal/private functions
    - Style preferences in documentation
    - Documentation that is correct but could be "better"

    ═══════════════════════════════════════════════════════════════════════════════
    OUTPUT RULES
    ═══════════════════════════════════════════════════════════════════════════════

    - Set category to "documentation" (the only allowed category)
    - Reference files by name and line number only — never copy source code into issues
    - Do not repeat patch content in reasoning steps. Keep each reasoning step to 1-2 sentences
    - Empty list if no issues found. No approval text ("LGTM", "looks good")
    - description: ≤25 words, imperative tone, no filler ("Update X section", "Add Y to README")
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
        desc="Documentation issues. Category must be 'documentation'. "
        "Titles <10 words. Descriptions ≤25 words, imperative. Empty list if none. "
        "File paths must be relative to scope root."
    )


class DocReviewer(dspy.Module):
    """Detects stale or wrong documentation caused by code changes.

    Cross-references code changes against README, API docs, config docs,
    and other prose files to find documentation that needs updating.

    MCP tools are scope-restricted: for each scope, tools are rooted at
    repo_path/scope.subroot so the agent cannot access files outside the scope.
    """

    def __init__(self) -> None:
        """Initialize the doc reviewer."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    async def aforward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for documentation issues.

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository

        Returns:
            List of documentation issues found across all scopes
        """
        if not self._settings.is_signature_enabled("doc_review"):
            logger.debug("Skipping doc_review: disabled")
            return []

        changed_scopes = [s for s in scopes if s.has_changes and s.changed_files]
        if not changed_scopes:
            logger.info("No scopes with changes for doc review")
            return []

        all_issues: list[Issue] = []
        max_iters = self._settings.get_max_iters("doc_review")

        total_files = sum(len(s.changed_files) for s in changed_scopes)
        logger.info(
            f"Doc review for {len(changed_scopes)} scopes "
            f"({total_files} changed files)..."
        )

        for scope in changed_scopes:
            scope_root = resolve_scope_root(repo_path, scope.subroot)
            tools, contexts = await create_mcp_tools(scope_root, "doc_reviewer")
            try:
                agent = dspy.ReAct(
                    signature=DocReviewSignature,
                    tools=tools,
                    max_iters=max_iters,
                )
                scoped = make_scope_relative(scope)
                logger.info(
                    f"  Doc review: scope {scope.subroot} "
                    f"({len(scope.changed_files)} files)"
                )
                async with SignatureContext("doc_review", self._cost_tracker):
                    result = await agent.acall(
                        scope=scoped,
                        categories=[IssueCategory.DOCUMENTATION],
                    )

                issues = [
                    issue for issue in (result.issues or [])
                    if issue.confidence >= MIN_CONFIDENCE
                ]
                restore_repo_paths(issues, scope.subroot)
                all_issues.extend(issues)
                logger.debug(
                    f"  Scope {scope.subroot}: {len(issues)} doc issues"
                )
            except Exception as e:
                logger.error(f"Doc review failed for scope {scope.subroot}: {e}")
            finally:
                await cleanup_mcp_contexts(contexts)

        logger.info(f"Doc review found {len(all_issues)} issues")
        return all_issues

    def forward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for documentation issues (sync wrapper).

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository

        Returns:
            List of documentation issues found across all scopes
        """
        return asyncio.run(self.aforward(scopes, repo_path))