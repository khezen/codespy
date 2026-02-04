"""Documentation review module using agentic exploration."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import ModuleContext, get_cost_tracker
from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class DocumentationReviewSignature(dspy.Signature):
    """Review documentation for accuracy based on code changes in a scope.

    You have tools to explore the repository filesystem, search for text, and analyze code.
    All file paths are relative to the repository root.

    TOKEN EFFICIENCY:
    - The patch in each changed_file shows exactly what changed - analyze it FIRST
    - Use search_literal to find references in docs BEFORE reading entire doc files
    - Use read_file only when you've confirmed a doc mentions the changed code
    - Stop exploring once you have enough evidence to confirm or dismiss an issue

    Follow this process:

    1. ANALYZE CODE CHANGES FIRST (from patches):
       - Review each changed_file's patch to understand what changed
       - Identify changed functions, types, APIs from the diff
       - Note any new public APIs or significant changes

    2. SEARCH FOR DOC REFERENCES:
       - Use search_literal to find if any docs reference the changed entities
       - This is more efficient than reading all doc files

    3. VERIFY ISSUES:
       - Use read_file ONLY on docs that reference changed code
       - Check if documentation accurately reflects the changes

    4. CHECK DOCSTRING CONVENTIONS (if needed):
       - Only if adding public APIs, check scope's docstring conventions
       - Use find_function_definitions on one or two existing files
       - Only flag missing docstrings if the scope consistently uses them

    REPORT only:
    - Documentation that references changed code but is now outdated
    - Missing documentation for new public APIs (if scope has doc convention)
    - Broken or stale references you've verified

    IMPORTANT: Only report concrete issues with high confidence.
    Do NOT report speculative issues or issues about code you haven't verified.
    """

    scope: ScopeResult = dspy.InputField(
        desc="Scope with changed files. Has: subroot, scope_type, "
        "changed_files (filename + patch - analyze patch first), language, package_manifest."
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field in Issue objects)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="Documentation issues found. Empty list if documentation is adequate. "
        "Each issue must have: category, severity, title, description, filename, confidence (0.0-1.0)."
    )


class DocumentationReviewer(dspy.Module):
    """Agentic documentation reviewer using ReAct pattern with MCP tools.

    This module uses an LLM agent to explore the repository, find documentation,
    and check if it needs updates based on code changes in each scope.
    """

    category = IssueCategory.DOCUMENTATION
    MODULE_NAME = "doc_reviewer"

    def __init__(self) -> None:
        """Initialize the documentation reviewer."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()

    async def _create_mcp_tools(self, repo_path: Path) -> tuple[list[Any], list[Any]]:
        """Create DSPy tools from MCP servers.

        Args:
            repo_path: Path to the repository root

        Returns:
            Tuple of (tools list, contexts list for cleanup)
        """
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        repo_path_str = str(repo_path)
        caller = "doc_reviewer"
        # Filesystem tools: read_file, list_directory, get_tree, file_exists, get_file_info
        tools.extend(
            await connect_mcp_server(
                tools_dir / "filesystem" / "server.py", [repo_path_str], contexts, caller
            )
        )
        # Ripgrep tools: search_literal, find_function_usages, find_type_usages, etc.
        tools.extend(
            await connect_mcp_server(
                tools_dir / "parsers" / "ripgrep" / "server.py", [repo_path_str], contexts, caller
            )
        )
        # Treesitter tools: find_function_definitions, find_function_calls, etc.
        tools.extend(
            await connect_mcp_server(
                tools_dir / "parsers" / "treesitter" / "server.py", [repo_path_str], contexts, caller
            )
        )
        return tools, contexts

    async def aforward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze documentation across all scopes for issues.

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository (used for MCP tool initialization)

        Returns:
            List of documentation issues found
        """
        # Filter to scopes that have changes
        changed_scopes = [s for s in scopes if s.has_changes and s.changed_files]
        if not changed_scopes:
            logger.info("No scopes with changes to review for documentation")
            return []
        all_issues: list[Issue] = []
        tools, contexts = await self._create_mcp_tools(repo_path)
        try:
            agent = dspy.ReAct(
                signature=DocumentationReviewSignature,
                tools=tools,
                max_iters=15,
            )
            total_files = sum(len(s.changed_files) for s in changed_scopes)
            logger.info(
                f"Reviewing documentation for {len(changed_scopes)} scopes "
                f"({total_files} changed files)..."
            )
            # Use ModuleContext to track costs and timing for this module
            async with ModuleContext(self.MODULE_NAME, self._cost_tracker):
                for scope in changed_scopes:
                    try:
                        result = await agent.acall(
                            scope=scope,
                            category=self.category,
                        )
                        issues = result.issues if result.issues else []
                        # Filter to high-confidence issues
                        filtered_issues = [
                            issue for issue in issues if issue.confidence >= 0.7
                        ]
                        all_issues.extend(filtered_issues)
                        logger.debug(
                            f"  Documentation in {scope.subroot}: {len(filtered_issues)} issues"
                        )
                    except Exception as e:
                        logger.error(f"Documentation review failed for scope {scope.subroot}: {e}")
            logger.info(f"Documentation review found {len(all_issues)} issues")
            return all_issues
        finally:
            await cleanup_mcp_contexts(contexts)

    def forward(self, scopes: Sequence[ScopeResult], repo_path: Path) -> list[Issue]:
        """Analyze documentation (sync wrapper).

        Args:
            scopes: List of identified scopes with their changed files
            repo_path: Path to the cloned repository (used for MCP tool initialization)

        Returns:
            List of documentation issues found
        """
        return asyncio.run(self.aforward(scopes, repo_path))