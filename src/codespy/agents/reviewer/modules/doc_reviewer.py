"""Documentation review module using agentic exploration."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class DocumentationReviewSignature(dspy.Signature):
    """Review documentation for accuracy based on code changes in each scope.

    You have tools to explore the repository filesystem, search for text, and analyze code.
    All file paths are relative to the repository root.

    FOR EACH SCOPE, follow this process:

    1. EXPLORE DOCUMENTATION:
       - Use get_tree to explore the scope's subroot directory
       - Look for: README.md, docs/, *.md files, CHANGELOG, API docs
       - Also check project root (".") for top-level documentation

    2. ANALYZE CODE CHANGES:
       - Review changed_files list for each scope
       - Use find_function_definitions to understand what code entities changed
       - Identify changed functions, types, APIs, configuration options

    3. CHECK DOCUMENTATION COVERAGE:
       - Use read_file to read documentation content
       - Use search_literal to find references to changed code in docs
       - Determine if documentation mentions the changed entities

    4. CHECK DOCSTRING/COMMENT CONVENTIONS:
       - Before reporting missing docstrings/comments, check scope's conventions
       - Use read_file on a few existing files in the scope
       - Only flag missing docstrings if the scope consistently uses them

    5. REPORT ISSUES:
       - Documentation that references changed code but may be outdated
       - Missing documentation for new public APIs/functions
       - Missing docstrings ONLY if scope has docstring convention
       - Broken or potentially stale references

    IMPORTANT: Only report concrete issues with high confidence.
    Do NOT report speculative issues or issues about code you haven't verified.
    """

    scopes: list[ScopeResult] = dspy.InputField(
        desc="List of identified scopes with their changed files. Each scope has: "
        "subroot (relative path), scope_type, changed_files list, language, package_manifest."
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

    def __init__(self) -> None:
        """Initialize the documentation reviewer."""
        super().__init__()

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
        # Filesystem tools: read_file, list_directory, get_tree, file_exists, get_file_info
        tools.extend(
            await connect_mcp_server(
                tools_dir / "filesystem" / "server.py", [repo_path_str], contexts
            )
        )
        # Ripgrep tools: search_literal, find_function_usages, find_type_usages, etc.
        tools.extend(
            await connect_mcp_server(
                tools_dir / "parsers" / "ripgrep" / "server.py", [repo_path_str], contexts
            )
        )
        # Treesitter tools: find_function_definitions, find_function_calls, etc.
        tools.extend(
            await connect_mcp_server(
                tools_dir / "parsers" / "treesitter" / "server.py", [repo_path_str], contexts
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
        tools, contexts = await self._create_mcp_tools(repo_path)
        try:
            agent = dspy.ReAct(
                signature=DocumentationReviewSignature,
                tools=tools,
                max_iters=30,
            )
            total_files = sum(len(s.changed_files) for s in changed_scopes)
            logger.info(
                f"Reviewing documentation for {len(changed_scopes)} scopes "
                f"({total_files} changed files)..."
            )
            result = await agent.acall(
                scopes=changed_scopes,
                category=self.category,
            )
            issues = result.issues if result.issues else []
            # Filter to high-confidence issues
            filtered_issues = [
                issue for issue in issues if issue.confidence >= 0.7
            ]
            logger.info(
                f"Documentation review found {len(filtered_issues)} issues "
                f"(filtered from {len(issues)} raw issues)"
            )
            return filtered_issues
        except Exception as e:
            logger.error(f"Documentation review agent failed: {e}")
            return []
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