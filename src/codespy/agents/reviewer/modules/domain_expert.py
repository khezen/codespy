"""Domain expert module for codebase-aware review using MCP tools."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.agents.reviewer.modules.helpers import is_speculative, MIN_CONFIDENCE
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class DomainExpertSignature(dspy.Signature):
    """Deep codebase analysis for architecture, patterns, and style consistency.

    You are a domain expert with access to the full codebase through exploration tools.
    Your goal is to deeply understand each scope and then review changes for consistency.

    AVAILABLE TOOLS:
    - Filesystem: read_file, list_directory, get_tree, file_exists, get_file_info
    - Search: find_function_usages, find_type_usages, find_imports_of, find_callers, search_literal
    - AST: find_function_definitions, find_function_calls, find_all_calls_in_file

    EXPLORATION STRATEGY FOR EACH SCOPE:
    1. UNDERSTAND THE SCOPE PURPOSE:
       - Use get_tree to see the directory structure of the scope's subroot
       - Read key files: README, main entry points, interfaces/APIs
       - Identify the scope's responsibility and boundaries

    2. ANALYZE IMPLEMENTATION PATTERNS:
       - Use find_function_definitions to understand the code structure
       - Use find_function_usages and find_callers to trace data/control flow
       - Identify design patterns (Factory, Repository, Service, etc.)
       - Look for dependency injection, error handling patterns

    3. IDENTIFY CODE STYLE AND CONVENTIONS:
       - Variable/function naming conventions (camelCase, snake_case, etc.)
       - File organization patterns (one class per file, feature folders, etc.)
       - Import organization and ordering
       - Error handling approach (exceptions, result types, error codes)
       - Logging and documentation style
       - Test file organization and naming

    4. REVIEW THE CHANGES:
       For each changed file in the scope, analyze:
       - Does the change align with the scope's overall purpose?
       - Is it consistent with existing implementation patterns?
       - Does it follow the established code style?
       - Are naming conventions respected?
       - Does error handling match the existing approach?
       - Are there missing updates to related files?

    WHAT TO REPORT:
    - Architectural inconsistencies (pattern violations, wrong layer for logic)
    - Style violations (naming, organization, formatting inconsistencies)
    - Missing related changes (if pattern X requires Y, but Y is missing)
    - Design pattern misuse or violations
    - API contract changes that break consistency
    - Code that doesn't fit the scope's responsibility

    CRITICAL RULES:
    - EXPLORE the codebase before making judgments
    - BASE all findings on EVIDENCE from the tools
    - COMPARE changes to actual existing code, not assumptions
    - QUALITY over quantity: only report verified issues
    - Include specific examples from the codebase to support findings
    """

    scopes: list[ScopeResult] = dspy.InputField(
        desc="List of identified scopes with their changed files. Each scope has: "
        "subroot (relative path), scope_type, changed_files list (with patch diffs), language, package_manifest."
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="VERIFIED issues based on codebase exploration. Empty list if none. "
        "Each issue must cite evidence from exploration."
    )


class DomainExpert(dspy.Module):
    """Agentic domain expert using ReAct pattern with MCP tools.

    This module uses an LLM agent to explore the codebase structure,
    understand design patterns and code style, and then review changes
    for consistency and architectural fit.
    """

    category = IssueCategory.CONTEXT

    def __init__(self) -> None:
        """Initialize the domain expert."""
        super().__init__()

    async def _create_mcp_tools(self, repo_path: Path) -> tuple[list[Any], list[Any]]:
        """Create DSPy tools from MCP servers for codebase exploration."""
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        repo_path_str = str(repo_path)
        # Filesystem tools for reading and navigating code
        tools.extend(await connect_mcp_server(
            tools_dir / "filesystem" / "mcp.py",
            [repo_path_str],
            contexts
        ))
        # Ripgrep tools for searching patterns and usages
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "ripgrep" / "mcp.py",
            [repo_path_str],
            contexts
        ))
        # Tree-sitter tools for AST analysis
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "treesitter" / "mcp.py",
            [repo_path_str],
            contexts
        ))

        return tools, contexts

    async def aforward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for architectural consistency and style compliance.

        Args:
            scopes: List of ScopeResult from ScopeIdentifier with changed files
            repo_path: Path to the cloned repository

        Returns:
            List of contextual issues found through deep codebase analysis
        """
        # Filter to scopes with changes
        changed_scopes = [s for s in scopes if s.has_changes and s.changed_files]

        if not changed_scopes:
            logger.debug("No scopes with changes to analyze for context")
            return []

        tools, contexts = await self._create_mcp_tools(repo_path)

        try:
            agent = dspy.ReAct(
                signature=DomainExpertSignature,
                tools=tools,
                max_iters=30,  # Allow more iterations for deep exploration
            )
            total_files = sum(len(s.changed_files) for s in changed_scopes)
            logger.info(
                f"Domain expert analyzing {len(changed_scopes)} scopes "
                f"({total_files} changed files)..."
            )
            result = await agent.acall(
                scopes=changed_scopes,
                category=self.category,
            )
            issues = result.issues if result.issues else []
            # Filter issues by confidence and speculation
            filtered_issues = [
                issue for issue in issues
                if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
            ]
            logger.info(
                f"Domain expert found {len(filtered_issues)} issues "
                f"(filtered from {len(issues)} raw issues)"
            )
            return filtered_issues
        except Exception as e:
            logger.error(f"Error in domain expert analysis: {e}")
            return []
        finally:
            await cleanup_mcp_contexts(contexts)

    def forward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for architectural consistency and style compliance.

        Args:
            scopes: List of ScopeResult from ScopeIdentifier with changed files
            repo_path: Path to the cloned repository

        Returns:
            List of contextual issues found through deep codebase analysis
        """
        return asyncio.run(self.aforward(scopes, repo_path))