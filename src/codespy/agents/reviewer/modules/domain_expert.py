"""Domain expert module for codebase-aware review using MCP tools."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.agents.reviewer.modules.helpers import is_speculative, MIN_CONFIDENCE
from codespy.config import get_settings
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class DomainExpertSignature(dspy.Signature):
    """Deep codebase analysis for business logic, architecture, patterns, and style consistency.

    You are a domain expert with access to the full codebase through exploration tools.
    Your goal is to deeply understand the scope's business purpose and then review changes
    for both business fit and technical consistency.

    TOKEN EFFICIENCY:
    - The patch in each changed_file shows exactly what changed - analyze it FIRST
    - Use targeted searches (find_function_usages, search_literal) before reading entire files
    - Use read_file ONLY when you need broader context not visible in patch or search results
    - Stop exploring once you have enough evidence to confirm or dismiss an issue

    AVAILABLE TOOLS:
    - Filesystem: read_file, list_directory, get_tree, file_exists, get_file_info
    - Search: find_function_usages, find_type_usages, find_imports_of, find_callers, search_literal
    - AST: find_function_definitions, find_function_calls, find_all_calls_in_file

    EXPLORATION STRATEGY:

    PHASE 1 - UNDERSTAND BUSINESS PURPOSE:
    - Use get_tree to see the directory structure of the scope's subroot
    - Search for documentation files (README.md, ARCHITECTURE.md, DESIGN.md) in the scope
    - Read main entry points, public APIs/interfaces
    - Identify what problem or feature this scope solves
    - Understand the domain concepts (entities, workflows, business rules)
    - Map the scope's relationships to other parts of the system
    - Identify the scope's responsibilities and boundaries

    PHASE 2 - ANALYZE IMPLEMENTATION PATTERNS:
    - Use find_function_definitions to understand the code structure
    - Use find_function_usages and find_callers to trace data/control flow
    - Identify design patterns (Factory, Repository, Service, etc.)
    - Look for dependency injection, error handling patterns
    - Variable/function naming conventions (camelCase, snake_case, etc.)
    - File organization patterns (one class per file, feature folders, etc.)
    - Import organization and ordering
    - Error handling approach (exceptions, result types, error codes)
    - Logging and documentation style

    PHASE 3 - REVIEW CHANGES FOR BUSINESS AND TECHNICAL FIT:
    For each changed file in the scope, analyze:

    Business Fit:
    - Do the changes align with the scope's business purpose?
    - Do they respect domain boundaries (logic belongs here vs elsewhere)?
    - Are naming choices consistent with domain language/terminology?
    - Do they break or change expected business behavior?
    - Are business validations present where needed?
    - Do changes to domain contracts/APIs make business sense?

    Technical Fit:
    - Is it consistent with existing implementation patterns?
    - Does it follow the established code style?
    - Are naming conventions respected?
    - Does error handling match the existing approach?
    - Are there missing updates to related files?

    WHAT TO REPORT:
    - Business logic inconsistencies (changes that contradict scope's purpose)
    - Domain boundary violations (logic that belongs in a different scope)
    - Domain naming violations (terms that don't fit the domain language)
    - Missing business validations or rule enforcement
    - Breaking changes to domain contracts/APIs
    - Architectural inconsistencies (pattern violations, wrong layer for logic)
    - Style violations (naming, organization, formatting inconsistencies)
    - Missing related changes (if pattern X requires Y, but Y is missing)
    - Design pattern misuse or violations

    CRITICAL RULES:
    - EXPLORE the codebase before making judgments
    - UNDERSTAND the business purpose first, then review technically
    - BASE all findings on EVIDENCE from the tools
    - COMPARE changes to actual existing code, not assumptions
    - QUALITY over quantity: only report verified issues
    - Include specific examples from the codebase to support findings
    """

    scope: ScopeResult = dspy.InputField(
        desc="The scope to analyze with its changed files. Contains: "
        "subroot (relative path), scope_type, changed_files list (filename + patch only - use read_file for content), language, package_manifest."
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
    understand business purpose and design patterns, and then review changes
    for business fit and technical consistency.
    """

    category = IssueCategory.CONTEXT

    def __init__(self) -> None:
        """Initialize the domain expert."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    async def _create_mcp_tools(self, repo_path: Path) -> tuple[list[Any], list[Any]]:
        """Create DSPy tools from MCP servers for codebase exploration."""
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        repo_path_str = str(repo_path)
        caller = "domain_expert"
        # Filesystem tools for reading and navigating code
        tools.extend(await connect_mcp_server(
            tools_dir / "filesystem" / "server.py",
            [repo_path_str],
            contexts,
            caller,
        ))
        # Ripgrep tools for searching patterns and usages
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "ripgrep" / "server.py",
            [repo_path_str],
            contexts,
            caller,
        ))
        # Tree-sitter tools for AST analysis
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "treesitter" / "server.py",
            [repo_path_str],
            contexts,
            caller,
        ))

        return tools, contexts

    async def aforward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for business fit and technical consistency.

        Args:
            scopes: List of ScopeResult from ScopeIdentifier with changed files
            repo_path: Path to the cloned repository

        Returns:
            List of contextual issues found through deep codebase analysis
        """
        # Check if signature is enabled
        if not self._settings.is_signature_enabled("domain_analysis"):
            logger.debug("Skipping domain_analysis: disabled")
            return []

        # Filter to scopes with changes
        changed_scopes = [s for s in scopes if s.has_changes and s.changed_files]

        if not changed_scopes:
            logger.debug("No scopes with changes to analyze for context")
            return []

        tools, contexts = await self._create_mcp_tools(repo_path)
        all_issues: list[Issue] = []

        try:
            # Get max_iters from signature config
            max_iters = self._settings.get_max_iters("domain_analysis")

            agent = dspy.ReAct(
                signature=DomainExpertSignature,
                tools=tools,
                max_iters=max_iters,
            )

            for scope in changed_scopes:
                file_count = len(scope.changed_files)
                logger.info(
                    f"Domain expert analyzing scope '{scope.subroot}' "
                    f"({file_count} changed files)..."
                )

                try:
                    # Track domain_analysis signature costs
                    async with SignatureContext("domain_analysis", self._cost_tracker):
                        result = await agent.acall(
                            scope=scope,
                            category=self.category,
                        )
                    issues = result.issues if result.issues else []
                    # Filter issues by confidence and speculation
                    filtered_issues = [
                        issue for issue in issues
                        if issue.confidence >= MIN_CONFIDENCE #and not is_speculative(issue)
                    ]
                    all_issues.extend(filtered_issues)
                    logger.info(
                        f"  Found {len(filtered_issues)} issues in scope '{scope.subroot}' "
                        f"(filtered from {len(issues)} raw issues)"
                    )
                except Exception as e:
                    logger.error(f"Error analyzing scope '{scope.subroot}': {e}")

            logger.info(f"Domain expert found {len(all_issues)} total issues")
            return all_issues
        except Exception as e:
            logger.error(f"Error in domain expert analysis: {e}")
            return []
        finally:
            await cleanup_mcp_contexts(contexts)

    def forward(
        self, scopes: Sequence[ScopeResult], repo_path: Path
    ) -> list[Issue]:
        """Analyze scopes for business fit and technical consistency.

        Args:
            scopes: List of ScopeResult from ScopeIdentifier with changed files
            repo_path: Path to the cloned repository

        Returns:
            List of contextual issues found through deep codebase analysis
        """
        return asyncio.run(self.aforward(scopes, repo_path))