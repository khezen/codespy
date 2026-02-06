"""Documentation review module using agentic exploration."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.config import get_settings
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class DocumentationReviewSignature(dspy.Signature):
    """Review documentation for accuracy based on code changes in a scope.

    You have tools to explore the repository filesystem, search for text, and analyze code.
    All file paths are relative to the repository root.

    ═══════════════════════════════════════════════════════════════════════════════
    MANDATORY FIRST STEP - DO THIS BEFORE ANYTHING ELSE:
    ═══════════════════════════════════════════════════════════════════════════════
    
    Use read_file to read the README at the scope root:
    - Path: {scope.subroot}/readme.md OR {scope.subroot}/README.md
    - If scope.subroot is "peaks/svc/authenticator-v1", read "peaks/svc/authenticator-v1/readme.md"
    - This file contains documentation that MUST be checked against code changes
    
    If README doesn't exist at scope root, check scope.doc_paths for alternatives.
    
    ═══════════════════════════════════════════════════════════════════════════════

    Follow this process:

    1. READ THE README (MANDATORY):
       - Call read_file("{scope.subroot}/readme.md") to get the documentation
       - Look for: API endpoints, request/response examples, error codes, usage examples
       - This is where you'll find outdated documentation that needs updating

    2. ANALYZE THE DIFF FOR CHANGE TYPES:
       First, categorize what changed in the diff to guide your documentation search:
       - HTTP/API changes: Handler files, response structs, status codes, Content-Type
       - Function signature changes: Parameters added/removed/renamed, return types changed
       - Configuration changes: Config structs, environment variables, defaults
       - Data model changes: Structs, fields, types, validation
       - Error handling changes: New error types, removed errors, status code changes
       - CLI changes: Commands, flags, arguments
       - Client library changes: Client methods, return types
       - check for missing docstrings, ONLY if the scope consistently uses them

    3. VERIFY DOCUMENTATION AGAINST SPECIFIC CHANGE TYPES:
       
       - The scope.doc_paths field contains documentation files/directories found by scope identifier

       HTTP/API CHANGES (CRITICAL - high miss rate):
       - Content-Type changes (text/plain → application/json) - check documented examples
       - HTTP status code changes (e.g., 428 error → 202 success) - update all references
       - Response body structure changes - verify documented examples match
       - New response fields added - ensure they're documented
       - Search docs for endpoint paths (e.g., /api/v1/login) to find all examples

       FUNCTION/METHOD SIGNATURE CHANGES:
       - Parameters added/removed/renamed → Check if docs reference old signatures
       - Return type changes (e.g., []byte → *Struct) → Update all examples
       - New public functions → Ensure documented if scope has doc conventions
       - Search for function names in markdown files to find usage examples

       CONFIGURATION & ENVIRONMENT VARIABLES:
       - New config fields → Check README Configuration section or .env.example
       - Removed/renamed fields → Search docs for old field names
       - Default value changes → Verify docs reflect new defaults
       - New environment variables → Check if documented with description

       ERROR TYPES & CODES:
       - New error types/codes → Check if error documentation lists them
       - Removed error types → Verify old errors aren't still documented
       - Error behavior changes (error → success response) → BREAKING, must document
       - HTTP status code semantics change → Update API docs

       DATA MODELS & STRUCTS:
       - New fields in request/response structs → Update API examples
       - Removed fields → Search docs for references to old fields
       - Field type changes → Update examples
       - Required/optional field changes → Update documentation

       CLIENT LIBRARY/SDK CHANGES:
       - New methods → Document with usage examples
       - Changed method signatures → Update code examples  
       - Return type changes (e.g., string → *Struct) → BREAKING, update all examples
       - Deprecations → Add deprecation notices

       CLI COMMANDS & FLAGS:
       - New commands/flags → Add to CLI reference
       - Removed/renamed flags → Search docs for old flag names
       - Changed default values → Update documentation

       CONSTANTS & ENUMS:
       - New enum values → Document new valid values
       - Removed values → Check if docs reference removed values

    4. BREAKING CHANGES - ALWAYS FLAG DOCUMENTATION UPDATES:
       - Public function signatures change
       - Return types change from primitive to struct (or vice versa)
       - Required parameters added
       - Response format changes (plain text → JSON)
       - HTTP status code semantics change (error → success)
       - Error behavior changes

    REPORT only:
    - Documentation that references changed code but is now outdated
    - Missing documentation for new public APIs (if scope has doc convention)
    - Broken or stale references you've verified
    - Breaking changes that require documentation updates

    IMPORTANT: Only report concrete issues with high confidence.
    Do NOT report speculative issues or issues about code you haven't verified.
    """

    scope: ScopeResult = dspy.InputField(
        desc="Scope with changed files. Has: subroot, scope_type, "
        "changed_files (filename + patch - analyze patch first), language, package_manifest, "
        "doc_paths (pre-identified documentation files/directories - check these first)."
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
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

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
        # Check if signature is enabled
        if not self._settings.is_signature_enabled("doc_review"):
            logger.debug("Skipping doc_review: disabled")
            return []

        # Filter to scopes that have changes
        changed_scopes = [s for s in scopes if s.has_changes and s.changed_files]
        if not changed_scopes:
            logger.info("No scopes with changes to review for documentation")
            return []
        all_issues: list[Issue] = []
        tools, contexts = await self._create_mcp_tools(repo_path)
        try:
            # Get max_iters from signature config
            max_iters = self._settings.get_max_iters("doc_review")
            agent = dspy.ReAct(
                signature=DocumentationReviewSignature,
                tools=tools,
                max_iters=max_iters,
            )
            total_files = sum(len(s.changed_files) for s in changed_scopes)
            logger.info(
                f"Reviewing documentation for {len(changed_scopes)} scopes "
                f"({total_files} changed files)..."
            )
            for scope in changed_scopes:
                try:
                    # Log doc_paths for debugging
                    logger.info(
                        f"  Scope {scope.subroot}: doc_paths={scope.doc_paths}, "
                        f"checking {scope.subroot}/readme.md"
                    )
                    # Track doc_review signature costs
                    async with SignatureContext("doc_review", self._cost_tracker):
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