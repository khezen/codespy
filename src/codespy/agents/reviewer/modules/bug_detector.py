"""Bug detection module with code exploration tools."""

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


class BugDetectionSignature(dspy.Signature):
    """Detect VERIFIED bugs and logic errors in code changes.

    You are an expert software engineer reviewing code for bugs.
    You have access to tools that let you explore the codebase to VERIFY your findings.

    INPUT:
    - scope: A ScopeResult containing:
      * subroot: Path relative to repo root (e.g., "packages/auth")
      * scope_type: Type of scope (library, service, application, script)
      * language: Primary programming language
      * changed_files: List of ChangedFile objects, each with:
        - filename: Path to the file
        - patch: The diff showing exactly what changed (additions/deletions)
      * package_manifest: Package manifest info if present
      * artifacts: Security-relevant artifacts (Dockerfiles, etc.)

    TOKEN EFFICIENCY:
    - The patch shows exactly what changed - analyze it FIRST before using tools
    - Use read_file ONLY when you need context outside the diff (e.g., checking base class, error handling)
    - Prefer targeted searches (find_function_definitions, find_callers) over reading entire files
    - Stop exploring once you have enough evidence to confirm or dismiss an issue

    CRITICAL RULES:
    - Analyze ALL changed files in the scope
    - ONLY report bugs you can VERIFY using the available tools
    - Before reporting any bug, USE the tools to verify your assumptions
    - DO NOT speculate about potential issues you cannot verify
    - DO NOT report "might be", "could be", "possibly", "may cause" issues
    - If you cannot point to the EXACT buggy code with evidence, do NOT report it
    - Quality over quantity: prefer 0 reports over 1 speculative report

    VERIFICATION WORKFLOW:
    1. Review each changed file's patch in scope.changed_files - the diff shows what changed
    2. For suspected bugs that need verification beyond the patch:
       - Use find_function_definitions to check function signatures and implementations
       - Use find_function_calls to understand how functions are called
       - Use find_function_usages/find_callers to trace usage patterns
       - Use read_file ONLY if you need broader context not visible in the patch
    3. Only report issues that are CONFIRMED by your verification

    CONCRETE bugs to look for (with verification):
    - Logic errors: verify the condition is actually incorrect by checking related code
    - Null/undefined references: verify the check is actually missing by reading the code
    - Resource leaks: verify there's no cleanup in finally/defer/close methods
    - Error handling: verify errors aren't handled elsewhere in the call chain
    - Type mismatches: verify types by checking definitions
    - Off-by-one errors: verify by understanding the data structure bounds

    DO NOT report:
    - Style issues or minor improvements
    - Hypothetical edge cases without evidence
    - Issues that might exist in code you haven't verified
    """

    scope: ScopeResult = dspy.InputField(
        desc="Full scope context including all changed files, scope type, subroot, and language"
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="VERIFIED bugs found across all changed files in scope. Empty list if none confirmed."
    )


class BugDetector(dspy.Module):
    """Detects bugs and logic errors using DSPy with code exploration tools."""

    category = IssueCategory.BUG

    def __init__(self) -> None:
        """Initialize the bug detector."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    async def _create_mcp_tools(self, repo_path: Path) -> tuple[list[Any], list[Any]]:
        """Create DSPy tools from filesystem and parser MCP servers.

        Args:
            repo_path: Path to the repository root

        Returns:
            Tuple of (tools, contexts) for cleanup
        """
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        repo_path_str = str(repo_path)
        caller = "bug_detector"
        # Add filesystem tools for reading files and exploring structure
        tools.extend(await connect_mcp_server(
            tools_dir / "filesystem" / "server.py",
            [repo_path_str],
            contexts,
            caller,
        ))
        # Add tree-sitter tools for parsing code structure
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "treesitter" / "server.py",
            [repo_path_str],
            contexts,
            caller,
        ))
        # Add ripgrep tools for searching code patterns
        tools.extend(await connect_mcp_server(
            tools_dir / "parsers" / "ripgrep" / "server.py",
            [repo_path_str],
            contexts,
            caller,
        ))
        return tools, contexts

    async def aforward(self, scopes: Sequence[ScopeResult], repo_path: Path) -> list[Issue]:
        """Analyze scopes for bugs and return issues.

        Args:
            scopes: The scopes containing changed files to analyze
            repo_path: Path to the cloned repository for code exploration

        Returns:
            List of bug issues found across all scopes
        """
        # Check if signature is enabled
        if not self._settings.is_signature_enabled("bug_detection"):
            logger.debug("Skipping bug_detection: disabled")
            return []

        all_issues: list[Issue] = []
        tools, contexts = await self._create_mcp_tools(repo_path)

        # Get max_iters from signature config
        max_iters = self._settings.get_max_iters("bug_detection")

        # Create ReAct agent with code exploration tools
        bug_detection_agent = dspy.ReAct(
            signature=BugDetectionSignature,
            tools=tools,
            max_iters=max_iters,
        )
        try:
            for scope in scopes:
                if not scope.changed_files:
                    logger.debug(f"Skipping scope {scope.subroot}: no changed files")
                    continue
                try:
                    logger.debug(f"Analyzing scope {scope.subroot} with {len(scope.changed_files)} files")
                    # Track bug_detection signature costs
                    async with SignatureContext("bug_detection", self._cost_tracker):
                        result = await bug_detection_agent.acall(
                            scope=scope,
                            category=self.category,
                        )
                    issues = [
                        issue for issue in result.issues
                        if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
                    ]
                    all_issues.extend(issues)
                    logger.debug(f"  Bugs in scope {scope.subroot}: {len(issues)} issues")
                except Exception as e:
                    logger.error(f"Error analyzing scope {scope.subroot}: {e}")
        finally:
            await cleanup_mcp_contexts(contexts)
        return all_issues

    def forward(self, scopes: Sequence[ScopeResult], repo_path: Path) -> list[Issue]:
        """Analyze scopes for bugs (sync wrapper).

        Args:
            scopes: The scopes containing changed files to analyze
            repo_path: Path to the cloned repository for code exploration

        Returns:
            List of bug issues found across all scopes
        """
        return asyncio.run(self.aforward(scopes, repo_path))