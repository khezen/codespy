"""Bug detection module with code exploration tools."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import ModuleContext, get_cost_tracker
from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.agents.reviewer.modules.helpers import get_language, is_speculative, MIN_CONFIDENCE
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class BugDetectionSignature(dspy.Signature):
    """Detect VERIFIED bugs and logic errors in code changes.

    You are an expert software engineer reviewing code for bugs.
    You have access to tools that let you explore the codebase to VERIFY your findings.

    CRITICAL RULES:
    - ONLY report bugs you can VERIFY using the available tools
    - Before reporting any bug, USE the tools to verify your assumptions
    - DO NOT speculate about potential issues you cannot verify
    - DO NOT report "might be", "could be", "possibly", "may cause" issues
    - If you cannot point to the EXACT buggy code with evidence, do NOT report it
    - Quality over quantity: prefer 0 reports over 1 speculative report

    VERIFICATION WORKFLOW:
    1. Analyze the diff and full content for potential issues
    2. For each suspected issue, VERIFY using tools:
       - Use read_file to examine related files (imports, dependencies, base classes)
       - Use find_function_definitions to check function signatures and implementations
       - Use find_function_calls to understand how functions are called
       - Use find_function_usages/find_callers to trace usage patterns
       - Use search_literal to find related code patterns
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

    diff: str = dspy.InputField(
        desc="The code diff showing changes"
    )
    full_content: str = dspy.InputField(
        desc="The full file content after changes"
    )
    filename: str = dspy.InputField(
        desc="Path to the file being analyzed"
    )
    language: str = dspy.InputField(
        desc="Programming language of the file"
    )
    category: IssueCategory = dspy.InputField(
        desc="Category for all issues (use this value for the 'category' field)"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="VERIFIED bugs found. Empty list if none confirmed."
    )


class BugDetector(dspy.Module):
    """Detects bugs and logic errors using DSPy with code exploration tools."""

    category = IssueCategory.BUG
    MODULE_NAME = "bug_detector"

    def __init__(self) -> None:
        """Initialize the bug detector."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()

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
        all_issues: list[Issue] = []
        tools, contexts = await self._create_mcp_tools(repo_path)
        # Create ReAct agent with code exploration tools
        bug_detection_agent = dspy.ReAct(
            signature=BugDetectionSignature,
            tools=tools,
            max_iters=10,
        )
        try:
            # Use ModuleContext to track costs and timing for this module
            async with ModuleContext(self.MODULE_NAME, self._cost_tracker):
                for scope in scopes:
                    for file in scope.changed_files:
                        if not file.patch:
                            logger.debug(f"Skipping {file.filename}: no patch available")
                            continue
                        try:
                            result = await bug_detection_agent.acall(
                                diff=file.patch or "",
                                full_content=file.content or "",
                                filename=file.filename,
                                language=get_language(file),
                                category=self.category,
                            )
                            issues = [
                                issue for issue in result.issues
                                if issue.confidence >= MIN_CONFIDENCE and not is_speculative(issue)
                            ]
                            all_issues.extend(issues)
                            logger.debug(f"  Bugs in {file.filename}: {len(issues)} issues")
                        except Exception as e:
                            logger.error(f"Error analyzing {file.filename}: {e}")
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