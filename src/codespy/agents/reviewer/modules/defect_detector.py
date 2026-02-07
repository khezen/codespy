"""Code defect detection module — bugs, logic errors, and security vulnerabilities."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.reviewer.models import Issue, IssueCategory, ScopeResult
from codespy.agents.reviewer.modules.helpers import MIN_CONFIDENCE
from codespy.config import get_settings
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class CodeDefectSignature(dspy.Signature):
    """Detect VERIFIED defects in code changes: bugs, logic errors, and security vulnerabilities.

    You are a busy Principal Engineer with very little time. Review code for critical, verified defects only.
    Be extremely terse. Use imperative mood ("Fix X", not "You should fix X").
    You have access to tools that let you explore the codebase to VERIFY your findings.

    CRITICAL RULES:
    - Analyze ALL changed files in the scope
    - ONLY report defects you can VERIFY using the available tools
    - Before reporting any defect, USE the tools to verify your assumptions
    - DO NOT speculate about potential issues you cannot verify
    - DO NOT report "might be", "could be", "possibly", "may cause" issues
    - If you cannot point to the EXACT defective code with evidence, do NOT report it
    - Quality over quantity: prefer 0 reports over 1 speculative report

    VERIFICATION WORKFLOW:
    1. Review each changed file's patch in scope.changed_files — the diff shows what changed
    2. For suspected defects that need verification beyond the patch:
       - Use find_function_definitions to check function signatures and implementations
       - Use find_function_calls to understand how functions are called and trace data flow
       - Use find_function_usages/find_callers to trace usage patterns
       - Use search_literal to find related patterns (sanitization, encoding, validation)
       - Use read_file ONLY if you need broader context not visible in the patch
    3. Only report issues that are CONFIRMED by your verification

    BUGS & LOGIC ERRORS to look for (with verification):
    - Logic errors: verify the condition is actually incorrect by checking related code
    - Null/undefined references: verify the check is actually missing by reading the code
    - Resource leaks: verify there's no cleanup in finally/defer/close methods
    - Error handling: verify errors aren't handled elsewhere in the call chain
    - Type mismatches: verify types by checking definitions
    - Off-by-one errors: verify by understanding the data structure bounds

    SECURITY VULNERABILITIES to look for (with verification):
    - Injection attacks (SQL, command, XSS, etc.): verify input reaches dangerous sink without sanitization
    - Authentication and authorization issues: verify auth check is actually missing by reading the code
    - Sensitive data exposure: verify sensitive data is actually exposed, not just accessed
    - Insecure cryptographic practices: verify weak crypto by checking the actual algorithm used
    - Security misconfigurations: verify misconfiguration by checking actual config values
    - Input validation issues: verify input is not validated by checking validation code
    - Path traversal vulnerabilities: verify path input is not validated/sanitized
    - Race conditions: verify shared state access without synchronization
    - Memory safety issues: verify unsafe memory operations (buffer overflows, use-after-free, etc.)

    CATEGORY ASSIGNMENT:
    - Use "bug" for logic errors, null refs, resource leaks, error handling, type mismatches, off-by-one
    - Use "security" for injection, auth, crypto, data exposure, misconfig, path traversal, memory safety

    DO NOT report:
    - Style issues or minor improvements
    - Hypothetical vulnerabilities or edge cases without evidence
    - Issues that might exist in code you haven't verified
    - "Could be vulnerable if..." scenarios

    OUTPUT RULES:
    - Reference files by name and line number only — never copy source code into issues.
    - Do not repeat patch content in reasoning steps. Keep each reasoning step to 1-2 sentences.
    - Empty list if no verified defects. No approval text ("LGTM", "looks good").
    - description: ≤25 words, imperative tone, no filler ("Fix X", "Sanitize Y", "Handle Z").
    - No polite or conversational language ("I suggest", "Please consider", "Great").
    - Do not populate code_snippet — use line numbers instead.
    - For security issues, include cwe_id where applicable.
    """

    scope: ScopeResult = dspy.InputField(
        desc="Full scope context including all changed files, scope type, subroot, and language"
    )

    issues: list[Issue] = dspy.OutputField(
        desc="Verified defects only. Set category to 'bug' or 'security' per issue. Titles <10 words. Descriptions ≤25 words, imperative. Empty list if none."
    )


class DefectDetector(dspy.Module):
    """Detects bugs, logic errors, and security vulnerabilities using DSPy with code exploration tools."""

    def __init__(self) -> None:
        """Initialize the defect detector."""
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
        caller = "defect_detector"
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
        """Analyze scopes for defects and return issues.

        Args:
            scopes: The scopes containing changed files to analyze
            repo_path: Path to the cloned repository for code exploration

        Returns:
            List of defect issues (bugs + security) found across all scopes
        """
        # Check if signature is enabled
        if not self._settings.is_signature_enabled("defect_detection"):
            logger.debug("Skipping defect_detection: disabled")
            return []

        all_issues: list[Issue] = []
        tools, contexts = await self._create_mcp_tools(repo_path)

        # Get max_iters from signature config
        max_iters = self._settings.get_max_iters("defect_detection")

        # Create ReAct agent with code exploration tools
        defect_agent = dspy.ReAct(
            signature=CodeDefectSignature,
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
                    # Track defect_detection signature costs
                    async with SignatureContext("defect_detection", self._cost_tracker):
                        result = await defect_agent.acall(
                            scope=scope,
                        )
                    issues = [
                        issue for issue in result.issues
                        if issue.confidence >= MIN_CONFIDENCE
                    ]
                    all_issues.extend(issues)
                    logger.debug(f"  Defects in scope {scope.subroot}: {len(issues)} issues")
                except Exception as e:
                    logger.error(f"Error analyzing scope {scope.subroot}: {e}")
        finally:
            await cleanup_mcp_contexts(contexts)
        logger.info(f"Defect detection found {len(all_issues)} issues")
        return all_issues

    def forward(self, scopes: Sequence[ScopeResult], repo_path: Path) -> list[Issue]:
        """Analyze scopes for defects (sync wrapper).

        Args:
            scopes: The scopes containing changed files to analyze
            repo_path: Path to the cloned repository for code exploration

        Returns:
            List of defect issues found across all scopes
        """
        return asyncio.run(self.aforward(scopes, repo_path))