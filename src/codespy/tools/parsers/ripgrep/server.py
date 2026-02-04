"""MCP server for ripgrep code search operations."""

import logging
import os
import sys
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from codespy.tools.parsers.ripgrep.client import RipgrepSearch

logger = logging.getLogger(__name__)
_caller_module = os.environ.get("MCP_CALLER_MODULE", "unknown")

mcp = FastMCP("ripgrep")
_search: RipgrepSearch | None = None


def _get_search() -> RipgrepSearch:
    """Get the RipgrepSearch instance, raising if not initialized."""
    if _search is None:
        raise RuntimeError("RipgrepSearch not initialized")
    return _search


@mcp.tool()
def find_function_usages(
    function_name: str,
    file_patterns: list[str] | None = None,
    exclude_file: str | None = None,
) -> list[dict]:
    """Find all usages of a function in the codebase.

    Args:
        function_name: Name of the function to search for
        file_patterns: Optional glob patterns to limit search (e.g., ["*.go", "*.py"])
        exclude_file: File to exclude from results (typically the definition file)

    Returns:
        List of search results with file, line_number, line_content, match_text
    """
    logger.info(f"[RG] {_caller_module} -> find_function_usages: {function_name}")
    results = _get_search().find_function_usages(function_name, file_patterns, exclude_file)
    return [asdict(r) for r in results]


@mcp.tool()
def find_type_usages(
    type_name: str,
    file_patterns: list[str] | None = None,
) -> list[dict]:
    """Find all usages of a type in the codebase.

    Args:
        type_name: Name of the type to search for
        file_patterns: Optional glob patterns to limit search

    Returns:
        List of search results
    """
    logger.info(f"[RG] {_caller_module} -> find_type_usages: {type_name}")
    results = _get_search().find_type_usages(type_name, file_patterns)
    return [asdict(r) for r in results]


@mcp.tool()
def find_imports_of(
    module_or_package: str,
    file_patterns: list[str] | None = None,
) -> list[dict]:
    """Find all files that import a module or package.

    Args:
        module_or_package: Module or package name to search for
        file_patterns: Optional glob patterns

    Returns:
        List of search results showing import statements
    """
    logger.info(f"[RG] {_caller_module} -> find_imports_of: {module_or_package}")
    results = _get_search().find_imports_of(module_or_package, file_patterns)
    return [asdict(r) for r in results]


@mcp.tool()
def find_callers(
    function_name: str,
    defining_file: str,
    language: str = "auto",
) -> list[dict]:
    """Find all files that call a specific function.

    Args:
        function_name: Name of the function
        defining_file: File where function is defined (will be excluded)
        language: Programming language ("go", "python", "typescript", "auto")

    Returns:
        List of search results showing callers
    """
    logger.info(f"[RG] {_caller_module} -> find_callers: {function_name} (from {defining_file})")
    results = _get_search().find_callers(function_name, defining_file, language)
    return [asdict(r) for r in results]


@mcp.tool()
def search_literal(
    text: str,
    file_patterns: list[str] | None = None,
) -> list[dict]:
    """Search for literal text (not regex).

    Args:
        text: Exact text to search for
        file_patterns: Optional glob patterns

    Returns:
        List of search results
    """
    logger.info(f"[RG] {_caller_module} -> search_literal: {text[:50]}...")
    results = _get_search().search_literal(text, file_patterns)
    return [asdict(r) for r in results]


if __name__ == "__main__":
    # Suppress noisy MCP server "Processing request" logs
    logging.getLogger("mcp.server").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)
    
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    _search = RipgrepSearch(repo_path)
    mcp.run()
