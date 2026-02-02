"""MCP server for tree-sitter AST parsing operations."""

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from codespy.tools.parsers.treesitter.parser import TreeSitterParser

mcp = FastMCP("treesitter-server")
_parser: TreeSitterParser | None = None


def _get_parser() -> TreeSitterParser:
    """Get the TreeSitterParser instance, raising if not initialized."""
    if _parser is None:
        raise RuntimeError("TreeSitterParser not initialized")
    return _parser


@mcp.tool()
def find_function_definitions(file_path: str, content: str | None = None) -> list[dict]:
    """Find all function/method definitions in a file.

    Args:
        file_path: Path to the file (relative to repo root)
        content: Optional file content (reads from file if not provided)

    Returns:
        List of function definitions with name, file, line_start, line_end,
        signature, parameters, return_type, docstring
    """
    parser = _get_parser()
    path = parser.repo_path / file_path
    functions = parser.find_function_definitions(path, content)
    return [f.model_dump() for f in functions]


@mcp.tool()
def find_function_calls(
    file_path: str,
    function_name: str,
    content: str | None = None,
) -> list[dict]:
    """Find all calls to a specific function in a file.

    Args:
        file_path: Path to the file (relative to repo root)
        function_name: Name of the function to find calls for
        content: Optional file content

    Returns:
        List of function calls with function_name, file, line_number,
        line_content, arguments_count, caller_function
    """
    parser = _get_parser()
    path = parser.repo_path / file_path
    calls = parser.find_function_calls(path, function_name, content)
    return [c.model_dump() for c in calls]


@mcp.tool()
def find_all_calls_in_file(file_path: str, content: str | None = None) -> list[dict]:
    """Find all function calls in a file.

    Args:
        file_path: Path to the file (relative to repo root)
        content: Optional file content

    Returns:
        List of all function calls with function_name, file, line_number,
        line_content, arguments_count, caller_function
    """
    parser = _get_parser()
    path = parser.repo_path / file_path
    calls = parser.find_all_calls_in_file(path, content)
    return [c.model_dump() for c in calls]


if __name__ == "__main__":
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    _parser = TreeSitterParser(Path(repo_path))
    mcp.run()