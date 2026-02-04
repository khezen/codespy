"""MCP server for tree-sitter AST parsing operations."""

import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from codespy.tools.parsers.treesitter.parser import TreeSitterParser

logger = logging.getLogger(__name__)
_caller_module = os.environ.get("MCP_CALLER_MODULE", "unknown")

mcp = FastMCP("treesitter")
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
    logger.info(f"[TS] {_caller_module} -> find_function_definitions: {file_path}")
    parser = _get_parser()
    path = parser.repo_path / file_path
    functions = parser.find_function_definitions(path, content)
    return [asdict(f) for f in functions]


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
    logger.info(f"[TS] {_caller_module} -> find_function_calls: {function_name} in {file_path}")
    parser = _get_parser()
    path = parser.repo_path / file_path
    calls = parser.find_function_calls(path, function_name, content)
    return [asdict(c) for c in calls]


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
    logger.info(f"[TS] {_caller_module} -> find_all_calls_in_file: {file_path}")
    parser = _get_parser()
    path = parser.repo_path / file_path
    calls = parser.find_all_calls_in_file(path, content)
    return [asdict(c) for c in calls]


# =============================================================================
# Terraform/HCL-specific tools
# =============================================================================


@mcp.tool()
def parse_terraform_file(file_path: str, content: str | None = None) -> dict[str, Any] | None:
    """Parse a Terraform file and extract all blocks.

    Args:
        file_path: Path to the .tf or .tfvars file (relative to repo root)
        content: Optional file content (reads from file if not provided)

    Returns:
        Dictionary containing all Terraform blocks:
        - resources: List of resource blocks (type, name, provider, attributes)
        - data_sources: List of data source blocks
        - variables: List of variable declarations
        - outputs: List of output declarations
        - modules: List of module calls (source, version, inputs)
        - providers: List of provider configurations
        - locals: List of local value definitions
    """
    logger.info(f"[TS] {_caller_module} -> parse_terraform_file: {file_path}")
    parser = _get_parser()
    path = parser.repo_path / file_path
    result = parser.parse_terraform_file(path, content)
    if result is None:
        return None
    return asdict(result)


@mcp.tool()
def list_terraform_resources(file_path: str, content: str | None = None) -> list[dict]:
    """List all Terraform resources in a file.

    Args:
        file_path: Path to the .tf file (relative to repo root)
        content: Optional file content

    Returns:
        List of resources with resource_type, resource_name, provider,
        file, line_start, line_end, attributes, depends_on
    """
    logger.info(f"[TS] {_caller_module} -> list_terraform_resources: {file_path}")
    parser = _get_parser()
    path = parser.repo_path / file_path
    result = parser.parse_terraform_file(path, content)
    if result is None:
        return []
    return [asdict(r) for r in result.resources]


@mcp.tool()
def list_terraform_variables(file_path: str, content: str | None = None) -> list[dict]:
    """List all Terraform variable declarations in a file.

    Args:
        file_path: Path to the .tf file (relative to repo root)
        content: Optional file content

    Returns:
        List of variables with name, var_type, default, description,
        sensitive, file, line_start, line_end
    """
    logger.info(f"[TS] {_caller_module} -> list_terraform_variables: {file_path}")
    parser = _get_parser()
    path = parser.repo_path / file_path
    result = parser.parse_terraform_file(path, content)
    if result is None:
        return []
    return [asdict(v) for v in result.variables]


@mcp.tool()
def list_terraform_outputs(file_path: str, content: str | None = None) -> list[dict]:
    """List all Terraform output declarations in a file.

    Args:
        file_path: Path to the .tf file (relative to repo root)
        content: Optional file content

    Returns:
        List of outputs with name, value_expression, description,
        sensitive, file, line_start, line_end
    """
    logger.info(f"[TS] {_caller_module} -> list_terraform_outputs: {file_path}")
    parser = _get_parser()
    path = parser.repo_path / file_path
    result = parser.parse_terraform_file(path, content)
    if result is None:
        return []
    return [asdict(o) for o in result.outputs]


@mcp.tool()
def list_terraform_modules(file_path: str, content: str | None = None) -> list[dict]:
    """List all Terraform module calls in a file.

    Args:
        file_path: Path to the .tf file (relative to repo root)
        content: Optional file content

    Returns:
        List of module calls with name, source, version, inputs,
        file, line_start, line_end
    """
    logger.info(f"[TS] {_caller_module} -> list_terraform_modules: {file_path}")
    parser = _get_parser()
    path = parser.repo_path / file_path
    result = parser.parse_terraform_file(path, content)
    if result is None:
        return []
    return [asdict(m) for m in result.modules]


@mcp.tool()
def list_terraform_data_sources(file_path: str, content: str | None = None) -> list[dict]:
    """List all Terraform data sources in a file.

    Args:
        file_path: Path to the .tf file (relative to repo root)
        content: Optional file content

    Returns:
        List of data sources with data_type, data_name, provider,
        file, line_start, line_end, attributes
    """
    logger.info(f"[TS] {_caller_module} -> list_terraform_data_sources: {file_path}")
    parser = _get_parser()
    path = parser.repo_path / file_path
    result = parser.parse_terraform_file(path, content)
    if result is None:
        return []
    return [asdict(d) for d in result.data_sources]


@mcp.tool()
def list_terraform_providers(file_path: str, content: str | None = None) -> list[dict]:
    """List all Terraform provider configurations in a file.

    Args:
        file_path: Path to the .tf file (relative to repo root)
        content: Optional file content

    Returns:
        List of providers with name, alias, attributes,
        file, line_start, line_end
    """
    logger.info(f"[TS] {_caller_module} -> list_terraform_providers: {file_path}")
    parser = _get_parser()
    path = parser.repo_path / file_path
    result = parser.parse_terraform_file(path, content)
    if result is None:
        return []
    return [asdict(p) for p in result.providers]


@mcp.tool()
def get_terraform_summary(file_path: str, content: str | None = None) -> dict[str, Any]:
    """Get a summary of Terraform resources in a file.

    Args:
        file_path: Path to the .tf file (relative to repo root)
        content: Optional file content

    Returns:
        Summary dictionary with counts and lists of names for each block type:
        - resource_count, resource_types
        - variable_count, variable_names
        - output_count, output_names
        - module_count, module_names
        - data_source_count, data_source_types
        - provider_count, provider_names
        - local_count, local_names
    """
    logger.info(f"[TS] {_caller_module} -> get_terraform_summary: {file_path}")
    parser = _get_parser()
    path = parser.repo_path / file_path
    result = parser.parse_terraform_file(path, content)

    if result is None:
        return {
            "error": "Failed to parse Terraform file",
            "file": file_path,
        }

    return {
        "file": file_path,
        "resource_count": len(result.resources),
        "resource_types": list({r.resource_type for r in result.resources}),
        "resources": [f"{r.resource_type}.{r.resource_name}" for r in result.resources],
        "variable_count": len(result.variables),
        "variable_names": [v.name for v in result.variables],
        "output_count": len(result.outputs),
        "output_names": [o.name for o in result.outputs],
        "module_count": len(result.modules),
        "module_names": [m.name for m in result.modules],
        "module_sources": [m.source for m in result.modules],
        "data_source_count": len(result.data_sources),
        "data_source_types": list({d.data_type for d in result.data_sources}),
        "provider_count": len(result.providers),
        "provider_names": list({p.name for p in result.providers}),
        "local_count": len(result.locals),
        "local_names": [loc.name for loc in result.locals],
    }


if __name__ == "__main__":
    # Suppress noisy MCP server "Processing request" logs
    logging.getLogger("mcp.server").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)
    
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    _parser = TreeSitterParser(Path(repo_path))
    mcp.run()
