"""Utilities for MCP server connections."""

import logging
import os
import sys
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]
from mcp import Client, StdioServerParameters

logger = logging.getLogger(__name__)


async def connect_mcp_server(
    mcp_path: Path,
    args: list[str] | None = None,
    contexts: list[Any] | None = None,
    caller_module: str = "unknown",
) -> list[Any]:
    """Connect to an MCP server and return DSPy tools.

    Args:
        mcp_path: Path to the MCP server Python module
        args: Additional command-line arguments for the server
        contexts: List to append context managers for cleanup (client)
        caller_module: Name of the calling module for logging (e.g., 'supply_chain_auditor')

    Returns:
        List of DSPy Tool objects from the MCP server
    """
    if args is None:
        args = []
    if contexts is None:
        contexts = []

    # Pass caller_module to subprocess via environment variable
    env = os.environ.copy()
    env["MCP_CALLER_MODULE"] = caller_module

    client = Client(
        StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_path)] + args,
            env=env,
        )
    )
    await client.__aenter__()
    contexts.append(client)

    tools_response = await client.list_tools()
    return [dspy.Tool.from_mcp_tool(client, tool) for tool in tools_response.tools]


async def cleanup_mcp_contexts(contexts: list[Any]) -> None:
    """Clean up MCP context managers.

    Args:
        contexts: List of context managers to close (in reverse order)
    """
    import logging

    logger = logging.getLogger(__name__)

    for ctx in reversed(contexts):
        try:
            await ctx.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Error cleaning up MCP context: {e}")
