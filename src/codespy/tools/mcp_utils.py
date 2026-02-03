"""Utilities for MCP server connections."""

import sys
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]
from mcp import ClientSession, StdioServerParameters  # type: ignore[import-not-found]
from mcp.client.stdio import stdio_client  # type: ignore[import-not-found]


async def connect_mcp_server(
    mcp_path: Path,
    args: list[str] | None = None,
    contexts: list[Any] | None = None,
) -> list[Any]:
    """Connect to an MCP server and return DSPy tools.

    Args:
        mcp_path: Path to the MCP server Python module
        args: Additional command-line arguments for the server
        contexts: List to append context managers for cleanup (transport, session)

    Returns:
        List of DSPy Tool objects from the MCP server
    """
    if args is None:
        args = []
    if contexts is None:
        contexts = []

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(mcp_path)] + args,
    )

    transport = stdio_client(params)
    streams = await transport.__aenter__()
    contexts.append(transport)

    session = ClientSession(*streams)
    await session.__aenter__()
    contexts.append(session)
    await session.initialize()

    tools_response = await session.list_tools()
    return [dspy.Tool.from_mcp_tool(session, tool) for tool in tools_response.tools]


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