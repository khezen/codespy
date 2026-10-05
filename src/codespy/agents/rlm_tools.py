"""RLM tool bridge and agent factory for async tool support.

This module bridges async MCP tools to sync RLM agents running off the event loop,
preventing 'This event loop is already running' errors when tools are invoked.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

import dspy  # type: ignore[import-untyped]

from codespy.agents.context_safe import CodespyRLM, ContextSafe

logger = logging.getLogger(__name__)

# Thread-local storage for the event loop at bridge time
_bridge_thread_id: int | None = None


def _is_coroutine_function(obj: Any) -> bool:
    """Check if an object is a coroutine function or callable that returns a coroutine."""
    if inspect.iscoroutinefunction(obj):
        return True
    # Handle dspy.Tool with async func
    if hasattr(obj, "func"):
        return inspect.iscoroutinefunction(obj.func)
    return False


def _get_tool_name(tool: Any) -> str:
    """Extract name from a tool object."""
    if hasattr(tool, "name"):
        return str(tool.name)
    if hasattr(tool, "__name__"):
        return str(tool.__name__)
    return str(type(tool).__name__)


def _get_tool_desc(tool: Any) -> str | None:
    """Extract description from a tool object."""
    if hasattr(tool, "desc"):
        return str(tool.desc) if tool.desc is not None else None
    if hasattr(tool, "__doc__"):
        return str(tool.__doc__) if tool.__doc__ is not None else None
    return None


def _get_tool_args(tool: Any) -> dict[str, Any] | None:
    """Extract args from a tool object."""
    if hasattr(tool, "args"):
        return tool.args  # type: ignore[return-value]
    return None


def _get_tool_arg_types(tool: Any) -> dict[str, type] | None:
    """Extract arg_types from a tool object."""
    if hasattr(tool, "arg_types"):
        return tool.arg_types  # type: ignore[return-value]
    return None


def _get_tool_arg_desc(tool: Any) -> dict[str, str] | None:
    """Extract arg_desc from a tool object."""
    if hasattr(tool, "arg_desc"):
        return tool.arg_desc  # type: ignore[return-value]
    return None


def _signature_from_args(
    args: dict[str, Any] | None, arg_types: dict[str, Any] | None
) -> inspect.Signature:
    """Build a real signature from dspy.Tool args (JSON schema per arg).

    dspy.RLM copies `inspect.signature(tool.func)` into the sandbox wrapper. A bare
    `sync_wrapper(**kwargs)` becomes `def tool(kwargs):` there, so keyword calls
    fail ("unexpected keyword argument") and positional ones arrive as `kwargs=`.
    """
    required: list[inspect.Parameter] = []
    optional: list[inspect.Parameter] = []
    for name, schema in (args or {}).items():
        annotation = (arg_types or {}).get(name, inspect.Parameter.empty)
        if isinstance(schema, dict) and "default" in schema:
            optional.append(
                inspect.Parameter(
                    name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=schema["default"], annotation=annotation,
                )
            )
        else:
            required.append(
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation)
            )
    return inspect.Signature(required + optional)


def bridge_tools_to_loop(
    tools: list[Any],
    loop: asyncio.AbstractEventLoop,
    timeout: float,
) -> list[dspy.Tool]:
    """Bridge async tools to run on the provided event loop from sync contexts.

    For each tool:
    - If already a dspy.Tool with sync func: pass through unchanged
    - If a dspy.Tool with async func: wrap with sync bridge
    - If a plain callable: wrap in dspy.Tool, then apply above

    Args:
        tools: List of tools (dspy.Tool or plain callables)
        loop: The event loop to schedule async tool calls on
        timeout: Timeout in seconds for tool calls

    Returns:
        List of bridged dspy.Tool instances with sync wrappers
    """
    global _bridge_thread_id
    _bridge_thread_id = threading.get_ident()

    bridged: list[dspy.Tool] = []

    for tool in tools:
        # Normalize to dspy.Tool
        if isinstance(tool, dspy.Tool):
            dspy_tool = tool
        else:
            # Wrap plain callable in dspy.Tool
            dspy_tool = dspy.Tool(tool)

        # Get the underlying function
        func = dspy_tool.func

        # Check if async
        if not _is_coroutine_function(func):
            # Sync tool - pass through unchanged
            bridged.append(dspy_tool)
            continue

        # Async tool - create sync wrapper that schedules on the loop
        tool_name = _get_tool_name(dspy_tool)
        tool_desc = _get_tool_desc(dspy_tool)
        tool_args = _get_tool_args(dspy_tool)
        tool_arg_types = _get_tool_arg_types(dspy_tool)
        tool_arg_desc = _get_tool_arg_desc(dspy_tool)

        def make_sync_wrapper(
            async_func: Any,
            tool_loop: asyncio.AbstractEventLoop,
            tool_timeout: float,
            name: str,
        ) -> Any:
            """Create a sync wrapper for an async function."""

            def sync_wrapper(**kwargs: Any) -> Any:
                # Deadlock guard: if we're on the bridge thread, raise
                if threading.get_ident() == _bridge_thread_id:
                    raise RuntimeError(
                        "RLM tools must run off the event loop; use run_rlm()"
                    )

                # Schedule the coroutine on the loop and wait for result
                future = asyncio.run_coroutine_threadsafe(
                    async_func(**kwargs), tool_loop
                )
                try:
                    return future.result(timeout=tool_timeout)
                except FutureTimeoutError as e:
                    future.cancel()
                    raise TimeoutError(
                        f"Tool '{name}' timed out after {tool_timeout}s"
                    ) from e
                except Exception as e:
                    # Re-raise with tool name context
                    raise type(e)(f"Tool '{name}' failed: {e}") from e

            return sync_wrapper

        sync_func = make_sync_wrapper(func, loop, timeout, tool_name)

        # Preserve function metadata for signature inspection
        if hasattr(func, "__name__"):
            sync_func.__name__ = func.__name__
        if hasattr(func, "__doc__"):
            sync_func.__doc__ = func.__doc__
        sync_func.__signature__ = _signature_from_args(tool_args, tool_arg_types)  # type: ignore[attr-defined]

        # Create new dspy.Tool with sync wrapper
        bridged_tool = dspy.Tool(
            sync_func,
            name=tool_name,
            desc=tool_desc,
            args=tool_args,
            arg_types=tool_arg_types,
            arg_desc=tool_arg_desc,
        )
        bridged.append(bridged_tool)

    return bridged


def build_rlm_agent(
    signature: Any,
    tools: list[Any],
    *,
    name: str,
    max_iters: int,
    max_llm_calls: int,
    rlm_threshold: float,
) -> ContextSafe:
    """Build an RLM agent with bridged tools.

    Must be called inside the running event loop.

    Args:
        signature: DSPy signature for the agent
        tools: List of tools (will be bridged to the current loop)
        name: Agent name for logging
        max_iters: Maximum RLM iterations
        max_llm_calls: Maximum LLM calls for RLM fallback
        rlm_threshold: ContextSafe RLM fallback threshold

    Returns:
        ContextSafe wrapped RLM agent with bridged tools
    """
    from codespy.config import get_settings

    settings = get_settings()
    loop = asyncio.get_running_loop()
    timeout = settings.llm.timeout

    # Bridge tools to the current loop
    bridged_tools = bridge_tools_to_loop(tools, loop, timeout)

    # Build RLM with bridged tools
    rlm = CodespyRLM(
        signature,
        tools=bridged_tools,
        max_iters=max_iters,
        max_llm_calls=max_llm_calls,
        verbose=logger.isEnabledFor(logging.DEBUG),
    )

    # Wrap in ContextSafe with same bridged tools
    return ContextSafe(
        rlm,
        signature,
        tools=bridged_tools,
        name=name,
        max_iters=max_iters,
        max_llm_calls=max_llm_calls,
        rlm_threshold=rlm_threshold,
    )


async def run_rlm(agent: ContextSafe, **kwargs: Any) -> Any:
    """Run an RLM agent off the event loop using asyncio.to_thread.

    This allows the RLM agent to run synchronously (including its sync tool
    wrappers) while the event loop remains free to service bridged tool calls.

    Args:
        agent: The ContextSafe-wrapped RLM agent
        **kwargs: Arguments to pass to the agent

    Returns:
        The agent result
    """
    return await asyncio.to_thread(agent, **kwargs)


def log_rlm_outcome(
    result: Any,
    module: str,
    scope: str,
) -> None:
    """Log diagnostics for RLM outcomes, especially extract fallback cases.

    Args:
        result: The prediction result from the RLM agent
        module: Module name (e.g., 'code_review', 'supply_chain')
        scope: Scope identifier for context
    """
    # Check if this was an extract fallback
    final_reasoning = getattr(result, "final_reasoning", None)
    if final_reasoning != "Extract forced final output":
        return

    # Get trajectory if available
    trajectory = getattr(result, "trajectory", [])
    if not trajectory:
        return

    # Count error entries
    error_count = 0
    first_error: str | None = None

    for entry in trajectory:
        # Entry is a REPLEntry-like object or dict
        if isinstance(entry, dict):
            output = entry.get("output", "")
        else:
            output = getattr(entry, "output", "")

        if output and "Error" in str(output):
            error_count += 1
            if first_error is None:
                first_error = str(output)

    # Log warning with diagnostic info
    if error_count > 0:
        msg = (
            f"RLM[{module}] scope {scope}: extract fallback after {len(trajectory)} "
            f"trajectory entries, {error_count} tool errors. "
        )
        if first_error:
            truncated = first_error[:300]
            if len(first_error) > 300:
                truncated += "..."
            msg += f"First error: {truncated}"
        logger.warning(msg)
    else:
        logger.debug(
            f"RLM[{module}] scope {scope}: extract fallback after {len(trajectory)} "
            "trajectory entries, no tool errors"
        )
