"""RLM tool bridge and agent factory for async tool support.

This module bridges async MCP tools to sync RLM agents running off the event loop,
preventing 'This event loop is already running' errors when tools are invoked.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import threading
from collections.abc import Iterable
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

import dspy  # type: ignore[import-untyped]
from dspy.primitives.repl_types import REPLEntry  # type: ignore[import-untyped]

from codespy.agents.context_safe import (
    COVERAGE_REJECTION_PREFIX,
    DSPY_DEFAULT_MAX_OUTPUT_CHARS,
    REFUSAL_EXTRACT_REASONING,
    CodespyRLM,
    ContextSafe,
    SubmitCheck,
)

logger = logging.getLogger(__name__)

# A changed file counts as seen when at least this share of its significant
# patch lines appear in what the root LM saw. Below 100% to tolerate escaping
# in reprs (e.g. `read_file` dict output).
COVERAGE_THRESHOLD = 0.9
_MIN_SIGNIFICANT_LINE_CHARS = 3
_EXTRACT_REASONING = "Extract forced final output"

# Thread id of the event loop at bridge time
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
    max_output_chars: int | None = None,
    submit_check: SubmitCheck | None = None,
) -> ContextSafe:
    """Build an RLM agent with bridged tools.

    Must be called inside the running event loop.

    Args:
        signature: DSPy signature for the agent
        tools: List of tools (will be bridged to the current loop)
        name: Agent name for logging; also the signature name used to resolve
            ``max_output_chars`` when not given
        max_iters: Maximum RLM iterations
        max_llm_calls: Maximum LLM calls for RLM fallback
        rlm_threshold: ContextSafe RLM fallback threshold
        max_output_chars: Max chars of one REPL step output shown to the model
            (``None``: ``settings.get_max_output_chars(name)``)
        submit_check: Optional SUBMIT gate (see ``CodespyRLM``)

    Returns:
        ContextSafe wrapped RLM agent with bridged tools
    """
    from codespy.config import get_settings

    settings = get_settings()
    loop = asyncio.get_running_loop()
    timeout = settings.llm.timeout
    if max_output_chars is None:
        max_output_chars = settings.get_max_output_chars(name)

    # Bridge tools to the current loop
    bridged_tools = bridge_tools_to_loop(tools, loop, timeout)

    # Build RLM with bridged tools
    rlm = CodespyRLM(
        signature,
        tools=bridged_tools,
        max_iters=max_iters,
        max_llm_calls=max_llm_calls,
        max_output_chars=max_output_chars,
        submit_check=submit_check,
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
        max_output_chars=max_output_chars,
        submit_check=submit_check,
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


def _entry_field(entry: Any, name: str) -> str:
    """Read a field from a REPLEntry, a trajectory dict, or a REPLEntry-like object."""
    value = entry.get(name, "") if isinstance(entry, dict) else getattr(entry, name, "")
    return value if isinstance(value, str) else str(value or "")


def _file_field(changed_file: Any, name: str) -> Any:
    if isinstance(changed_file, dict):
        return changed_file.get(name)
    return getattr(changed_file, name, None)


def _significant_lines(patch: str) -> set[str]:
    """Patch lines that identify a file's content (diff markers and blanks dropped)."""
    lines: set[str] = set()
    for raw in patch.splitlines():
        if raw.startswith("@@"):
            continue
        line = raw[1:] if raw[:1] in ("+", "-", " ") else raw
        line = line.strip()
        if len(line) >= _MIN_SIGNIFICANT_LINE_CHARS:
            lines.add(line)
    return lines


def unseen_files(
    changed_files: Iterable[Any],
    history: Iterable[Any],
    max_output_chars: int,
) -> list[str]:
    """Changed files whose patch the root LM has not actually seen.

    The corpus is what the model saw: each history entry's output after the
    ``max_output_chars`` cap (``REPLEntry.format_output``, the same rendering
    the action prompt uses), plus the entry's code (pasted lines were seen).
    A file is seen when at least ``COVERAGE_THRESHOLD`` of its significant patch
    lines are substrings of the corpus. A file with no patch counts as seen.

    Args:
        changed_files: Objects or dicts with ``filename`` and ``patch``
        history: ``REPLHistory``, or REPLEntry objects / trajectory dicts
        max_output_chars: Output cap the model saw outputs under

    Returns:
        Filenames not seen yet, in input order.
    """
    chunks: list[str] = []
    for entry in history:
        chunks.append(REPLEntry.format_output(_entry_field(entry, "output"), max_output_chars))
        chunks.append(_entry_field(entry, "code"))
    corpus = "\n".join(chunks)
    corpus_lines = {line.strip() for line in corpus.splitlines()}

    missing: list[str] = []
    for changed_file in changed_files:
        patch = _file_field(changed_file, "patch")
        if not patch:
            continue
        lines = _significant_lines(patch)
        if not lines:
            continue
        allowed_misses = len(lines) - math.ceil(round(COVERAGE_THRESHOLD * len(lines), 9))
        misses = 0
        for line in lines:
            if line in corpus_lines or line in corpus:
                continue
            misses += 1
            if misses > allowed_misses:
                missing.append(str(_file_field(changed_file, "filename")))
                break
    return missing


def _end_reason(final_reasoning: Any) -> str:
    if final_reasoning == REFUSAL_EXTRACT_REASONING:
        return "refusal"
    if final_reasoning == _EXTRACT_REASONING:
        return "extract"
    return "SUBMIT"


def log_rlm_outcome(
    result: Any,
    module: str,
    scope: str,
    *,
    changed_files: Iterable[Any] | None = None,
    max_output_chars: int | None = None,
    max_iters: int | None = None,
) -> list[str]:
    """Log diagnostics for RLM outcomes, especially extract fallback cases.

    With ``changed_files`` or ``max_output_chars``, also logs coverage
    telemetry at INFO: iterations used, end reason, SUBMIT gate rejections,
    truncated outputs and unseen changed files.

    Args:
        result: The prediction result from the RLM agent
        module: Module name (e.g., 'code_review', 'supply_chain')
        scope: Scope identifier for context
        changed_files: Changed files the run had to cover (optional)
        max_output_chars: Output cap the run used (optional)
        max_iters: Iteration budget the run used (optional, for the log)

    Returns:
        Changed files not seen by the model (empty without ``changed_files``).
    """
    final_reasoning = getattr(result, "final_reasoning", None)
    unseen: list[str] = []
    if changed_files is not None or max_output_chars is not None:
        trajectory = list(getattr(result, "trajectory", None) or [])
        cap = max_output_chars or DSPY_DEFAULT_MAX_OUTPUT_CHARS
        if changed_files is not None:
            unseen = unseen_files(changed_files, trajectory, cap)
        rejections = sum(
            1 for e in trajectory if _entry_field(e, "output").startswith(COVERAGE_REJECTION_PREFIX)
        )
        truncated = sum(1 for e in trajectory if len(_entry_field(e, "output")) > cap)
        logger.info(
            f"RLM[{module}] scope {scope}: iterations={len(trajectory)}/{max_iters or '?'}, "
            f"end={_end_reason(final_reasoning)}, coverage_rejections={rejections}, "
            f"truncated_outputs={truncated} (cap {cap:,}), "
            f"unseen_files={len(unseen)}{': ' + ', '.join(unseen) if unseen else ''}"
        )

    _log_fallback(result, module, scope, final_reasoning)
    return unseen


def _log_fallback(result: Any, module: str, scope: str, final_reasoning: Any) -> None:
    """Warn on refusal/extract endings (tool errors in the trajectory)."""
    if final_reasoning == REFUSAL_EXTRACT_REASONING:
        logger.warning(
            f"RLM[{module}] scope {scope}: action step refused (content_filter) after "
            f"{len(getattr(result, 'trajectory', []) or [])} steps; outputs extracted from history"
        )
        return
    if final_reasoning != _EXTRACT_REASONING:
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
