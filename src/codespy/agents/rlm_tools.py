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


def _seen_corpus(history: Iterable[Any], max_output_chars: int) -> tuple[str, set[str]]:
    """What the root LM saw: each step's capped output plus its code.

    Outputs are rendered with ``REPLEntry.format_output`` under
    ``max_output_chars`` (the same rendering the action prompt uses); code is
    included because pasted lines were seen.

    Returns:
        ``(corpus, corpus_lines)``: the joined text and its stripped lines.
    """
    chunks: list[str] = []
    for entry in history:
        chunks.append(REPLEntry.format_output(_entry_field(entry, "output"), max_output_chars))
        chunks.append(_entry_field(entry, "code"))
    corpus = "\n".join(chunks)
    return corpus, {line.strip() for line in corpus.splitlines()}


def _lines_seen(lines: set[str], corpus: str, corpus_lines: set[str]) -> bool:
    """At least ``COVERAGE_THRESHOLD`` of ``lines`` appear in the corpus."""
    if not lines:
        return True
    allowed_misses = len(lines) - math.ceil(round(COVERAGE_THRESHOLD * len(lines), 9))
    misses = 0
    for line in lines:
        if line in corpus_lines or line in corpus:
            continue
        misses += 1
        if misses > allowed_misses:
            return False
    return True


def unseen_files(
    changed_files: Iterable[Any],
    history: Iterable[Any],
    max_output_chars: int,
) -> list[str]:
    """Changed files whose patch the root LM has not actually seen.

    The corpus is what the model saw (see ``_seen_corpus``). A file is seen when
    at least ``COVERAGE_THRESHOLD`` of its significant patch lines are
    substrings of the corpus. A file with no patch counts as seen.

    Args:
        changed_files: Objects or dicts with ``filename`` and ``patch``
        history: ``REPLHistory``, or REPLEntry objects / trajectory dicts
        max_output_chars: Output cap the model saw outputs under

    Returns:
        Filenames not seen yet, in input order.
    """
    corpus, corpus_lines = _seen_corpus(history, max_output_chars)
    missing: list[str] = []
    for changed_file in changed_files:
        patch = _file_field(changed_file, "patch")
        if not patch:
            continue
        if not _lines_seen(_significant_lines(patch), corpus, corpus_lines):
            missing.append(str(_file_field(changed_file, "filename")))
    return missing


# --- Prefrontal memory coverage ---------------------------------------------

_UNTITLED_MEMORY_SECTION = "memory"


def memory_sections(text: str | None) -> list[tuple[str, int, int, str]]:
    """Split rendered Prefrontal memory into its sections.

    Splits only on known section headers (``## <title>`` lines where the title
    is one ``format_context_sections`` renders), so markdown headings inside a
    briefing or a reflect answer stay in their section. Text before the first
    header (or the whole text, without any header) is a section titled
    ``memory``.

    Returns:
        ``(title, start, end, body)`` per section, in order. ``text[start:end]``
        is the header line plus the body (trailing blank lines excluded).
    """
    from codespy.agents.memory.prefrontal.format import (
        SECTION_HEADER_PREFIX,
        is_section_title,
    )

    if not text or not text.strip():
        return []
    # (title, header_start, body_start)
    headers: list[tuple[str, int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if bare.startswith(SECTION_HEADER_PREFIX):
            title = bare[len(SECTION_HEADER_PREFIX):].strip()
            if title and is_section_title(title):
                headers.append((title, offset, offset + len(line)))
        offset += len(line)

    sections: list[tuple[str, int, int, str]] = []

    def add(title: str, start: int, body_start: int, stop: int) -> None:
        end = start + len(text[start:stop].rstrip())
        body = text[body_start:max(body_start, end)]
        sections.append((title, start, max(end, start), body))

    first = headers[0][1] if headers else len(text)
    if text[:first].strip():
        add(_UNTITLED_MEMORY_SECTION, 0, 0, first)
    for i, (title, start, body_start) in enumerate(headers):
        stop = headers[i + 1][1] if i + 1 < len(headers) else len(text)
        add(title, start, body_start, stop)
    return sections


def _memory_lines(body: str) -> set[str]:
    return {
        line.strip()
        for line in body.splitlines()
        if len(line.strip()) >= _MIN_SIGNIFICANT_LINE_CHARS
    }


def unseen_memory_sections(
    memory: str | None,
    history: Iterable[Any],
    max_output_chars: int,
) -> list[tuple[str, int, int]]:
    """Memory sections the root LM has not actually seen.

    Same rule as ``unseen_files``: a section is seen when at least
    ``COVERAGE_THRESHOLD`` of its significant body lines appear in the corpus.

    Returns:
        ``(title, start, end)`` of unseen sections (offsets into ``memory``).
    """
    sections = memory_sections(memory)
    if not sections:
        return []
    corpus, corpus_lines = _seen_corpus(history, max_output_chars)
    return [
        (title, start, end)
        for title, start, end, body in sections
        if not _lines_seen(_memory_lines(body), corpus, corpus_lines)
    ]


def memory_slices(
    memory: str, ranges: Iterable[tuple[int, int]], max_output_chars: int
) -> list[tuple[int, int]]:
    """Printable ``memory[a:b]`` slices covering ``ranges``.

    Ranges separated only by whitespace are merged; a range longer than the cap
    is split, at a line break when possible, so that ``print(memory[a:b])``
    (slice plus newline) stays under ``max_output_chars``.
    """
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and not memory[merged[-1][1]:start].strip():
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    limit = max(1, max_output_chars - 1)
    slices: list[tuple[int, int]] = []
    for start, end in merged:
        a = start
        while end - a > limit:
            cut = memory.rfind("\n", a + 1, a + limit + 1)
            b = cut + 1 if cut > a else a + limit
            slices.append((a, b))
            a = b
        if end > a:
            slices.append((a, end))
    return slices


def memory_check(memory: str | None, max_output_chars: int) -> SubmitCheck | None:
    """SUBMIT gate: send SUBMIT back while memory sections remain unseen.

    Returns None when ``memory`` is empty (nothing to gate on).
    """
    from codespy.agents.memory.prefrontal.prefrontal import PREFRONTAL_FIELD

    if not memory or not memory_sections(memory):
        return None

    def check(history: Any) -> str | None:
        unseen = unseen_memory_sections(memory, history, max_output_chars)
        if not unseen:
            return None
        slices = memory_slices(memory, [(s, e) for _, s, e in unseen], max_output_chars)
        return (
            f"{COVERAGE_REJECTION_PREFIX} Memory not read yet: "
            f"{', '.join(title for title, _, _ in unseen)}. "
            f"Print {', '.join(f'{PREFRONTAL_FIELD}[{a}:{b}]' for a, b in slices)} "
            f"(slices under the {max_output_chars:,}-char cap) before SUBMIT."
        )

    return check


def combine_checks(*checks: SubmitCheck | None) -> SubmitCheck | None:
    """Run every non-None check; join their messages (None when all pass).

    Returns None when no check is given, so callers can pass it straight to
    ``build_rlm_agent(submit_check=...)``.
    """
    active = [c for c in checks if c is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def check(history: Any) -> str | None:
        messages: list[str] = []
        for one in active:
            message = one(history)
            if not message:
                continue
            if message.startswith(COVERAGE_REJECTION_PREFIX):
                message = message[len(COVERAGE_REJECTION_PREFIX):].strip()
            messages.append(message)
        if not messages:
            return None
        return f"{COVERAGE_REJECTION_PREFIX} {' '.join(messages)}"

    return check


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
    memory: str | None = None,
) -> list[str]:
    """Log diagnostics for RLM outcomes, especially extract fallback cases.

    With ``changed_files``, ``max_output_chars`` or a non-empty ``memory``, also
    logs coverage telemetry at INFO: iterations used, end reason, SUBMIT gate
    rejections, truncated outputs, unseen changed files and (with ``memory``)
    ``memory_seen=<seen>/<total> sections`` plus the unseen section titles.
    Use ``unseen_memory_sections`` for the unseen sections themselves.

    Args:
        result: The prediction result from the RLM agent
        module: Module name (e.g., 'code_review', 'supply_chain')
        scope: Scope identifier for context
        changed_files: Changed files the run had to cover (optional)
        max_output_chars: Output cap the run used (optional)
        max_iters: Iteration budget the run used (optional, for the log)
        memory: ``prefrontal_memory`` text the run received (optional)

    Returns:
        Changed files not seen by the model (empty without ``changed_files``).
    """
    final_reasoning = getattr(result, "final_reasoning", None)
    unseen: list[str] = []
    if changed_files is not None or max_output_chars is not None or memory:
        trajectory = list(getattr(result, "trajectory", None) or [])
        cap = max_output_chars or DSPY_DEFAULT_MAX_OUTPUT_CHARS
        if changed_files is not None:
            unseen = unseen_files(changed_files, trajectory, cap)
        rejections = sum(
            1 for e in trajectory if _entry_field(e, "output").startswith(COVERAGE_REJECTION_PREFIX)
        )
        truncated = sum(1 for e in trajectory if len(_entry_field(e, "output")) > cap)
        memory_part = ""
        if memory:
            total = len(memory_sections(memory))
            unseen_memory = [t for t, _, _ in unseen_memory_sections(memory, trajectory, cap)]
            memory_part = (
                f", memory_seen={total - len(unseen_memory)}/{total} sections"
                f"{' (unseen: ' + ', '.join(unseen_memory) + ')' if unseen_memory else ''}"
            )
        logger.info(
            f"RLM[{module}] scope {scope}: iterations={len(trajectory)}/{max_iters or '?'}, "
            f"end={_end_reason(final_reasoning)}, coverage_rejections={rejections}, "
            f"truncated_outputs={truncated} (cap {cap:,}), "
            f"unseen_files={len(unseen)}{': ' + ', '.join(unseen) if unseen else ''}"
            f"{memory_part}"
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
