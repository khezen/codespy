"""Compatibility shim for Hindsight delta operations parsing.

The root cause of the "operations must be a list, got <class 'str'>" error:

When using models without native structured output support on Bedrock (e.g.
Nvidia Nemotron), litellm sends the schema as a forced synthetic
`json_tool_call` tool. The model returns the delta `operations` field as a
JSON-encoded string instead of a native array.

For example, instead of:
    {"operations": [{"op": "append_block", "section_id": "s1", "text": "..."}]}

The model returns:
    {"operations": '[{"op": "append_block", "section_id": "s1", "text": "..."}]'}

Hindsight's `parse_delta_operation_list` expects `operations` to be a list
and raises `TypeError` when it's a string. This shim intercepts the raw
payload before Hindsight validates it, decoding any JSON-encoded operations.

This shim can be removed once:
- Hindsight accepts string-encoded `operations` natively, OR
- The briefing model supports native structured output on Bedrock.

See: hindsight_api/engine/reflect/delta_ops.py::parse_delta_operation_list
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _normalize(raw: Any) -> tuple[Any, bool]:
    """Normalize delta operations payload, decoding JSON-encoded strings.

    Args:
        raw: The raw payload, which may be a dict, string, or other type.

    Returns:
        Tuple of (normalized_payload, changed). When `changed` is False,
        the original `raw` is returned unchanged (identity preserved).
    """
    # If raw is a string, try to parse it first
    if isinstance(raw, str):
        try:
            from hindsight_api.engine.llm_wrapper import parse_llm_json

            parsed = parse_llm_json(raw)
            return _normalize_dict_operations(parsed)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON - return unchanged, let original handler deal with it
            return raw, False

    # If raw is a dict, normalize its operations field
    if isinstance(raw, dict):
        return _normalize_dict_operations(raw)

    # For other types (e.g., list), return unchanged
    return raw, False


def _normalize_dict_operations(raw: dict[str, Any]) -> tuple[Any, bool]:
    """Normalize a dict payload, handling string-encoded operations.

    Handles these cases:
    1. {"operations": "[...]"} -> {"operations": [...]}
    2. {"operations": '{"operations": [...]}'} -> {"operations": [...]}
    3. {"operations": ["{...}", "{...}"]} -> {"operations": [{...}, {...}]}
    4. Well-formed payloads pass through unchanged
    """
    from hindsight_api.engine.llm_wrapper import parse_llm_json

    if "operations" not in raw:
        return raw, False

    ops = raw["operations"]

    # Case 1 & 2: operations is a string that might be JSON-encoded
    if isinstance(ops, str):
        try:
            decoded = parse_llm_json(ops)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON - log warning and return unchanged
            logger.warning(
                "cerebral: delta operations arrived as a non-JSON string; "
                "first 200 chars: %s",
                ops[:200],
            )
            return raw, False

        # If decoded is a list, use it directly
        if isinstance(decoded, list):
            new_raw = dict(raw)
            new_raw["operations"] = _decode_operation_strings(decoded)
            return new_raw, True

        # If decoded is a dict with operations, unwrap it
        if isinstance(decoded, dict) and "operations" in decoded:
            inner_ops = decoded["operations"]
            if isinstance(inner_ops, list):
                new_raw = dict(raw)
                new_raw["operations"] = _decode_operation_strings(inner_ops)
                return new_raw, True

        # Any other decoded type - return unchanged
        return raw, False

    # Case 3: operations is a list that might contain JSON-encoded strings
    if isinstance(ops, list):
        decoded_ops = _decode_operation_strings(ops)
        if decoded_ops is not ops:  # Identity check - was anything decoded?
            new_raw = dict(raw)
            new_raw["operations"] = decoded_ops
            return new_raw, True
        return raw, False

    return raw, False


def _decode_operation_strings(ops: list[Any]) -> list[Any]:
    """Decode any JSON-encoded operation strings in a list.

    Args:
        ops: List of operations, which may contain JSON-encoded strings.

    Returns:
        List with JSON strings decoded to dicts. Returns the original list
        if no decoding was needed (identity preserved).
    """
    from hindsight_api.engine.llm_wrapper import parse_llm_json

    changed = False
    result: list[Any] = []

    for item in ops:
        if isinstance(item, str):
            try:
                decoded = parse_llm_json(item)
                if isinstance(decoded, dict):
                    result.append(decoded)
                    changed = True
                else:
                    result.append(item)
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON - keep as-is
                result.append(item)
        else:
            result.append(item)

    return result if changed else ops


def _wrap_parse_delta_operation_list(original: Any) -> Any:
    """Wrap parse_delta_operation_list to handle string-encoded operations."""

    def wrapper(raw: Any) -> Any:
        # Normalize the payload before passing to original
        normalized, changed = _normalize(raw)

        if changed:
            # Log at debug level when decoding was applied
            ops = normalized.get("operations", []) if isinstance(normalized, dict) else []
            logger.debug(
                "cerebral: decoded JSON-encoded delta operations (%d op(s))",
                len(ops) if isinstance(ops, list) else 0,
            )

        # Call the original function with normalized payload
        return original(normalized)

    # Mark the wrapper so we can detect double-installation
    wrapper._codespy_delta_coercion = True  # type: ignore[attr-defined]
    return wrapper


def install_delta_ops_coercion() -> None:
    """Install the delta operations coercion shim.

    This patches hindsight_api.engine.reflect.delta_ops.parse_delta_operation_list
    to handle JSON-encoded operations strings before validation.

    Idempotent: calling multiple times has no additional effect.
    """
    from hindsight_api.engine.reflect import delta_ops

    original = delta_ops.parse_delta_operation_list

    # Check if already installed
    if getattr(original, "_codespy_delta_coercion", False):
        return

    # Install the wrapper
    delta_ops.parse_delta_operation_list = _wrap_parse_delta_operation_list(original)
    logger.debug("cerebral: installed delta operations coercion shim")
