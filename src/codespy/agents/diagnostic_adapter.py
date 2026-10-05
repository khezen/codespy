"""Diagnostic adapter for logging empty LM response details.

Wraps TwoStepAdapter to capture diagnostics when the main LM returns an empty
or null response, which helps diagnose why Claude occasionally returns empty
text with hidden thinking content.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]
from dspy.adapters.two_step_adapter import TwoStepAdapter  # type: ignore[import-untyped]
from dspy.adapters.base import AdapterParseError  # type: ignore[import-untyped]

from codespy.config import get_settings

logger = logging.getLogger(__name__)


def _getattr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    """Get attribute or dict key, tolerating both access patterns."""
    if hasattr(obj, name):
        return getattr(obj, name, default)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _safe_repr(value: Any, max_len: int = 500) -> str:
    """Safe repr with length limit."""
    try:
        s = repr(value)
        if len(s) > max_len:
            return s[:max_len] + f"... ({len(s)} chars total)"
        return s
    except Exception:
        return "<repr failed>"


def _get_finish_reason(entry: dict) -> str:
    """Extract finish_reason from history entry."""
    resp = entry.get("response")
    if resp is None:
        return "unknown"

    resp_class_name = type(resp).__name__
    if resp_class_name == "Response":
        # lm15 style Response object
        return _getattr_or_key(resp, "finish_reason", "unknown")
    else:
        # Standard dict-style response
        choices = _getattr_or_key(resp, "choices", [])
        if choices and len(choices) > 0:
            choice = choices[0]
            return _getattr_or_key(choice, "finish_reason", "unknown")
    return "unknown"


def _write_refusal_dump(entry: dict, model: str, lm_kwargs: dict, signature_name: str) -> Path | None:
    """Write a refusal dump to disk for later analysis.

    Args:
        entry: The history entry containing messages and response.
        model: The model name.
        lm_kwargs: The LM kwargs (will have API keys stripped).
        signature_name: The signature name for the dump filename.

    Returns:
        Path to the written dump file, or None if writing failed.
    """
    try:
        settings = get_settings()
        cache_dir = Path(settings.review.cache_dir).expanduser().resolve()
        refusals_dir = cache_dir / "refusals"
        refusals_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_sig = signature_name.replace("/", "_").replace("\\", "_")[:50]
        dump_path = refusals_dir / f"{timestamp}-{safe_sig}.json"

        # Strip API keys from lm_kwargs
        safe_kwargs = {}
        for key, value in lm_kwargs.items():
            lower_key = key.lower()
            # Redact actual secret keys, but not legitimate settings like max_tokens
            if lower_key.endswith("_key") or lower_key.endswith("_secret") or lower_key.endswith("_token") or lower_key == "api_key":
                safe_kwargs[key] = "<redacted>"
            else:
                safe_kwargs[key] = value

        dump = {
            "model": model,
            "lm_kwargs": safe_kwargs,
            "messages": entry.get("messages", []),
            "response": entry.get("response"),
        }

        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2, default=str)

        logger.warning("Empty LM response: refusal dump written to %s", dump_path)
        return dump_path
    except Exception as e:
        logger.debug("Empty LM response: failed to write refusal dump: %s", e)
        return None


def _log_empty_response(lm: Any, signature: Any) -> None:
    """Log detailed diagnostics for an empty LM response.

    Wraps everything in try/except so diagnostics can never mask the original error.
    """
    try:
        model = getattr(lm, "model", "unknown")
        max_tokens = getattr(lm, "kwargs", {}).get("max_tokens") if hasattr(lm, "kwargs") else None
        reasoning_effort = getattr(lm, "kwargs", {}).get("reasoning_effort") if hasattr(lm, "kwargs") else None

        # Get output field names from signature
        output_fields: list[str] = []
        signature_name = "unknown"
        if signature is not None:
            try:
                signature_name = getattr(signature, "__name__", type(signature).__name__)
                if hasattr(signature, "output_fields"):
                    output_fields = list(getattr(signature, "output_fields", []))
                elif hasattr(signature, "fields"):
                    # Try to identify output fields from signature fields
                    fields = getattr(signature, "fields", {})
                    if isinstance(fields, dict):
                        output_fields = [k for k, v in fields.items() if getattr(v, "is_input", False) is False]
            except Exception:
                pass

        logger.warning(
            "Empty LM response: model=%s max_tokens=%s reasoning_effort=%s output_fields=%s",
            model, max_tokens, reasoning_effort, output_fields
        )

        # Read from LM history
        history = getattr(lm, "history", None)
        if not history:
            logger.warning("Empty LM response: no LM history available")
            return

        try:
            entry = history[-1]
        except IndexError:
            logger.warning("Empty LM response: LM history is empty")
            return

        if not isinstance(entry, dict):
            logger.warning("Empty LM response: unexpected history entry type %s", type(entry).__name__)
            return

        # Log usage info
        usage = entry.get("usage")
        if usage:
            prompt_tokens = _getattr_or_key(usage, "prompt_tokens") or _getattr_or_key(usage, "input_tokens", 0)
            completion_tokens = _getattr_or_key(usage, "completion_tokens") or _getattr_or_key(usage, "output_tokens", 0)
            reasoning_tokens = _getattr_or_key(usage, "reasoning_tokens", 0)
            logger.warning(
                "Empty LM response: usage prompt_tokens=%s completion_tokens=%s reasoning_tokens=%s",
                prompt_tokens, completion_tokens, reasoning_tokens
            )
        else:
            logger.warning("Empty LM response: no usage info in history entry")

        # Log response details
        resp = entry.get("response")
        if resp is None:
            logger.warning("Empty LM response: no response object in history entry")
            return

        # Handle lm15 Response object vs dict
        resp_class_name = type(resp).__name__
        if resp_class_name == "Response":
            # lm15 style Response object
            finish_reason = _getattr_or_key(resp, "finish_reason", "unknown")
            message = _getattr_or_key(resp, "message", None)
            parts_info: list[str] = []
            if message and hasattr(message, "parts"):
                parts = getattr(message, "parts", [])
                parts_info = [type(p).__name__ for p in parts]
            usage_obj = _getattr_or_key(resp, "usage", None)
            logger.warning(
                "Empty LM response: lm15 Response finish_reason=%s message_parts=%s usage=%s",
                finish_reason, parts_info, usage_obj
            )
        else:
            # Standard dict-style response
            choices = _getattr_or_key(resp, "choices", [])
            if choices and len(choices) > 0:
                choice = choices[0]
                finish_reason = _getattr_or_key(choice, "finish_reason", "unknown")
                message = _getattr_or_key(choice, "message", {})
                content = _getattr_or_key(message, "content", "")

                logger.warning(
                    "Empty LM response: finish_reason=%s content=%s",
                    finish_reason, _safe_repr(content, 500)
                )

                # Check for reasoning_content (Anthropic)
                reasoning_content = _getattr_or_key(message, "reasoning_content", None)
                if reasoning_content:
                    rc_len = len(str(reasoning_content))
                    logger.warning(
                        "Empty LM response: reasoning_content len=%s first_300=%s",
                        rc_len, _safe_repr(str(reasoning_content)[:300], 300)
                    )
                else:
                    # Check for thinking_blocks
                    thinking_blocks = _getattr_or_key(message, "thinking_blocks", None)
                    if thinking_blocks:
                        block_types = [type(b).__name__ for b in thinking_blocks]
                        logger.warning(
                            "Empty LM response: thinking_blocks count=%s types=%s",
                            len(thinking_blocks), block_types
                        )
                    else:
                        logger.warning("Empty LM response: no reasoning_content or thinking_blocks")

                # Check for tool_calls
                tool_calls = _getattr_or_key(message, "tool_calls", None)
                if tool_calls:
                    logger.warning("Empty LM response: tool_calls count=%s", len(tool_calls))
                else:
                    logger.warning("Empty LM response: no tool_calls")

                # Log provider_specific_fields if present
                provider_fields = _getattr_or_key(message, "provider_specific_fields", None)
                if provider_fields:
                    logger.warning("Empty LM response: provider_specific_fields keys=%s", list(provider_fields.keys()) if isinstance(provider_fields, dict) else type(provider_fields).__name__)
            else:
                logger.warning("Empty LM response: no choices in response")

        # Log request size
        messages = entry.get("messages", [])
        request_size = len(str(messages))
        logger.warning("Empty LM response: request_size_chars=%s", request_size)

        # If finish_reason is content_filter, write a refusal dump
        finish_reason = _get_finish_reason(entry)
        if finish_reason == "content_filter":
            lm_kwargs = getattr(lm, "kwargs", {}) if hasattr(lm, "kwargs") else {}
            _write_refusal_dump(entry, model, lm_kwargs, signature_name)

    except Exception as diag_exc:
        # Never let diagnostics mask the original error
        logger.debug("Empty LM response: diagnostic logging failed: %s", diag_exc, exc_info=True)


def _get_finish_reason_from_lm(lm: Any) -> str | None:
    """Extract finish_reason from LM history entry."""
    history = getattr(lm, "history", None)
    if not history:
        return None
    try:
        entry = history[-1]
        if not isinstance(entry, dict):
            return None
        return _get_finish_reason(entry)
    except Exception:
        return None


class DiagnosticTwoStepAdapter(TwoStepAdapter):
    """TwoStepAdapter that logs diagnostics on empty LM responses.

    Wraps the parent's __call__ and __acall__ to catch AdapterParseError for empty/null
    responses and log detailed diagnostics before re-raising the original
    exception unchanged.
    """

    def _check_and_log_empty_response(
        self, exc: AdapterParseError, lm: dspy.LM, signature: Any
    ) -> None:
        """Check if exception is for empty response and log diagnostics.

        If finish_reason is content_filter, raises a more descriptive error.
        """
        # Only log for empty/null response errors
        if "empty or null response" in str(exc).lower():
            _log_empty_response(lm, signature)
            # If finish_reason is content_filter, raise a more descriptive error
            finish_reason = _get_finish_reason_from_lm(lm)
            if finish_reason == "content_filter":
                raise AdapterParseError(
                    adapter_name=getattr(self, "__class__.__name__", "DiagnosticTwoStepAdapter"),
                    signature=signature,
                    lm_response="",
                    message="model refused (content_filter)",
                ) from exc

    def __call__(
        self,
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
        signature: Any,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Call parent adapter, logging diagnostics on empty response."""
        try:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except AdapterParseError as exc:
            self._check_and_log_empty_response(exc, lm, signature)
            # Always re-raise the original exception unchanged
            raise

    async def acall(
        self,
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
        signature: Any,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Async call parent adapter, logging diagnostics on empty response."""
        try:
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)
        except AdapterParseError as exc:
            self._check_and_log_empty_response(exc, lm, signature)
            # Always re-raise the original exception unchanged
            raise
