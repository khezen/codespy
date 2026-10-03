"""Structured output support for Cerebral memory models.

Provides:
- Native structured output registration for Bedrock models that support it
  but lack the flag in litellm's model map.
- JSON schema sanitization for Bedrock's structured output constraints.
- Two-step structured output fallback for models that don't enforce schemas.

See: .kilo/plans/1790926015516-retain-facts-string-coercion.md
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Models that AWS documents as supporting native structured output on Bedrock,
# but which litellm's model map lacks the flag for.
# Format: bare_model_key -> litellm_provider (as passed to register_model)
# AWS model card: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-nvidia-nemotron-super-3-120b.html
_NATIVE_STRUCTURED_OUTPUT_OVERRIDES: dict[str, str] = {
    "nvidia.nemotron-super-3-120b": "bedrock_converse",
}

# Marker for idempotent wrapper installation
_SCHEMA_SANITIZER_MARKER = "_codespy_schema_sanitizer"


def register_native_structured_output_overrides() -> None:
    """Register native structured output capability for known-capable models.

    Idempotent: safe to call multiple times. Skips keys where
    supports_native_structured_output is already true.
    """
    import litellm
    from litellm.utils import supports_native_structured_output

    for bare_key, provider in _NATIVE_STRUCTURED_OUTPUT_OVERRIDES.items():
        try:
            # Check if already registered
            if supports_native_structured_output(bare_key, provider):
                logger.debug(
                    "Native structured output already registered for %s (%s)",
                    bare_key,
                    provider,
                )
                continue
        except Exception:
            # Model info not found - proceed with registration
            pass

        # Register the capability
        try:
            litellm.register_model(
                {
                    bare_key: {
                        "litellm_provider": provider,
                        "supports_native_structured_output": True,
                    }
                }
            )
            logger.debug(
                "Registered native structured output for %s (%s)", bare_key, provider
            )
        except Exception as e:
            logger.warning(
                "Failed to register native structured output for %s: %s", bare_key, e
            )

    # Verify registration for the primary model (with full litellm path prefix)
    _verify_registration("bedrock/converse/nvidia.nemotron-super-3-120b")


def _verify_registration(model: str) -> None:
    """Verify that native structured output is enabled for a model."""
    from litellm.utils import supports_native_structured_output

    try:
        if supports_native_structured_output(model):
            logger.debug("Verified native structured output for %s", model)
        else:
            logger.warning(
                "Native structured output NOT enabled for %s after registration", model
            )
    except Exception as e:
        logger.warning("Could not verify native structured output for %s: %s", model, e)


def _strip_disallowed_keywords(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip JSON Schema keywords disallowed by Bedrock.

    Bedrock's structured output accepts a subset of Draft 2020-12.
    Disallowed keywords are silently rejected with a 400.

    Strips:
    - minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf
    - minLength, maxLength, pattern
    - maxItems
    - minItems with value > 1 (minItems: 0 or 1 is allowed)
    - format values outside: date-time, time, date, duration, email,
      hostname, uri, ipv4, ipv6, uuid
    - additionalProperties values other than false (forced to false)

    Recurses through: properties, items, $defs, definitions, anyOf, allOf, oneOf.
    Keys under properties/$defs are names, not keywords - never stripped.

    Returns a deep copy; input is never mutated.
    """
    if not isinstance(schema, dict):
        return schema

    # Deep copy to avoid mutating input
    result = copy.deepcopy(schema)
    return _strip_in_place(result)


def _strip_in_place(schema: Any) -> Any:
    """In-place recursive strip. Internal helper."""
    if not isinstance(schema, dict):
        return schema

    allowed_formats = {
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "uri",
        "ipv4",
        "ipv6",
        "uuid",
    }

    # Build list of keys to strip from this level
    to_strip: list[str] = []
    modified: list[str] = []

    for key, value in schema.items():
        if key in (
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "pattern",
            "maxItems",
        ):
            to_strip.append(key)
            modified.append(key)
        elif key == "minItems":
            # Only strip if value > 1
            if isinstance(value, int) and value > 1:
                to_strip.append(key)
                modified.append(key)
        elif key == "format":
            if value not in allowed_formats:
                to_strip.append(key)
                modified.append(f"format:{value}")
        elif key == "additionalProperties":
            # Force to false if not already false
            if value is not False:
                schema[key] = False
                modified.append(f"additionalProperties->{value}")

    # Strip disallowed keys
    for key in to_strip:
        del schema[key]

    # Log what was stripped at this level
    if modified:
        logger.debug("Bedrock schema sanitizer stripped: %s", modified)

    # Recurse into sub-schemas
    # Keys under properties/$defs are names, not keywords - preserve them
    for key in ("properties", "$defs", "definitions"):
        if key in schema and isinstance(schema[key], dict):
            for sub_schema in schema[key].values():
                _strip_in_place(sub_schema)

    # Array items
    if "items" in schema:
        _strip_in_place(schema["items"])

    # Combiners
    for combiner in ("anyOf", "allOf", "oneOf"):
        if combiner in schema and isinstance(schema[combiner], list):
            for sub_schema in schema[combiner]:
                _strip_in_place(sub_schema)

    # AdditionalProperties (if it's a schema, not just false)
    if isinstance(schema.get("additionalProperties"), dict):
        _strip_in_place(schema["additionalProperties"])

    # PropertyNames
    if isinstance(schema.get("propertyNames"), dict):
        _strip_in_place(schema["propertyNames"])

    return schema


def sanitize_bedrock_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a JSON Schema for Bedrock's structured output constraints.

    Args:
        schema: The JSON Schema dict to sanitize.

    Returns:
        A deep copy with disallowed keywords stripped.
    """
    return _strip_disallowed_keywords(schema)


def install_bedrock_schema_sanitizer() -> None:
    """Install the Bedrock JSON schema sanitizer.

    Patches litellm's AmazonConverseConfig._create_output_config_for_response_format
    to sanitize schemas before Bedrock receives them.

    Idempotent: calling multiple times has no additional effect.
    """
    import functools

    try:
        from litellm.llms.bedrock.chat.converse_transformation import (
            AmazonConverseConfig,
        )
    except ImportError:
        logger.debug("AmazonConverseConfig not found; skipping schema sanitizer")
        return

    # Get the raw descriptor from the class __dict__ to handle staticmethod correctly
    raw_descriptor = AmazonConverseConfig.__dict__.get(
        "_create_output_config_for_response_format"
    )
    if raw_descriptor is None:
        logger.warning(
            "AmazonConverseConfig._create_output_config_for_response_format not found"
        )
        return

    # Unwrap staticmethod to get the actual function
    if isinstance(raw_descriptor, staticmethod):
        original = raw_descriptor.__func__
    else:
        original = raw_descriptor

    # Check if already installed (check on the underlying function)
    if getattr(original, _SCHEMA_SANITIZER_MARKER, False):
        logger.debug("Bedrock schema sanitizer already installed")
        return

    # The original signature is: (json_schema=None, name=None, description=None)
    @functools.wraps(original)
    def wrapper(
        json_schema: dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Wrapped staticmethod that sanitizes schemas before passing to Bedrock."""
        if json_schema is not None and isinstance(json_schema, dict):
            # Deep copy and sanitize the json_schema
            json_schema = _strip_disallowed_keywords(json_schema)

        return original(json_schema=json_schema, name=name, description=description)

    # Mark wrapper
    setattr(wrapper, _SCHEMA_SANITIZER_MARKER, True)

    # Install wrapper as staticmethod
    setattr(
        AmazonConverseConfig,
        "_create_output_config_for_response_format",
        staticmethod(wrapper),
    )
    logger.debug("Installed Bedrock JSON schema sanitizer")


def enforces_schema(model: str | None) -> bool | None:
    """Check if a model enforces structured output schemas natively.

    Returns:
        - True if the model natively enforces schemas
        - False if it uses fallback tool-based extraction
        - None if the model is unmapped (unknown to litellm)
    """
    from litellm.utils import supports_native_structured_output, supports_response_schema

    if model is None:
        return None

    # Parse provider from model string
    # Format: "provider/model" or "provider/converse/model" etc.
    parts = model.split("/")
    provider = parts[0] if parts else None

    # Determine which litellm function to use
    if provider in ("bedrock", "bedrock_converse"):
        # Use native structured output flag for Bedrock
        try:
            return supports_native_structured_output(model)
        except Exception:
            return None
    elif provider == "anthropic":
        # Anthropic direct
        try:
            return supports_native_structured_output(model)
        except Exception:
            return None
    elif provider == "vertex_ai":
        # Vertex AI - check if it's a Claude model
        if "claude" in model.lower():
            try:
                return supports_native_structured_output(model)
            except Exception:
                return None
        # Other Vertex models use response_schema
        try:
            return supports_response_schema(model)
        except Exception:
            return None
    else:
        # Other providers use response_schema
        try:
            return supports_response_schema(model)
        except Exception:
            return None


class TwoStepStructuredLLM:
    """Two-step structured output wrapper for non-schema-enforcing LLMs.

    First call: ask the main LLM to produce JSON matching a schema (in the prompt).
    Second call (if first fails validation): ask the extraction LLM to extract
    structured data from the first response.

    Inner calls are costed via the existing span recorder.
    """

    def __init__(
        self,
        main_impl: Any,
        extraction_impl: Any,
    ):
        """Initialize with main and extraction LLM implementations.

        Args:
            main_impl: The main LiteLLMLLM to wrap.
            extraction_impl: The extraction LiteLLMLLM for fallback.
        """
        self._main = main_impl
        self._extraction = extraction_impl

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the main LLM."""
        return getattr(self._main, name)

    async def call(
        self,
        messages: list[dict[str, Any]],
        response_format: type | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call with two-step structured output handling.

        Args:
            messages: The conversation messages.
            response_format: A pydantic model class for structured output.
            **kwargs: Additional args forwarded to inner calls.

        Returns:
            LLMCallResult from either the main or extraction LLM.
        """
        from hindsight_api.engine.llm_wrapper import (
            LLMCallResult,
            OutputTooLongError,
            parse_llm_json,
        )
        from hindsight_api.engine.response_models import TokenUsage
        from hindsight_api.engine.structured_output import provider_json_schema

        # If no response format, pass through directly (await the coroutine)
        if response_format is None:
            return await self._main.call(messages, response_format=None, **kwargs)

        # Check if response_format has a schema - use provider_json_schema for consistency
        model_schema = None
        try:
            if hasattr(response_format, "model_json_schema"):
                model_schema = provider_json_schema(response_format)
        except Exception:
            pass

        if model_schema is None:
            # No schema available, pass through
            return await self._main.call(messages, response_format=response_format, **kwargs)

        # Extract configuration values (forward all kwargs except ones we handle specially)
        scope = kwargs.get("scope", "unknown")
        skip_validation = kwargs.get("skip_validation", False)
        strict_schema = kwargs.get("strict_schema", False)

        # Build step 1 messages with schema instruction
        step1_messages = self._build_step1_messages(messages, model_schema)

        # Step 1: Call main LLM without response_format, with schema in prompt
        # Forward all kwargs except response_format, skip_validation, strict_schema
        step1_kwargs = {k: v for k, v in kwargs.items() if k not in ("response_format", "skip_validation", "strict_schema")}
        step1_result = await self._main.call(
            step1_messages,
            response_format=None,
            **step1_kwargs,
        )

        # LLMCallResult has `content`, not `text`
        step1_content = step1_result.content
        step1_text = step1_content if isinstance(step1_content, str) else str(step1_content)

        # Try to parse and validate step 1 result
        try:
            parsed = parse_llm_json(step1_text)
            if skip_validation:
                # Return raw dict as content
                return LLMCallResult(content=parsed, usage=step1_result.usage)
            else:
                # Validate against response_format
                validated = response_format.model_validate(parsed)
                # Return validated model as content
                return LLMCallResult(content=validated, usage=step1_result.usage)
        except Exception:
            # Step 1 failed validation or parsing - proceed to step 2
            pass

        # Step 2: Call extraction LLM
        extraction_messages = self._build_extraction_messages(step1_text, model_schema)

        # Step 2 overrides temperature=0.0 and adds response_format, skip_validation, strict_schema
        step2_kwargs = {k: v for k, v in kwargs.items() if k not in ("temperature", "response_format", "skip_validation", "strict_schema")}
        step2_result = await self._extraction.call(
            extraction_messages,
            response_format=response_format,
            temperature=0.0,
            skip_validation=skip_validation,
            strict_schema=strict_schema,
            **step2_kwargs,
        )

        # Let OutputTooLongError propagate (handled by caller)
        # Note: step2_result is LLMCallResult, not an exception
        # OutputTooLongError would have been raised, not returned

        # Sum usage from both calls using TokenUsage.__add__
        combined_usage = step1_result.usage + step2_result.usage

        # Return step 2 result with combined usage
        # Use model_copy (Pydantic v2) instead of _replace
        return step2_result.model_copy(update={"usage": combined_usage})

    def _build_step1_messages(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build messages for step 1 with schema instruction appended to system."""
        import json

        messages_copy = list(messages)

        # Find system message or create one
        system_idx = None
        for i, msg in enumerate(messages_copy):
            if msg.get("role") == "system":
                system_idx = i
                break

        schema_instruction = (
            f"\n\nRespond with a single JSON object matching this JSON schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )

        if system_idx is not None:
            # Append to existing system message
            original_content = messages_copy[system_idx].get("content", "")
            messages_copy[system_idx] = {
                **messages_copy[system_idx],
                "content": original_content + schema_instruction,
            }
        else:
            # Create new system message at the start
            messages_copy.insert(
                0, {"role": "system", "content": schema_instruction.strip()}
            )

        return messages_copy

    def _build_extraction_messages(
        self,
        text: str,
        schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build messages for extraction step."""
        import json

        system_prompt = (
            f"Extract the structured data from the response below into this JSON schema. "
            f"Copy values verbatim; do not invent, merge or drop items.\n\n"
            f"{json.dumps(schema, indent=2)}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

    # call_with_tools is forwarded via __getattr__ to self._main.call_with_tools


def install_two_step_structured_output(
    engine: Any,
    extraction_impl: Any,
) -> None:
    """Install two-step structured output wrapper on eligible LLM configs.

    Wraps engine LLM configs whose main model doesn't enforce schemas but
    the extraction model does.

    Args:
        engine: The MemoryEngine instance.
        extraction_impl: The extraction LLM implementation (LiteLLMLLM).
    """
    # Correct import paths per plan fix B1
    from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM
    from hindsight_api.engine.multi_llm import MultiLLMProvider

    # Collect unique LLM configs from the engine
    configs: set[int] = set()
    llm_configs: list[Any] = []

    for attr in (
        "_llm_config",
        "_retain_llm_config",
        "_consolidation_llm_config",
        "_reflect_llm_config",
        "_mental_model_refresh_llm_override",
    ):
        cfg = getattr(engine, attr, None)
        if cfg is not None and id(cfg) not in configs:
            configs.add(id(cfg))
            llm_configs.append(cfg)

    wrapped_count = 0
    for cfg in llm_configs:
        # Check if the config itself is a MultiLLMProvider (fix B2)
        # _build_llm returns MultiLLMProvider as the engine attribute
        if isinstance(cfg, MultiLLMProvider):
            logger.debug("Skipping MultiLLMProvider config for two-step wrapper")
            continue

        impl = getattr(cfg, "_provider_impl", None)

        if impl is None:
            logger.debug("No _provider_impl found on config")
            continue

        # Skip if the impl is a MultiLLMProvider
        if isinstance(impl, MultiLLMProvider):
            logger.debug("Skipping MultiLLMProvider implementation for two-step wrapper")
            continue

        if not isinstance(impl, LiteLLMLLM):
            logger.debug("Skipping non-LiteLLMLLM implementation: %s", type(impl))
            continue

        # Get the main model
        main_model = getattr(impl, "model", None)
        extraction_model = getattr(extraction_impl, "model", None)

        # Check if wrapping is needed
        main_enforces = enforces_schema(main_model)
        extraction_enforces = enforces_schema(extraction_model)

        logger.debug(
            "Checking two-step for model=%s (enforces=%s), extraction=%s (enforces=%s)",
            main_model,
            main_enforces,
            extraction_model,
            extraction_enforces,
        )

        # Wrap when main doesn't enforce but extraction does
        if main_enforces is False and extraction_enforces is True:
            wrapper = TwoStepStructuredLLM(impl, extraction_impl)
            cfg._provider_impl = wrapper
            wrapped_count += 1
            logger.info(
                "Installed two-step structured output wrapper for %s (extraction=%s)",
                main_model,
                extraction_model,
            )
        elif main_enforces is None:
            logger.debug(
                "Model %s is unmapped; skipping two-step wrapper", main_model
            )

    if wrapped_count > 0:
        logger.info("Installed two-step structured output on %d LLM config(s)", wrapped_count)
