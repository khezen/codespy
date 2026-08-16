"""DSPy signatures configuration and environment variable handling."""

import logging
import os
from typing import Any, Literal


from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Reasoning budget hint sent to the provider. LiteLLM normalises this to each
# provider's native parameter (Anthropic thinking budget, OpenAI reasoning
# effort, ...), so it works across every supported model.
ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class MemorySignatureConfig(BaseModel):

    """Per-signature Hippocampus memory overrides.

    All fields are optional — ``None`` means "use the global memory default"
    (see ``codespy.config_memory.MemoryConfig``).
    """

    enabled: bool | None = None                   # <SIG>_MEMORY_ENABLED
    max_reflects: int | None = None               # <SIG>_MEMORY_MAX_REFLECTS


class SignatureConfig(BaseModel):
    """Configuration for a single signature."""

    enabled: bool = True
    max_iters: int | None = None
    model: str | None = None
    reasoning_effort: ReasoningEffort | None = None  # Provider reasoning budget
    temperature: float | None = None  # Must be 1 when reasoning is enabled
    max_tokens: int | None = None  # Output token budget (reasoning tokens included)
    scan_unchanged: bool | None = None  # For supply_chain: scan unmodified artifacts/manifests

    memory: MemorySignatureConfig = Field(default_factory=MemorySignatureConfig)


# Known signature names for env var routing
SIGNATURE_NAMES = {
    "code_review",
    "doc",
    "scope",
    "supply_chain",
    "summary",
    "audit",
}

# Create uppercase prefixes for matching (e.g., "CODE_REVIEW_", "SUPPLY_CHAIN_")
SIGNATURE_PREFIXES = {name.upper() + "_": name for name in SIGNATURE_NAMES}

# Known signature settings for validation, derived from the models so the env
# var routing can never drift from the declared fields. ``memory`` is excluded
# because it is nested and routed via <SIG>_MEMORY_<SETTING> instead.
SIGNATURE_SETTINGS = set(SignatureConfig.model_fields) - {"memory"}

# Known per-signature memory settings, routed via <SIG>_MEMORY_<SETTING>
MEMORY_SIGNATURE_SETTINGS = set(MemorySignatureConfig.model_fields)



def convert_env_value(value: str) -> Any:
    """Convert environment variable string to appropriate Python type."""
    import json

    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    elif value.isdigit():
        return int(value)
    elif value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    elif value.lower() == "null" or value == "":
        return None
    return value


def apply_signature_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to config for signature settings.

    Handles three patterns:
    - ``CODE_REVIEW_MAX_ITERS``      -> signatures.code_review.max_iters
    - ``SUPPLY_CHAIN_ENABLED``       -> signatures.supply_chain.enabled
    - ``CODE_REVIEW_MEMORY_ENABLED`` -> signatures.code_review.memory.enabled
    - ``SCOPE_MEMORY_MAX_CONTEXT_MEMORY_TOKENS`` -> signatures.scope.memory.max_context_memory_tokens
    - ``SCOPE_MEMORY_MAX_CONTEXT_ITEM_TOKENS`` -> signatures.scope.memory.max_context_item_tokens

    Top-level settings (DEFAULT_MODEL, AWS_REGION, MEMORY_DEFAULT_ENABLED, etc.)
    are handled directly by pydantic-settings and should NOT be processed here.
    """
    # Load .env file first to ensure env vars are available
    from dotenv import dotenv_values

    env_vars = {**dotenv_values(".env"), **os.environ}  # .env + actual env vars

    for key, value in env_vars.items():
        if value is None:
            continue
        key_upper = key.upper()

        # Match signature prefix
        signature_name = None
        remainder = None
        for prefix, sig_name in SIGNATURE_PREFIXES.items():
            if key_upper.startswith(prefix):
                signature_name = sig_name
                remainder = key_upper[len(prefix):]
                break

        if not signature_name or remainder is None:
            continue

        # Nested memory setting: <SIG>_MEMORY_<SETTING>
        if remainder.startswith("MEMORY_"):
            memory_setting = remainder[len("MEMORY_"):].lower()
            if memory_setting not in MEMORY_SIGNATURE_SETTINGS:
                continue
            if "signatures" not in config:
                config["signatures"] = {}
            if signature_name not in config["signatures"]:
                config["signatures"][signature_name] = {}
            if "memory" not in config["signatures"][signature_name]:
                config["signatures"][signature_name]["memory"] = {}
            sig_memory = config["signatures"][signature_name]["memory"]
            sig_memory[memory_setting] = convert_env_value(value)
            continue

        # Flat signature setting
        setting = remainder.lower()
        if setting not in SIGNATURE_SETTINGS:
            continue
        if "signatures" not in config:
            config["signatures"] = {}
        if signature_name not in config["signatures"]:
            config["signatures"][signature_name] = {}
        config["signatures"][signature_name][setting] = convert_env_value(value)

    return config
