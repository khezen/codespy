"""DSPy signatures configuration and environment variable handling."""

import logging
import os
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SignatureConfig(BaseModel):
    """Configuration for a single signature."""

    enabled: bool = True
    max_iters: int | None = None
    model: str | None = None
    max_context_size: int | None = None
    max_reasoning_tokens: int | None = None  # Limit reasoning verbosity for JSONAdapter reliability
    temperature: float | None = None  # Lower = more deterministic JSON output
    scan_unchanged: bool | None = None  # For supply_chain: scan unmodified artifacts/manifests


# Known signature names for env var routing
SIGNATURE_NAMES = {
    "bug_review",
    "doc_review",
    "smell_review",
    "supply_chain",
    "scope_identification",
    "deduplication",
    "summarization",
}

# Create uppercase prefixes for matching (e.g., "CODE_AND_DOC_REVIEW_")
SIGNATURE_PREFIXES = {name.upper() + "_": name for name in SIGNATURE_NAMES}

# Known signature settings for validation
SIGNATURE_SETTINGS = {"enabled", "max_iters", "model", "max_context_size", "max_reasoning_tokens", "temperature", "scan_unchanged"}


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

    Handles signature settings with pattern:
    - BUG_REVIEW_MAX_ITERS -> signatures.bug_review.max_iters
    - SUPPLY_CHAIN_ENABLED -> signatures.supply_chain.enabled

    Top-level settings (DEFAULT_MODEL, AWS_REGION, etc.) are handled directly
    by pydantic-settings and should NOT be processed here.
    """
    # Load .env file first to ensure env vars are available
    from dotenv import dotenv_values

    env_vars = {**dotenv_values(".env"), **os.environ}  # .env + actual env vars

    for key, value in env_vars.items():
        if value is None:
            continue
        key_upper = key.upper()

        # Only process signature-specific settings
        signature_name = None
        setting = None

        for prefix, sig_name in SIGNATURE_PREFIXES.items():
            if key_upper.startswith(prefix):
                signature_name = sig_name
                setting = key_upper[len(prefix) :].lower()
                break

        # Skip if not a signature setting or not a valid setting name
        if not signature_name or setting not in SIGNATURE_SETTINGS:
            continue

        # Ensure signatures dict exists
        if "signatures" not in config:
            config["signatures"] = {}
        if signature_name not in config["signatures"]:
            config["signatures"][signature_name] = {}

        # Set the value
        config["signatures"][signature_name][setting] = convert_env_value(value)

    return config