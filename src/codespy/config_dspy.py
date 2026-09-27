"""DSPy signatures configuration and environment variable handling."""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from codespy.config_io import DEFAULT_EXCLUDED_DIRECTORIES, OutputFormat

logger = logging.getLogger(__name__)


# Reasoning budget hint sent to the provider. LiteLLM normalises this to each
# provider's native parameter (Anthropic thinking budget, OpenAI reasoning
# effort, ...), so it works across every supported model.
ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class RLMFallbackConfig(BaseModel):
    """Proactive RLM fallback thresholds to avoid context rot.

    When input tokens exceed threshold * max_input_tokens, ContextSafe
    switches to RLM before quality degrades. Thresholds differ by DSPy
    module type because multi-step reasoning (ReAct) is more vulnerable
    to context rot than single-pass (ChainOfThought/Predict).
    """

    enabled: bool = True
    react_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    chain_of_thought_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    predict_threshold: float = Field(default=0.50, ge=0.0, le=1.0)


class MemorySignatureConfig(BaseModel):
    """Per-signature Hippocampus memory overrides.

    All fields are optional — ``None`` means "use the global memory default"
    (see ``codespy.config_memory.MemoryConfig``).
    """

    enabled: bool | None = None  # <SIG>_MEMORY_ENABLED


class SignatureConfig(BaseModel):
    """Configuration for a single signature."""

    enabled: bool = True
    max_iters: int | None = None
    max_llm_calls: int | None = None  # RLM fallback max calls (default_max_llm_calls if None)
    model: str | None = None
    reasoning_effort: ReasoningEffort | None = None  # Provider reasoning budget
    temperature: float | None = None
    max_tokens: int | None = None  # Output token budget (reasoning tokens included)
    scan_unchanged: bool | None = None  # For supply_chain: scan unmodified artifacts/manifests
    skip_refinement_when_clean: bool | None = None  # For scope: skip LLM refinement when 1 scope, 0 orphans

    memory: MemorySignatureConfig = Field(default_factory=MemorySignatureConfig)


class ReviewConfig(BaseModel):
    """Configuration for the review subsystem.

    Contains per-signature configs and top-level review settings.
    """

    # Minimum confidence threshold for reported issues
    min_confidence: float = Field(default=0.81, ge=0.0, le=1.0)

    # Output settings
    output_format: OutputFormat = "markdown"
    output_stdout: bool = True  # Enable stdout output
    output_git: bool = True  # Enable Git platform review comments

    # Cache directory
    cache_dir: str = "~/.cache/codespy"

    # File exclusion settings
    excluded_directories: list[str] = Field(default=DEFAULT_EXCLUDED_DIRECTORIES)

    # Per-signature configs
    supply_chain: SignatureConfig = Field(default_factory=SignatureConfig)
    code_review: SignatureConfig = Field(default_factory=SignatureConfig)
    doc: SignatureConfig = Field(default_factory=lambda: SignatureConfig(max_iters=1, max_llm_calls=2))
    scope: SignatureConfig = Field(default_factory=lambda: SignatureConfig(max_iters=3, max_llm_calls=5))
    summary: SignatureConfig = Field(default_factory=lambda: SignatureConfig(max_iters=1, max_llm_calls=2))
    audit: SignatureConfig = Field(default_factory=lambda: SignatureConfig(max_iters=1, max_llm_calls=2))

    def signatures(self) -> dict[str, SignatureConfig]:
        """Return a dict of signature name -> SignatureConfig."""
        return {
            "supply_chain": self.supply_chain,
            "code_review": self.code_review,
            "doc": self.doc,
            "scope": self.scope,
            "summary": self.summary,
            "audit": self.audit,
        }


# Known signature names derived from ReviewConfig
SIGNATURE_NAMES = set(ReviewConfig.model_fields.keys()) - {
    "min_confidence",
    "output_format",
    "output_stdout",
    "output_git",
    "cache_dir",
    "excluded_directories",
}

# Create uppercase prefixes for matching (e.g., "CODE_REVIEW_", "SUPPLY_CHAIN_")
SIGNATURE_PREFIXES = {name.upper() + "_": name for name in SIGNATURE_NAMES}

# Known signature settings for validation, derived from the models so the env
# var routing can never drift from the declared fields. ``memory`` is excluded
# because it is nested and routed via <SIG>_MEMORY_<SETTING> instead.
SIGNATURE_SETTINGS = set(SignatureConfig.model_fields) - {"memory"}

# Known per-signature memory settings, routed via <SIG>_MEMORY_<SETTING>
MEMORY_SIGNATURE_SETTINGS = set(MemorySignatureConfig.model_fields)


# Env var name (without RLM_FALLBACK_ prefix) -> RLMFallbackConfig field name.
RLM_FALLBACK_ENV_SETTINGS = {
    "ENABLED": "enabled",
    "REACT_THRESHOLD": "react_threshold",
    "CHAIN_OF_THOUGHT_THRESHOLD": "chain_of_thought_threshold",
    "PREDICT_THRESHOLD": "predict_threshold",
}
