"""DSPy signatures configuration and environment variable handling."""

import logging
from typing import Literal

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


class SEOSignatureConfig(SignatureConfig):
    """Configuration for the SEO signature with scope filtering."""

    scopes: list[str] = Field(default_factory=list, description="fnmatch globs on scope.subroot; empty means all scopes")


class PRContextConfig(BaseModel):
    """PR conversation context (commits, comments, feedback on earlier reviews).

    Commits and comments are always on for remote reviews. ``feedback`` turns off
    learning from human reactions to codespy's findings (no suppression, no
    feedback input, no feedback memory, ``learn`` is a no-op): useful on open
    source projects where anyone can react.
    ``max_commits``/``max_comments`` are applied at fetch time,
    ``max_comment_chars``/``max_block_chars`` at render time.
    """

    feedback: bool = True
    # Accounts trusted as codespy when verifying who posted a finding, in addition
    # to the account of the token codespy runs with. Default covers reviews posted
    # from GitHub Actions while `learn` runs elsewhere (e.g. a personal token).
    trusted_authors: list[str] = Field(default_factory=lambda: ["github-actions[bot]"])

    max_commits: int = Field(default=50, ge=0)
    max_comments: int = Field(default=100, ge=0)
    max_comment_chars: int = Field(default=1000, ge=0)
    max_block_chars: int = Field(default=12000, ge=0)


class ReviewConfig(BaseModel):
    """Configuration for the review subsystem.

    Contains per-signature configs and top-level review settings.
    """

    # Minimum confidence threshold for reported issues
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Output settings
    output_format: OutputFormat = "markdown"
    output_stdout: bool = True  # Enable stdout output
    output_git: bool = True  # Enable Git platform review comments

    # Cache directory
    cache_dir: str = "~/.cache/codespy"

    # File exclusion settings
    excluded_directories: list[str] = Field(default=DEFAULT_EXCLUDED_DIRECTORIES)

    # PR conversation context size caps
    pr_context: PRContextConfig = Field(default_factory=PRContextConfig)

    # Concurrency limit for per-scope review modules
    max_concurrent_scopes: int = Field(default=8, ge=1)

    # Per-signature configs
    supply_chain: SignatureConfig = Field(
        default_factory=lambda: SignatureConfig(model="bedrock/converse/nvidia.nemotron-super-3-120b")
    )
    code_review: SignatureConfig = Field(
        default_factory=lambda: SignatureConfig(model="bedrock/converse/global.anthropic.claude-opus-5-5")
    )
    doc: SignatureConfig = Field(
        default_factory=lambda: SignatureConfig(model="bedrock/converse/global.anthropic.claude-opus-5-5")
    )
    scope: SignatureConfig = Field(
        default_factory=lambda: SignatureConfig(max_iters=3, max_llm_calls=10, model="bedrock/converse/global.anthropic.claude-opus-5-5")
    )
    summary: SignatureConfig = Field(
        default_factory=lambda: SignatureConfig(model="bedrock/converse/nvidia.nemotron-super-3-120b")
    )
    audit: SignatureConfig = Field(
        default_factory=lambda: SignatureConfig(model="bedrock/converse/global.anthropic.claude-opus-5-5")
    )
    seo: SEOSignatureConfig = Field(
        default_factory=lambda: SEOSignatureConfig(enabled=False, model="bedrock/converse/global.anthropic.claude-opus-5-5")
    )

    def signatures(self) -> dict[str, SignatureConfig]:
        """Return a dict of signature name -> SignatureConfig."""
        return {
            "supply_chain": self.supply_chain,
            "code_review": self.code_review,
            "doc": self.doc,
            "scope": self.scope,
            "summary": self.summary,
            "audit": self.audit,
            "seo": self.seo,
        }


# Known signature names derived from ReviewConfig
SIGNATURE_NAMES = set(ReviewConfig.model_fields.keys()) - {
    "min_confidence",
    "output_format",
    "output_stdout",
    "output_git",
    "cache_dir",
    "excluded_directories",
    "pr_context",
    "max_concurrent_scopes",
}
