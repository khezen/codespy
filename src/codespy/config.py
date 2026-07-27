"""Configuration management for codespy."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from codespy.config_dspy import (
    ReasoningEffort,
    SignatureConfig,
    apply_signature_env_overrides,
)

from codespy.config_git import (
    GitHubConfig,
    GitLabConfig,
    discover_github_token,
    discover_gitlab_token,
    get_github_token_source,
    get_gitlab_token_source,
    set_github_token_source,
    set_gitlab_token_source,
)
from codespy.config_io import DEFAULT_EXCLUDED_DIRECTORIES, OutputFormat
from codespy.config_llm import (
    LLMConfig,
    discover_anthropic_api_key,
    discover_aws_credentials,
    discover_gemini_api_key,
    discover_openai_api_key,
)
from codespy.config_memory import (
    LLMSettings,
    MemoryConfig,
    REFLECTION_MODULES,
    apply_memory_env_overrides,
    reset_memory_store,
)





logger = logging.getLogger(__name__)

# Custom config path (set via CLI --config flag)
_custom_config_path: str | None = None

# Re-export for convenience
__all__ = [
    "Settings",
    "get_settings",
    "reload_settings",
    "get_github_token_source",
    "get_gitlab_token_source",
    "LLMConfig",
    "GitHubConfig",
    "GitLabConfig",
    "SignatureConfig",
    "MemoryConfig",
    "OutputFormat",
]


def _load_yaml_config() -> dict[str, Any]:
    """Load YAML config file if it exists.

    If _custom_config_path is set (via --config CLI flag), load from that
    exact path and raise FileNotFoundError if it doesn't exist.
    Otherwise, search the default locations.
    """
    if _custom_config_path is not None:
        path = Path(_custom_config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        logger.debug(f"Loading config from {path} (via --config)")
        with open(path) as f:
            return yaml.safe_load(f) or {}

    config_paths = [
        Path("codespy.yaml"),
        Path("codespy.yml"),
        Path.home() / ".config" / "codespy" / "config.yaml",
        Path.home() / ".config" / "codespy" / "config.yml",
    ]

    for path in config_paths:
        if path.exists():
            logger.debug(f"Loading config from {path}")
            with open(path) as f:
                return yaml.safe_load(f) or {}

    return {}


class Settings(BaseSettings):
    """Application settings loaded from YAML + environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested config sections
    llm: LLMConfig = Field(default_factory=LLMConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    gitlab: GitLabConfig = Field(default_factory=GitLabConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # Flat signature configs (signature_name -> SignatureConfig)
    signatures: dict[str, SignatureConfig] = Field(default_factory=dict)

    # Top-level defaults (also available via env vars DEFAULT_MODEL, etc.)
    default_model: str = "anthropic/claude-opus-4-6"
    extraction_model: str | None = None  # TwoStepAdapter extraction (falls back to default_model)
    default_max_iters: int = 3
    # Provider reasoning budget; LiteLLM maps this to each provider's native parameter.
    default_reasoning_effort: ReasoningEffort = "medium"
    # Providers require temperature=1 when reasoning is enabled.
    default_temperature: float = 1.0


    # Global LLM reliability settings
    llm_retries: int = 3  # Number of retries for LLM API calls
    llm_timeout: int = 240  # Timeout in seconds for LLM calls

    # Enable provider-side prompt caching (Anthropic, OpenAI, Bedrock, etc.)
    enable_prompt_caching: bool = True

    # Top-level settings
    output_format: OutputFormat = "markdown"
    cache_dir: Path = Path.home() / ".cache" / "codespy"

    # Output destinations
    output_stdout: bool = True  # Enable stdout output (markdown or json)
    output_git: bool = True  # Enable Git platform review comments (GitHub PR or GitLab MR)

    # File exclusion settings
    excluded_directories: list[str] = Field(default=DEFAULT_EXCLUDED_DIRECTORIES)

    # GitHub token (can also use GITHUB_TOKEN or GH_TOKEN env var)
    github_token: str = Field(default="", repr=False)
    gh_token: str = Field(default="", repr=False)
    github_auto_discover_token: bool = True  # GITHUB_AUTO_DISCOVER_TOKEN

    # GitLab token (can also use GITLAB_TOKEN or GITLAB_PRIVATE_TOKEN env var)
    gitlab_token: str = Field(default="", repr=False)
    gitlab_url: str = "https://gitlab.com"  # GITLAB_URL for self-hosted instances
    gitlab_auto_discover_token: bool = True  # GITLAB_AUTO_DISCOVER_TOKEN

    # LLM provider settings (flat for simple env var access)
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = Field(default=None, repr=False)
    aws_secret_access_key: str | None = Field(default=None, repr=False)
    aws_profile: str | None = None
    openai_api_key: str | None = Field(default=None, repr=False)
    anthropic_api_key: str | None = Field(default=None, repr=False)
    gemini_api_key: str | None = Field(default=None, repr=False)

    # Auto-discovery toggles (flat for env var access)
    auto_discover_aws: bool = True  # AUTO_DISCOVER_AWS
    auto_discover_openai: bool = True  # AUTO_DISCOVER_OPENAI
    auto_discover_anthropic: bool = True  # AUTO_DISCOVER_ANTHROPIC
    auto_discover_gemini: bool = True  # AUTO_DISCOVER_GEMINI

    # Helper methods for signature config
    def get_signature_config(self, signature_name: str) -> SignatureConfig:
        """Get config for a signature."""
        return self.signatures.get(signature_name, SignatureConfig())

    def is_signature_enabled(self, signature_name: str) -> bool:
        """Check if a signature is enabled."""
        return self.get_signature_config(signature_name).enabled

    def get_max_iters(self, signature_name: str) -> int:
        """Get max_iters for a signature (signature-specific or default)."""
        config = self.get_signature_config(signature_name)
        return config.max_iters or self.default_max_iters

    def get_llm_config(self, name: str) -> LLMSettings:
        """Resolve the LLM settings for one named unit of LLM work.

        ``name`` addresses either a signature or a memory reflection module::

            "code_review"  -> signatures.code_review
            "distiller"    -> memory.distiller

        Every field falls back to its top-level ``default_*`` counterpart, so
        the result has no ``None`` fields and callers never re-apply fallbacks.

        Args:
            name: A signature name, or a reflection module name
                (see ``REFLECTION_MODULES``).

        Returns:
            The fully resolved settings for ``name``.
        """
        if name in REFLECTION_MODULES:
            config = getattr(self.memory, name)
        else:
            config = self.get_signature_config(name)

        model = config.model or self.default_model
        # Only reflection modules carry their own extraction model; signatures
        # share the global one.
        module_extraction = getattr(config, "extraction_model", None)
        return LLMSettings(
            model=model,
            extraction_model=module_extraction or self.extraction_model or model,
            reasoning_effort=config.reasoning_effort or self.default_reasoning_effort,
            temperature=(
                config.temperature
                if config.temperature is not None
                else self.default_temperature
            ),
        )


    def get_scan_unchanged(self, signature_name: str) -> bool:
        """Get scan_unchanged for a signature (signature-specific, default: False).

        When True, scans all artifacts/manifests regardless of whether they changed.
        When False, only scans artifacts/manifests that were modified in the PR.
        """
        config = self.get_signature_config(signature_name)
        return config.scan_unchanged if config.scan_unchanged is not None else False

    # Helper methods for per-signature memory config (Hippocampus)
    def get_memory_enabled(self, signature_name: str) -> bool:
        """Whether Hippocampus memory is enabled for a signature.

        Per-signature ``memory.enabled`` overrides ``memory.default_enabled``.
        """
        config = self.get_signature_config(signature_name).memory
        return config.enabled if config.enabled is not None else self.memory.default_enabled

    def get_memory_max_reflects(self, signature_name: str) -> int | None:
        """Get max_reflects for a signature's memory (signature-specific or default)."""
        config = self.get_signature_config(signature_name).memory
        return (
            config.max_reflects
            if config.max_reflects is not None
            else self.memory.default_max_reflects
        )

    def get_memory_token_budget(self, signature_name: str) -> int:
        """Get token_budget for a signature's memory (signature-specific or default)."""
        config = self.get_signature_config(signature_name).memory
        return (
            config.token_budget
            if config.token_budget is not None
            else self.memory.default_token_budget
        )

    def get_memory_max_trajectory_tokens(self, signature_name: str) -> int | None:
        """Get max_trajectory_tokens for a signature's memory (signature-specific or default)."""
        config = self.get_signature_config(signature_name).memory
        return (
            config.max_trajectory_tokens
            if config.max_trajectory_tokens is not None
            else self.memory.default_max_trajectory_tokens
        )

    def log_signature_configs(self) -> None:
        """Log all signature and reflection module LLM configurations."""
        logger.info("Signature configurations:")
        for sig_name, sig_config in self.signatures.items():
            status = "enabled" if sig_config.enabled else "disabled"
            llm = self.get_llm_config(sig_name)
            logger.info(
                f"  {sig_name}: {status}, model={llm.model}, "
                f"max_iters={self.get_max_iters(sig_name)}, "
                f"reasoning_effort={llm.reasoning_effort}, temperature={llm.temperature}"
            )
        for module in REFLECTION_MODULES:
            llm = self.get_llm_config(module)
            logger.info(
                f"  {module}: model={llm.model}, "
                f"extraction_model={llm.extraction_model}, "
                f"reasoning_effort={llm.reasoning_effort}, temperature={llm.temperature}"
            )


    @model_validator(mode="before")
    @classmethod
    def load_yaml_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load YAML config and merge with env vars.

        Priority: Environment Variables > YAML Config > Defaults
        """
        yaml_config = _load_yaml_config()
        yaml_config = apply_signature_env_overrides(yaml_config)
        # MEMORY_* env vars target the nested `memory` model, which
        # pydantic-settings cannot populate on its own (no env_nested_delimiter).
        yaml_config = apply_memory_env_overrides(yaml_config)


        # Merge YAML config into values only if not already set (env vars take precedence)
        for key, val in yaml_config.items():
            if val is not None and key not in values:
                values[key] = val

        return values

    @model_validator(mode="after")
    def resolve_github_token(self) -> "Settings":
        """Auto-discover GitHub token if not explicitly set."""

        def is_placeholder(token: str) -> bool:
            """Check if token looks like a placeholder."""
            placeholders = ["xxx", "your", "token", "example", "placeholder"]
            token_lower = token.lower()
            return any(p in token_lower for p in placeholders)

        # First check nested config
        if self.github.token and not is_placeholder(self.github.token):
            self.github_token = self.github.token
            set_github_token_source("YAML config or GITHUB_TOKEN environment variable")
            return self

        # If github_token is set and not a placeholder, use it
        if self.github_token and not is_placeholder(self.github_token):
            self.github.token = self.github_token
            set_github_token_source("GITHUB_TOKEN environment variable or .env file")
            return self

        # If GH_TOKEN is set and not a placeholder, use it
        if self.gh_token and not is_placeholder(self.gh_token):
            self.github_token = self.gh_token
            self.github.token = self.gh_token
            set_github_token_source("GH_TOKEN environment variable")
            return self

        # Clear placeholder if present
        if self.github_token and is_placeholder(self.github_token):
            self.github_token = ""
            self.github.token = None

        # Try auto-discovery if enabled
        auto_discover = self.github.auto_discover_token and self.github_auto_discover_token

        if auto_discover:
            token, source = discover_github_token()
            if token and not is_placeholder(token):
                self.github_token = token
                self.github.token = token
                set_github_token_source(source)
                logger.debug(f"GitHub token discovered from: {source}")
            else:
                set_github_token_source("not found")
        else:
            set_github_token_source("auto-discovery disabled")
            logger.debug("GitHub token auto-discovery is disabled")

        return self

    @model_validator(mode="after")
    def resolve_gitlab_token(self) -> "Settings":
        """Auto-discover GitLab token if not explicitly set."""

        def is_placeholder(token: str) -> bool:
            """Check if token looks like a placeholder."""
            placeholders = ["xxx", "your", "token", "example", "placeholder"]
            token_lower = token.lower()
            return any(p in token_lower for p in placeholders)

        # First check nested config
        if self.gitlab.token and not is_placeholder(self.gitlab.token):
            self.gitlab_token = self.gitlab.token
            set_gitlab_token_source("YAML config or GITLAB_TOKEN environment variable")
            return self

        # Sync URL from nested config
        if self.gitlab.url:
            self.gitlab_url = self.gitlab.url

        # If gitlab_token is set and not a placeholder, use it
        if self.gitlab_token and not is_placeholder(self.gitlab_token):
            self.gitlab.token = self.gitlab_token
            set_gitlab_token_source("GITLAB_TOKEN environment variable or .env file")
            return self

        # Clear placeholder if present
        if self.gitlab_token and is_placeholder(self.gitlab_token):
            self.gitlab_token = ""
            self.gitlab.token = None

        # Try auto-discovery if enabled
        auto_discover = self.gitlab.auto_discover_token and self.gitlab_auto_discover_token

        if auto_discover:
            token, source = discover_gitlab_token()
            if token and not is_placeholder(token):
                self.gitlab_token = token
                self.gitlab.token = token
                set_gitlab_token_source(source)
                logger.debug(f"GitLab token discovered from: {source}")
            else:
                set_gitlab_token_source("not found")
        else:
            set_gitlab_token_source("auto-discovery disabled")
            logger.debug("GitLab token auto-discovery is disabled")

        return self

    @model_validator(mode="after")
    def expand_paths(self) -> "Settings":
        """Expand ~ in paths to the user's home directory."""
        self.cache_dir = Path(self.cache_dir).expanduser().resolve()
        return self

    @model_validator(mode="after")
    def sync_llm_settings(self) -> "Settings":
        """Sync LLM settings between nested and flat fields.

        Delegates to LLMConfig.sync_from_flat which enforces:
        env vars (flat fields from pydantic-settings) > YAML/nested > defaults.
        """
        merged = self.llm.sync_from_flat(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
            gemini_api_key=self.gemini_api_key,
            aws_region=self.aws_region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        # Propagate merged values back to flat fields
        for field, value in merged.items():
            if value is not None:
                setattr(self, field, value)
        return self

    @model_validator(mode="after")
    def resolve_llm_credentials(self) -> "Settings":
        """Auto-discover LLM provider credentials if not explicitly set."""

        def is_placeholder(value: str | None) -> bool:
            """Check if value looks like a placeholder."""
            if not value:
                return True
            placeholders = ["xxx", "your", "key", "example", "placeholder", "null"]
            value_lower = value.lower()
            return any(p in value_lower for p in placeholders)

        # AWS credentials auto-discovery
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            auto_discover = self.llm.auto_discover_aws and self.auto_discover_aws
            if auto_discover:
                access_key, secret_key, region, profile, source = discover_aws_credentials()
                if access_key and secret_key:
                    self.aws_access_key_id = access_key
                    self.aws_secret_access_key = secret_key
                    self.llm.aws_access_key_id = access_key
                    self.llm.aws_secret_access_key = secret_key
                    if region:
                        self.aws_region = region
                        self.llm.aws_region = region
                    if profile:
                        self.aws_profile = profile
                        self.llm.aws_profile = profile
                    logger.debug(f"AWS credentials discovered from: {source}")
            else:
                logger.debug("AWS credentials auto-discovery is disabled")

        # OpenAI API key auto-discovery
        if not self.openai_api_key or is_placeholder(self.openai_api_key):
            auto_discover = self.llm.auto_discover_openai and self.auto_discover_openai
            if auto_discover:
                key, source = discover_openai_api_key()
                if key and not is_placeholder(key):
                    self.openai_api_key = key
                    self.llm.openai_api_key = key
                    logger.debug(f"OpenAI API key discovered from: {source}")
            else:
                logger.debug("OpenAI API key auto-discovery is disabled")

        # Anthropic API key auto-discovery
        if not self.anthropic_api_key or is_placeholder(self.anthropic_api_key):
            auto_discover = self.llm.auto_discover_anthropic and self.auto_discover_anthropic
            if auto_discover:
                key, source = discover_anthropic_api_key()
                if key and not is_placeholder(key):
                    self.anthropic_api_key = key
                    self.llm.anthropic_api_key = key
                    logger.debug(f"Anthropic API key discovered from: {source}")
            else:
                logger.debug("Anthropic API key auto-discovery is disabled")

        # Gemini API key auto-discovery
        if not self.gemini_api_key or is_placeholder(self.gemini_api_key):
            auto_discover = self.llm.auto_discover_gemini and self.auto_discover_gemini
            if auto_discover:
                key, source = discover_gemini_api_key()
                if key and not is_placeholder(key):
                    self.gemini_api_key = key
                    self.llm.gemini_api_key = key
                    logger.debug(f"Gemini API key discovered from: {source}")
                elif source != "not found":
                    logger.debug(f"Gemini: {source}")
            else:
                logger.debug("Gemini API key auto-discovery is disabled")

        return self


# Global settings instance
settings = Settings()


def get_settings(config_file: str | None = None) -> Settings:
    """Get the current settings instance.

    Args:
        config_file: Optional path to a YAML config file. If provided,
            reloads settings using that file instead of the default locations.

    Raises:
        FileNotFoundError: If config_file is provided but does not exist.
    """
    global settings, _custom_config_path
    if config_file is not None:
        # Validate early (before pydantic) to avoid leaking secrets in tracebacks
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        _custom_config_path = config_file
        settings = Settings()
    return settings


def reload_settings(config_file: str | None = None) -> Settings:
    """Reload settings (useful after environment changes).

    Also resets the cached Hippocampus memory store so a changed
    ``memory`` configuration takes effect on next access.

    Args:
        config_file: Optional path to a YAML config file. If provided,
            uses that file instead of the default locations.
    """
    global settings, _custom_config_path
    if config_file is not None:
        _custom_config_path = config_file
    settings = Settings()
    reset_memory_store()
    return settings
