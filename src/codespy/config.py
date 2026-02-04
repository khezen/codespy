"""Configuration management for codespy."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def discover_github_token() -> tuple[str | None, str]:
    """Try to discover GitHub token from local environment.

    Returns:
        Tuple of (token, source) where source describes where the token was found.
    """
    # 1. Check environment variables
    for env_var in ("GITHUB_TOKEN", "GH_TOKEN"):
        if token := os.environ.get(env_var):
            return token, f"environment variable ${env_var}"

    # 2. Try GitHub CLI (gh auth token)
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip(), "GitHub CLI (gh auth token)"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Try git credential helper
    try:
        result = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n",
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("password="):
                    return line.split("=", 1)[1], "git credential helper"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 4. Try .netrc file
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            content = netrc_path.read_text()
            in_github = False
            for line in content.split("\n"):
                line = line.strip()
                if "github.com" in line:
                    in_github = True
                if in_github and line.startswith("password"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1], "~/.netrc file"
                if in_github and line.startswith("machine") and "github.com" not in line:
                    in_github = False
        except Exception:
            pass

    return None, "not found"


# Global variable to track token source
_token_source: str = "not set"


def get_token_source() -> str:
    """Get the source of the GitHub token."""
    return _token_source


# ============================================================================
# YAML CONFIG MODELS
# ============================================================================


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    # OpenAI
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_api_base: str | None = None

    # Anthropic
    anthropic_api_key: str | None = Field(default=None, repr=False)

    # Google Gemini
    gemini_api_key: str | None = Field(default=None, repr=False)

    # AWS Bedrock
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = Field(default=None, repr=False)
    aws_secret_access_key: str | None = Field(default=None, repr=False)
    aws_profile: str | None = None

    # Azure OpenAI
    azure_api_key: str | None = Field(default=None, repr=False)
    azure_api_base: str | None = None
    azure_api_version: str | None = None

    # Enable provider-side prompt caching (Anthropic, OpenAI, Bedrock, etc.)
    enable_prompt_caching: bool = True


class GitHubConfig(BaseModel):
    """GitHub configuration."""

    token: str | None = Field(default=None, repr=False)


class SignatureConfig(BaseModel):
    """Configuration for a single signature."""

    enabled: bool = True
    max_iters: int | None = None
    model: str | None = None
    max_context_size: int | None = None


def _load_yaml_config() -> dict[str, Any]:
    """Load YAML config file if it exists."""
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


# Known signature names for env var routing
_SIGNATURE_NAMES = {
    "code_security",
    "supply_chain",
    "bug_detection",
    "doc_review",
    "domain_analysis",
    "scope_identification",
    "deduplication",
    "summarization",
}

# Create uppercase prefixes for matching (e.g., "CODE_SECURITY_")
_SIGNATURE_PREFIXES = {name.upper() + "_": name for name in _SIGNATURE_NAMES}

# Known signature settings for validation
_SIGNATURE_SETTINGS = {"enabled", "max_iters", "model", "max_context_size"}


def _convert_env_value(value: str) -> Any:
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


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to config for signature settings.

    Handles signature settings with pattern:
    - CODE_SECURITY_MAX_ITERS -> signatures.code_security.max_iters
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

        for prefix, sig_name in _SIGNATURE_PREFIXES.items():
            if key_upper.startswith(prefix):
                signature_name = sig_name
                setting = key_upper[len(prefix):].lower()
                break

        # Skip if not a signature setting or not a valid setting name
        if not signature_name or setting not in _SIGNATURE_SETTINGS:
            continue

        # Ensure signatures dict exists
        if "signatures" not in config:
            config["signatures"] = {}
        if signature_name not in config["signatures"]:
            config["signatures"][signature_name] = {}

        # Set the value
        config["signatures"][signature_name][setting] = _convert_env_value(value)

    return config


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

    # Flat signature configs (signature_name -> SignatureConfig)
    signatures: dict[str, SignatureConfig] = Field(default_factory=dict)

    # Top-level defaults (also available via env vars DEFAULT_MODEL, etc.)
    default_model: str = "gpt-5"
    default_max_iters: int = 10
    default_max_context_size: int = 50000

    # Enable provider-side prompt caching (Anthropic, OpenAI, Bedrock, etc.)
    # This caches system prompts on the LLM provider's servers to reduce latency and costs
    enable_prompt_caching: bool = True

    # Top-level settings
    output_format: Literal["markdown", "json"] = "markdown"
    cache_dir: Path = Path.home() / ".cache" / "codespy"

    # GitHub token (can also use GITHUB_TOKEN or GH_TOKEN env var)
    github_token: str = Field(default="", repr=False)
    gh_token: str = Field(default="", repr=False)

    # LLM provider settings (flat for simple env var access)
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = Field(default=None, repr=False)
    aws_secret_access_key: str | None = Field(default=None, repr=False)
    openai_api_key: str | None = Field(default=None, repr=False)
    anthropic_api_key: str | None = Field(default=None, repr=False)
    gemini_api_key: str | None = Field(default=None, repr=False)

    # Helper methods for signature config
    def get_signature_config(self, signature_name: str) -> SignatureConfig:
        """Get config for a signature."""
        return self.signatures.get(signature_name, SignatureConfig())

    def is_signature_enabled(self, signature_name: str) -> bool:
        """Check if a signature is enabled."""
        return self.get_signature_config(signature_name).enabled

    def get_model(self, signature_name: str) -> str:
        """Get model for a signature (signature-specific or default)."""
        config = self.get_signature_config(signature_name)
        return config.model or self.default_model

    def get_max_iters(self, signature_name: str) -> int:
        """Get max_iters for a signature (signature-specific or default)."""
        config = self.get_signature_config(signature_name)
        return config.max_iters or self.default_max_iters

    def get_max_context_size(self, signature_name: str) -> int:
        """Get max_context_size for a signature (signature-specific or default)."""
        config = self.get_signature_config(signature_name)
        return config.max_context_size or self.default_max_context_size

    def log_signature_configs(self) -> None:
        """Log all signature configurations."""
        logger.info("Signature configurations:")
        for sig_name, sig_config in self.signatures.items():
            status = "enabled" if sig_config.enabled else "disabled"
            model = sig_config.model or self.default_model
            max_iters = sig_config.max_iters or self.default_max_iters
            logger.info(f"  {sig_name}: {status}, model={model}, max_iters={max_iters}")

    @model_validator(mode="before")
    @classmethod
    def load_yaml_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load YAML config and merge with env vars.

        Priority: Environment Variables > YAML Config > Defaults
        """
        yaml_config = _load_yaml_config()
        yaml_config = _apply_env_overrides(yaml_config)

        # Merge YAML config into values only if not already set (env vars take precedence)
        for key, val in yaml_config.items():
            if val is not None and key not in values:
                values[key] = val

        return values

    @model_validator(mode="after")
    def resolve_github_token(self) -> "Settings":
        """Auto-discover GitHub token if not explicitly set."""
        global _token_source

        def is_placeholder(token: str) -> bool:
            """Check if token looks like a placeholder."""
            placeholders = ["xxx", "your", "token", "example", "placeholder"]
            token_lower = token.lower()
            return any(p in token_lower for p in placeholders)

        # First check nested config
        if self.github.token and not is_placeholder(self.github.token):
            self.github_token = self.github.token
            _token_source = "YAML config or GITHUB_TOKEN environment variable"
            return self

        # If github_token is set and not a placeholder, use it
        if self.github_token and not is_placeholder(self.github_token):
            self.github.token = self.github_token
            _token_source = "GITHUB_TOKEN environment variable or .env file"
            return self

        # If GH_TOKEN is set and not a placeholder, use it
        if self.gh_token and not is_placeholder(self.gh_token):
            self.github_token = self.gh_token
            self.github.token = self.gh_token
            _token_source = "GH_TOKEN environment variable"
            return self

        # Clear placeholder if present
        if self.github_token and is_placeholder(self.github_token):
            self.github_token = ""
            self.github.token = None

        # Try auto-discovery
        token, source = discover_github_token()
        if token and not is_placeholder(token):
            self.github_token = token
            self.github.token = token
            _token_source = source
            logger.debug(f"GitHub token discovered from: {source}")
        else:
            _token_source = "not found"

        return self

    @model_validator(mode="after")
    def expand_paths(self) -> "Settings":
        """Expand ~ in paths to the user's home directory."""
        self.cache_dir = Path(self.cache_dir).expanduser().resolve()
        return self

    @model_validator(mode="after")
    def sync_llm_settings(self) -> "Settings":
        """Sync LLM settings between nested and flat fields."""
        # Sync from nested to flat
        if self.llm.openai_api_key:
            self.openai_api_key = self.llm.openai_api_key
        if self.llm.anthropic_api_key:
            self.anthropic_api_key = self.llm.anthropic_api_key
        if self.llm.gemini_api_key:
            self.gemini_api_key = self.llm.gemini_api_key
        if self.llm.aws_region:
            self.aws_region = self.llm.aws_region
        if self.llm.aws_access_key_id:
            self.aws_access_key_id = self.llm.aws_access_key_id
        if self.llm.aws_secret_access_key:
            self.aws_secret_access_key = self.llm.aws_secret_access_key

        # Sync from flat to nested (for backward compat)
        if self.openai_api_key and not self.llm.openai_api_key:
            self.llm.openai_api_key = self.openai_api_key
        if self.anthropic_api_key and not self.llm.anthropic_api_key:
            self.llm.anthropic_api_key = self.anthropic_api_key
        if self.gemini_api_key and not self.llm.gemini_api_key:
            self.llm.gemini_api_key = self.gemini_api_key

        # Sync prompt caching setting (nested takes precedence if explicitly set in YAML)
        # Note: llm.enable_prompt_caching defaults to True, so we sync it to flat field
        self.enable_prompt_caching = self.llm.enable_prompt_caching

        return self


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the current settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings (useful after environment changes)."""
    global settings
    settings = Settings()
    return settings