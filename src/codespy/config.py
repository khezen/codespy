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


class GitHubConfig(BaseModel):
    """GitHub configuration."""

    token: str | None = Field(default=None, repr=False)


class ModuleConfig(BaseModel):
    """Configuration for a single review module."""

    enabled: bool = True
    max_iters: int | None = None
    model: str | None = None
    max_context_size: int | None = None


class ModulesConfig(BaseModel):
    """Configuration for all review modules."""

    # Individual module configs
    security_auditor: ModuleConfig = Field(default_factory=ModuleConfig)
    bug_detector: ModuleConfig = Field(default_factory=ModuleConfig)
    doc_reviewer: ModuleConfig = Field(default_factory=lambda: ModuleConfig(max_iters=15))
    domain_expert: ModuleConfig = Field(default_factory=lambda: ModuleConfig(max_iters=30))
    scope_identifier: ModuleConfig = Field(default_factory=lambda: ModuleConfig(max_iters=20))
    deduplicator: ModuleConfig = Field(default_factory=lambda: ModuleConfig(model="gpt-3.5-turbo"))
    summarizer: ModuleConfig = Field(default_factory=lambda: ModuleConfig(model="gpt-3.5-turbo"))

    def get_module_config(self, module_name: str) -> ModuleConfig:
        """Get config for a module by name."""
        return getattr(self, module_name, ModuleConfig())

    def is_enabled(self, module_name: str) -> bool:
        """Check if a module is enabled."""
        config = self.get_module_config(module_name)
        return config.enabled


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


# Known module names for env var routing (with their uppercase prefixes for matching)
_MODULE_NAMES = {
    "security_auditor",
    "bug_detector",
    "doc_reviewer",
    "domain_expert",
    "scope_identifier",
    "deduplicator",
    "summarizer",
}

# Create uppercase prefixes for matching (e.g., "SECURITY_AUDITOR_")
_MODULE_PREFIXES = {name.upper() + "_": name for name in _MODULE_NAMES}


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to config for module settings only.

    Only handles module-specific settings with known prefixes:
    - SECURITY_AUDITOR_MAX_ITERS -> config['modules']['security_auditor']['max_iters']
    - BUG_DETECTOR_MODEL -> config['modules']['bug_detector']['model']

    Top-level settings (DEFAULT_MODEL, AWS_REGION, etc.) are handled directly
    by pydantic-settings and should NOT be processed here.
    """
    import json

    for key, value in os.environ.items():
        key_upper = key.upper()

        # Only process module-specific settings
        module_name = None
        setting = None

        for prefix, mod_name in _MODULE_PREFIXES.items():
            if key_upper.startswith(prefix):
                module_name = mod_name
                setting = key[len(prefix):].lower()
                break

        # Skip if not a module setting
        if not module_name or not setting:
            continue

        # Ensure modules dict exists
        if "modules" not in config:
            config["modules"] = {}
        if module_name not in config["modules"]:
            config["modules"][module_name] = {}

        # Set the value (with type conversion)
        if value.lower() in ("true", "false"):
            config["modules"][module_name][setting] = value.lower() == "true"
        elif value.isdigit():
            config["modules"][module_name][setting] = int(value)
        elif value.startswith("[") or value.startswith("{"):
            try:
                config["modules"][module_name][setting] = json.loads(value)
            except json.JSONDecodeError:
                config["modules"][module_name][setting] = value
        elif value.lower() == "null" or value == "":
            config["modules"][module_name][setting] = None
        else:
            config["modules"][module_name][setting] = value

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
    modules: ModulesConfig = Field(default_factory=ModulesConfig)

    # Top-level defaults (also available via env vars DEFAULT_MODEL, etc.)
    default_model: str = "gpt-4o"
    default_max_iters: int = 10
    default_max_context_size: int = 50000

    # Top-level settings
    output_format: Literal["markdown", "json"] = "markdown"
    cache_dir: Path = Path.home() / ".cache" / "codespy"
    exclude_patterns: list[str] = Field(
        default=[
            "vendor/",
            "node_modules/",
            "third_party/",
            "external/",
            "deps/",
            ".bundle/",
            "Pods/",
            "Carthage/",
            "bower_components/",
            "jspm_packages/",
            "_vendor/",
            "vendored/",
        ]
    )
    include_vendor: bool = False

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

    # Helper methods for module config
    def get_effective_model(self, module_name: str) -> str:
        """Get effective model for a module (module-specific or default)."""
        config = self.modules.get_module_config(module_name)
        return config.model or self.default_model

    def get_effective_max_iters(self, module_name: str) -> int:
        """Get effective max_iters for a module (module-specific or default)."""
        config = self.modules.get_module_config(module_name)
        return config.max_iters or self.default_max_iters

    def get_effective_max_context_size(self, module_name: str) -> int:
        """Get effective max_context_size for a module (module-specific or default)."""
        config = self.modules.get_module_config(module_name)
        return config.max_context_size or self.default_max_context_size

    def log_module_configs(self) -> None:
        """Log the configuration for all modules."""
        module_names = [
            "scope_identifier",
            "security_auditor",
            "bug_detector",
            "doc_reviewer",
            "domain_expert",
            "deduplicator",
            "summarizer",
        ]
        logger.info("Module configurations:")
        for module_name in module_names:
            enabled = self.modules.is_enabled(module_name)
            model = self.get_effective_model(module_name)
            max_iters = self.get_effective_max_iters(module_name)
            status = "enabled" if enabled else "disabled"
            logger.info(f"  {module_name}: {status}, model={model}, max_iters={max_iters}")

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