"""Configuration management for codespy."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from codespy.config_dspy import ReviewConfig, RLMFallbackConfig, SignatureConfig
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
from codespy.config_llm import LLMConfig
from codespy.config_memory import (
    MEMORY_CONSOLIDATION,
    MEMORY_MENTAL_MODELS,
    MEMORY_RECALL,
    MEMORY_RETAIN,
    REFLECTION_MODULES,
    LLMSettings,
    MemoryConfig,
    ReflectionModuleConfig,
    reflection_module_config,
    reset_episode_store,
)
from codespy.config_utils import (
    apply_env_overrides,
    build_env_map,
    secret_value,
)

if TYPE_CHECKING:
    from codespy.agents.memory.hippocampus.budget import MemoryBudget


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
]


# Build the env map once at module level
# collapsed_paths drops the prefix from descendant env var names.
# full_name_paths keeps the full name for specific leaves (exceptions).
_ENV_MAP = build_env_map(
    sections={
        "llm": LLMConfig,
        "review": ReviewConfig,
        "memory": MemoryConfig,
    },
    collapsed_paths={
        ("llm",),
        ("memory", "hippocampus"),
        ("memory", "cerebral"),
        ("memory", "prefrontal"),
    },
    full_name_paths={
        ("llm", "retries"),
        ("llm", "timeout"),
        ("memory", "prefrontal", "reflects"),
    },
)


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
    review: ReviewConfig = Field(default_factory=ReviewConfig)

    # GitHub token (can also use GITHUB_TOKEN or GH_TOKEN env var)
    github_token: SecretStr | None = None
    gh_token: SecretStr | None = None
    github_auto_discover_token: bool = True  # GITHUB_AUTO_DISCOVER_TOKEN

    # GitLab token (can also use GITLAB_TOKEN or GITLAB_PRIVATE_TOKEN env var)
    gitlab_token: SecretStr | None = None
    gitlab_url: str = "https://gitlab.com"  # GITLAB_URL for self-hosted instances
    gitlab_auto_discover_token: bool = True  # GITLAB_AUTO_DISCOVER_TOKEN

    def get_signature_config(self, signature_name: str) -> SignatureConfig:
        """Get config for a signature.

        Only returns actual signature configs, not other review fields.
        """
        return getattr(self.review, signature_name, SignatureConfig())

    def is_signature_enabled(self, signature_name: str) -> bool:
        """Check if a signature is enabled."""
        return self.get_signature_config(signature_name).enabled

    def get_max_iters(self, name: str) -> int:
        """Get max_iters for a signature or reflection module (name-specific or default)."""
        if name in REFLECTION_MODULES:
            config = reflection_module_config(self.memory.hippocampus, name)
            return config.max_iters or self.llm.default_max_iters
        config = self.get_signature_config(name)
        return config.max_iters or self.llm.default_max_iters

    def get_max_llm_calls(self, name: str) -> int:
        """Get max_llm_calls for a signature or reflection module (name-specific or default)."""
        if name in REFLECTION_MODULES:
            config = reflection_module_config(self.memory.hippocampus, name)
            return config.max_llm_calls or self.llm.default_max_llm_calls
        config = self.get_signature_config(name)
        return config.max_llm_calls or self.llm.default_max_llm_calls

    def get_llm_config(self, name: str) -> LLMSettings:
        """Resolve the LLM settings for one named unit of LLM work.

        ``name`` addresses either a signature or a memory reflection module::

            "code_review"       -> review.code_review
            "memory_distiller"  -> memory.hippocampus.distiller
            "memory_retain"     -> memory.cerebral.retain
            "memory_consolidation" -> memory.cerebral.consolidation
            "memory_mental_models" -> memory.cerebral.mental_models

        Every field falls back to its ``llm.default_*`` counterpart, so
        the result has no ``None`` fields and callers never re-apply fallbacks.

        Args:
            name: A signature name, or a reflection module name
                (see ``REFLECTION_MODULES``), or "memory_retain",
                "memory_consolidation", "memory_mental_models".

        Returns:
            The fully resolved settings for ``name``.
        """
        if name == MEMORY_RETAIN:
            # Cerebral uses only a model from retain config; all other fields use llm defaults
            model = self.memory.cerebral.retain.model or self.llm.default_model
            config = ReflectionModuleConfig()  # Empty config - all fields fall back to defaults
            defaults = self.llm
        elif name == MEMORY_CONSOLIDATION:
            # Consolidation model: falls back to retain model, then default
            model = (
                self.memory.cerebral.consolidation.model
                or self.memory.cerebral.retain.model
                or self.llm.default_model
            )
            config = ReflectionModuleConfig()  # Empty config - all fields fall back to defaults
            defaults = self.llm
        elif name == MEMORY_RECALL:
            # Recall model: falls back to retain model, then default
            model = (
                self.memory.prefrontal.recall.model
                or self.memory.cerebral.retain.model
                or self.llm.default_model
            )
            config = ReflectionModuleConfig()  # Empty config - all fields fall back to defaults
            defaults = self.llm
        elif name == MEMORY_MENTAL_MODELS:
            # Mental-model refresh model: falls back to prefrontal.recall → retain → default
            model = (
                self.memory.cerebral.mental_models.model
                or self.memory.prefrontal.recall.model
                or self.memory.cerebral.retain.model
                or self.llm.default_model
            )
            config = ReflectionModuleConfig()  # Empty config - all fields fall back to defaults
            defaults = self.llm
        elif name in REFLECTION_MODULES:
            config = reflection_module_config(self.memory.hippocampus, name)
            model = config.model or self.llm.default_model
            defaults = self.llm
        elif name == "default":
            # Pure defaults for configure_dspy
            model = self.llm.default_model
            config = ReflectionModuleConfig()  # Empty config
            defaults = self.llm
        else:
            config = self.get_signature_config(name)
            model = config.model or self.llm.default_model
            defaults = self.llm

        return LLMSettings(
            model=model,
            extraction_model=self.llm.extraction_model or model,
            reasoning_effort=config.reasoning_effort or defaults.default_reasoning_effort,
            temperature=(
                config.temperature if config.temperature is not None else defaults.default_temperature
            ),
            max_tokens=config.max_tokens or defaults.default_max_tokens,
        )

    def get_scan_unchanged(self, signature_name: str) -> bool:
        """Get scan_unchanged for a signature (signature-specific, default: False).

        When True, scans all artifacts/manifests regardless of whether they changed.
        When False, only scans artifacts/manifests that were modified in the PR.
        """
        config = self.get_signature_config(signature_name)
        return config.scan_unchanged if config.scan_unchanged is not None else False

    def get_memory_budget(self) -> "MemoryBudget":
        """Resolve the ``MemoryBudget`` for memory operations.

        Token budgets are global (``memory.hippocampus.*``).
        """
        from codespy.agents.memory.hippocampus.budget import MemoryBudget

        return MemoryBudget(
            max_hippocampus_tokens=self.memory.hippocampus.max_hippocampus_tokens,
            max_hippocampus_item_tokens=self.memory.hippocampus.max_hippocampus_item_tokens,
            max_trajectory_tokens=self.memory.hippocampus.max_trajectory_tokens,
            max_question_tokens=self.memory.hippocampus.max_question_tokens,
            compact_trajectory=self.memory.hippocampus.compact_trajectory,
        )

    def get_scope_skip_refinement(self) -> bool:
        """Whether to skip LLM refinement when deterministic resolution is clean.

        Returns True (skip) when:
        - There's 1 scope and 0 orphans (single scope, all files assigned)
        - The setting skip_refinement_when_clean is True or not explicitly set

        Per-signature ``skip_refinement_when_clean`` overrides default (True).
        """
        val = self.get_signature_config("scope").skip_refinement_when_clean
        return val if val is not None else True  # default: skip

    def get_rlm_threshold(self, module_type: str) -> float:
        """Resolve RLM fallback threshold for a module type.

        Args:
            module_type: "react" | "chain_of_thought" | "predict"

        Returns:
            Threshold ratio (0.0-1.0), or 1.0 if disabled.
        """
        if not self.llm.rlm_fallback.enabled:
            return 1.0
        return getattr(self.llm.rlm_fallback, f"{module_type}_threshold", 1.0)

    def log_signature_configs(self) -> None:
        """Log all signature and reflection module LLM configurations."""
        logger.info("RLM fallback configuration:")
        logger.info(
            f"  enabled={self.llm.rlm_fallback.enabled}, "
            f"react_threshold={self.llm.rlm_fallback.react_threshold}, "
            f"chain_of_thought_threshold={self.llm.rlm_fallback.chain_of_thought_threshold}, "
            f"predict_threshold={self.llm.rlm_fallback.predict_threshold}"
        )
        logger.info("Signature configurations:")
        for sig_name, sig_config in self.review.signatures().items():
            status = "enabled" if sig_config.enabled else "disabled"
            llm = self.get_llm_config(sig_name)
            logger.info(
                f"  {sig_name}: {status}, model={llm.model}, "
                f"max_iters={self.get_max_iters(sig_name)}, "
                f"reasoning_effort={llm.reasoning_effort}, temperature={llm.temperature}, "
                f"max_tokens={llm.max_tokens}"
            )
        for module in REFLECTION_MODULES:
            llm = self.get_llm_config(module)
            logger.info(
                f"  {module}: model={llm.model}, "
                f"extraction_model={llm.extraction_model}, "
                f"max_iters={self.get_max_iters(module)}, "
                f"max_llm_calls={self.get_max_llm_calls(module)}, "
                f"reasoning_effort={llm.reasoning_effort}, temperature={llm.temperature}, "
                f"max_tokens={llm.max_tokens}"
            )
        # Prefrontal recall model (also refreshes briefings)
        pf_llm = self.get_llm_config(MEMORY_RECALL)
        logger.info(
            f"  {MEMORY_RECALL}: model={pf_llm.model}, "
            f"reflects={self.memory.prefrontal.reflects}"
        )
        # Consolidation model
        cons_llm = self.get_llm_config(MEMORY_CONSOLIDATION)
        logger.info(f"  {MEMORY_CONSOLIDATION}: model={cons_llm.model}")
        # Mental-models refresh model (only when reflects > 0)
        if self.memory.prefrontal.reflects > 0:
            mm_llm = self.get_llm_config(MEMORY_MENTAL_MODELS)
            logger.info(f"  {MEMORY_MENTAL_MODELS}: model={mm_llm.model}")

    @model_validator(mode="before")
    @classmethod
    def load_yaml_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load YAML config and merge with env vars.

        Priority: Environment Variables > YAML Config > Defaults
        """
        yaml_config = _load_yaml_config()
        # Apply unified env overrides before merging
        yaml_config = apply_env_overrides(yaml_config, _ENV_MAP)

        # Merge YAML config into values only if not already set (env vars take precedence)
        for key, val in yaml_config.items():
            if val is not None and key not in values:
                values[key] = val

        return values

    @model_validator(mode="after")
    def resolve_github_token(self) -> "Settings":
        """Auto-discover GitHub token if not explicitly set.

        Precedence: flat github_token (GITHUB_TOKEN env) → flat gh_token (GH_TOKEN env)
        → YAML github.token → auto-discovery.
        """

        def is_placeholder(token: str) -> bool:
            """Check if token looks like a placeholder."""
            placeholders = ["xxx", "your", "token", "example", "placeholder"]
            token_lower = token.lower()
            return any(p in token_lower for p in placeholders)

        def clear_placeholders():
            """Clear any placeholder values from both fields."""
            gh_val = secret_value(self.github_token)
            if gh_val and is_placeholder(gh_val):
                self.github_token = None
            nested_val = secret_value(self.github.token)
            if nested_val and is_placeholder(nested_val):
                self.github.token = None

        # First, clear any placeholders
        clear_placeholders()

        # Precedence 1: flat github_token (GITHUB_TOKEN env / .env)
        gh_val = secret_value(self.github_token)
        if gh_val:
            self.github.token = self.github_token  # SecretStr → SecretStr
            set_github_token_source("GITHUB_TOKEN environment variable or .env file")
            return self

        # Precedence 2: flat gh_token (GH_TOKEN env)
        gh_token_val = secret_value(self.gh_token)
        if gh_token_val:
            self.github_token = self.gh_token      # SecretStr → SecretStr
            self.github.token = self.gh_token
            set_github_token_source("GH_TOKEN environment variable")
            return self

        # Precedence 3: YAML github.token (already set if present)
        nested_val = secret_value(self.github.token)
        if nested_val:
            self.github_token = self.github.token  # SecretStr → SecretStr
            set_github_token_source("YAML config (github.token)")
            return self

        # Precedence 4: auto-discovery
        auto_discover = self.github.auto_discover_token and self.github_auto_discover_token

        if auto_discover:
            token, source = discover_github_token()
            if token and not is_placeholder(token):
                self.github_token = SecretStr(token)   # raw str → SecretStr
                self.github.token = SecretStr(token)
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
        """Auto-discover GitLab token if not explicitly set.

        URL precedence: GITLAB_URL env → YAML gitlab.url → default (https://gitlab.com)
        Token precedence: GITLAB_TOKEN env → YAML gitlab.token → auto-discovery
        """

        def is_placeholder(token: str) -> bool:
            """Check if token looks like a placeholder."""
            placeholders = ["xxx", "your", "token", "example", "placeholder"]
            token_lower = token.lower()
            return any(p in token_lower for p in placeholders)

        def clear_placeholders():
            """Clear any placeholder values from both fields."""
            token_val = secret_value(self.gitlab_token)
            if token_val and is_placeholder(token_val):
                self.gitlab_token = None
            nested_val = secret_value(self.gitlab.token)
            if nested_val and is_placeholder(nested_val):
                self.gitlab.token = None

        # Clear placeholders first
        clear_placeholders()

        # URL resolution: env wins over YAML
        # Check if GITLAB_URL came from env (via model_fields_set)
        if "gitlab_url" in self.model_fields_set:
            # Env var was set, keep it and sync to nested
            self.gitlab.url = self.gitlab_url
        else:
            # No env var, sync from YAML to flat (YAML wins over default)
            self.gitlab_url = self.gitlab.url

        # Precedence 1: flat gitlab_token (GITLAB_TOKEN env / .env)
        token_val = secret_value(self.gitlab_token)
        if token_val:
            self.gitlab.token = self.gitlab_token  # SecretStr → SecretStr
            set_gitlab_token_source("GITLAB_TOKEN environment variable or .env file")
            return self

        # Precedence 2: YAML gitlab.token (already set if present)
        nested_val = secret_value(self.gitlab.token)
        if nested_val:
            self.gitlab_token = self.gitlab.token  # SecretStr → SecretStr
            set_gitlab_token_source("YAML config (gitlab.token)")
            return self

        # Precedence 3: auto-discovery
        auto_discover = self.gitlab.auto_discover_token and self.gitlab_auto_discover_token

        if auto_discover:
            token, source = discover_gitlab_token()
            if token and not is_placeholder(token):
                self.gitlab_token = SecretStr(token)  # raw str → SecretStr
                self.gitlab.token = SecretStr(token)
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
        self.review.cache_dir = Path(self.review.cache_dir).expanduser().resolve()
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
    reset_episode_store()
    return settings
