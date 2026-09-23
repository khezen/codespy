"""Memory (Hippocampus) configuration and storage factory."""

from __future__ import annotations

import logging
import os

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from codespy.config_dspy import ReasoningEffort

if TYPE_CHECKING:
    from codespy.agents.memory.postgres import EpisodeStore
    from codespy.config import Settings

logger = logging.getLogger(__name__)


# Default embedding model per LLM provider prefix. Used by get_cerebral()
# when hindsight.embeddings_model is None. litellm-sdk routes through litellm.
EMBEDDING_MODELS: dict[str, str] = {
    "bedrock": "bedrock/cohere.embed-multilingual-v3",
    "openai": "openai/text-embedding-3-small",
    "anthropic": "openai/text-embedding-3-small",
    "gemini": "gemini/text-embedding-004",
    "azure": "azure/text-embedding-3-small",
    "litellm": "openai/text-embedding-3-small",
}


class HindsightConfig(BaseModel):
    """Hindsight MemoryEngine settings for Cerebral semantic memory."""
    embeddings_model: str | None = None  # MEMORY_HINDSIGHT_EMBEDDINGS_MODEL


class ReflectionModuleConfig(BaseModel):
    """LLM overrides for a single reflection module (Distiller / Cartographer).

    All fields are optional — ``None`` means "fall back to the corresponding
    top-level ``default_*`` setting" (see ``codespy.config.Settings``).

    The reflection modules are compact summarize/curate tasks rather than deep
    analysis, so they are good candidates for a cheaper model tier than the
    one used for code review.
    """

    model: str | None = None  # MEMORY_<MODULE>_MODEL
    reasoning_effort: ReasoningEffort | None = None  # MEMORY_<MODULE>_REASONING_EFFORT
    temperature: float | None = None  # MEMORY_<MODULE>_TEMPERATURE
    max_tokens: int | None = None  # MEMORY_<MODULE>_MAX_TOKENS
    max_iters: int | None = 1  # MEMORY_<MODULE>_MAX_ITERS
    max_llm_calls: int | None = 2  # MEMORY_<MODULE>_MAX_LLM_CALLS


class LLMSettings(BaseModel):
    """Fully resolved LLM settings for one named unit of work.

    Produced by ``Settings.get_llm_config()`` for either a signature
    (``signatures.<name>``) or a reflection module (``memory.<name>``): every
    field is either the name-specific override or the corresponding top-level
    default, so consumers never re-apply fallback logic.
    """

    model: str
    extraction_model: str
    reasoning_effort: ReasoningEffort
    temperature: float
    # Output token budget for a single completion. Reasoning/thinking tokens are
    # charged against it, so it must comfortably exceed the expected answer size.
    # ``new_lm`` clamps this to the model's real output ceiling before use.
    max_tokens: int


class PostgresConfig(BaseModel):
    """External PostgreSQL connection settings (production)."""
    host: str | None = None      # MEMORY_POSTGRES_HOST
    port: int = 5432             # MEMORY_POSTGRES_PORT
    user: str | None = None      # MEMORY_POSTGRES_USER
    password: str | None = None  # MEMORY_POSTGRES_PASSWORD
    database: str = "codespy"    # MEMORY_POSTGRES_DATABASE
    schema: str | None = "episodic"  # MEMORY_POSTGRES_SCHEMA (search_path per memory type; None = public)

    def build_uri(self) -> str | None:
        """Build a psycopg connection URI. Returns None when host is unset.

        Schema is NOT included in the URI. Each memory store (EpisodeStore,
        future SemanticStore) handles CREATE SCHEMA and SET search_path
        itself, so multiple stores can share the same base URI while
        targeting different schemas.
        """
        if not self.host:
            return None
        from urllib.parse import quote_plus
        user = quote_plus(self.user) if self.user else "postgres"
        cred = f"{user}:{quote_plus(self.password)}" if self.password else user
        return f"postgresql://{cred}@{self.host}:{self.port}/{self.database}"


class Pg0Config(BaseModel):
    """pg0-embedded settings (local dev only, ignored when postgres.host is set)."""
    name: str = "codespy"        # MEMORY_PG0_NAME
    port: int | None = None      # MEMORY_PG0_PORT
    data_dir: str | None = None  # MEMORY_PG0_DATA_DIR


class MemoryConfig(BaseModel):
    """Global memory (Hippocampus + Cerebral) configuration.

    Controls where episodes are persisted and the memory knob applied to
    every agent.  Per-signature ``memory:`` blocks override ``enabled``.
    """

    # PostgreSQL connection settings
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    pg0: Pg0Config = Field(default_factory=Pg0Config)
    bank_id: str | None = None  # MEMORY_BANK_ID (defaults to "codespy")

    # Master switch — overridable per-signature via signatures.<name>.memory.enabled
    enabled: bool = False  # MEMORY_ENABLED

    # Whether to apply head+tail trajectory bounding before distillation.
    # When false, the full trajectory goes to the Distiller and ContextSafe
    # RLM fallback handles overflow if it exceeds the model's context window.
    compact_trajectory: bool = True  # MEMORY_COMPACT_TRAJECTORY

    # Ceiling on the rendered ContextMemory. This is the *persisted* artifact and it
    # is prepended to every predictor of the wrapped agent, so it is re-sent on
    # every ReAct iteration (~default_max_iters times per scope) plus once per
    # reflection call. Easily the most cost-sensitive of the three budgets.
    # Approximate item capacity is max_context_memory_tokens divided by
    # max_context_item_tokens (16384 / 512 = 32 items).
    # MEMORY_MAX_CONTEXT_MEMORY_TOKENS
    max_context_memory_tokens: int = Field(default=16384)

    # Per-item ceiling handed to the Distiller/Cartographer as a prompt input, so
    # they keep each context-memory item compact instead of spending the whole memory
    # budget on one verbose entry. Soft limit: it is expressed to the LLM rather
    # than enforced in code (truncating an item could corrupt an exact constant).
    # The hard, memory-wide limit is max_context_memory_tokens, enforced by the
    # Evictor. MEMORY_MAX_CONTEXT_ITEM_TOKENS
    max_context_item_tokens: int = Field(default=512)

    # Head+tail cap on the agent trajectory fed to the Distiller. Without it a
    # single tool-heavy scope can produce a 100k+ token trajectory; TwoStepAdapter
    # then sends it twice. 16384 is ~12% of a 128k window and preserves both the
    # orientation steps (60% head) and the conclusions (40% tail).
    max_trajectory_tokens: int | None = 16384  # MEMORY_MAX_TRAJECTORY_TOKENS

    # Head+tail cap on the serialized agent inputs used as the Distiller/Cartographer
    # "question". Only applies when the caller passes no 'question': otherwise
    # every input field is serialized, which for code review means the full patch
    # of every changed file. See Hippocampus.max_question_tokens.
    max_question_tokens: int | None = 8192  # MEMORY_MAX_QUESTION_TOKENS

    # Per-module LLM overrides for the reflection pipeline.
    distiller: ReflectionModuleConfig = Field(default_factory=ReflectionModuleConfig)
    cartographer: ReflectionModuleConfig = Field(default_factory=ReflectionModuleConfig)
    cerebral: ReflectionModuleConfig = Field(default_factory=ReflectionModuleConfig)  # MEMORY_CEREBRAL_*
    hindsight: HindsightConfig = Field(default_factory=HindsightConfig)


# Env var suffix (after MEMORY_POSTGRES_) -> PostgresConfig field name.
POSTGRES_ENV_SETTINGS = {
    "HOST": "host",
    "PORT": "port",
    "USER": "user",
    "PASSWORD": "password",
    "DATABASE": "database",
    "SCHEMA": "schema",
}

# Env var suffix (after MEMORY_PG0_) -> Pg0Config field name.
PG0_ENV_SETTINGS = {
    "NAME": "name",
    "PORT": "port",
    "DATA_DIR": "data_dir",
}

HINDSIGHT_ENV_SETTINGS = {
    "EMBEDDINGS_MODEL": "embeddings_model",
}

# Env var name (without the MEMORY_ prefix) -> MemoryConfig field name.
# ``memory`` is a nested model and ``Settings`` does not set
# ``env_nested_delimiter``, so pydantic-settings cannot populate these fields
# from the environment on its own. apply_memory_env_overrides() bridges the gap.
MEMORY_ENV_SETTINGS = {
    "BANK_ID": "bank_id",
    "ENABLED": "enabled",
    "COMPACT_TRAJECTORY": "compact_trajectory",
    "MAX_CONTEXT_MEMORY_TOKENS": "max_context_memory_tokens",
    "MAX_CONTEXT_ITEM_TOKENS": "max_context_item_tokens",
    "MAX_TRAJECTORY_TOKENS": "max_trajectory_tokens",
    "MAX_QUESTION_TOKENS": "max_question_tokens",
}

# The reflection modules, derived from the MemoryConfig fields that hold a
# ReflectionModuleConfig. Iterate this instead of hardcoding module names so
# adding a new reflection module only requires declaring its field above.
REFLECTION_MODULES: tuple[str, ...] = tuple(
    name
    for name, field in MemoryConfig.model_fields.items()
    if field.annotation is ReflectionModuleConfig
)

# Env var suffix -> ReflectionModuleConfig field name, routed via
# MEMORY_<MODULE>_<SETTING> (e.g. MEMORY_DISTILLER_MODEL).
REFLECTION_MODULE_ENV_SETTINGS = {
    name.upper(): name for name in ReflectionModuleConfig.model_fields
}

# Maps env prefix (after MEMORY_) -> (config field name, suffix->field map)
NESTED_ENV_PREFIXES: dict[str, tuple[str, dict[str, str]]] = {
    "POSTGRES_": ("postgres", POSTGRES_ENV_SETTINGS),
    "PG0_": ("pg0", PG0_ENV_SETTINGS),
    "HINDSIGHT_": ("hindsight", HINDSIGHT_ENV_SETTINGS),
}
# Add reflection modules dynamically (same pattern, shared settings map)
for _mod in REFLECTION_MODULES:
    NESTED_ENV_PREFIXES[f"{_mod.upper()}_"] = (_mod, REFLECTION_MODULE_ENV_SETTINGS)


def _generate_bank_id() -> str:
    """Generate a default bank_id."""
    return "codespy"


def apply_memory_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply ``MEMORY_*`` environment variable overrides to the ``memory`` block.

    Maps flat env vars onto the nested ``memory`` config, e.g.::

        MEMORY_POSTGRES_HOST=myhost                    -> memory.postgres.host
        MEMORY_ENABLED=true                            -> memory.enabled
        MEMORY_MAX_CONTEXT_MEMORY_TOKENS=512           -> memory.max_context_memory_tokens

    Nested sub-model overrides use a second level of nesting::

        MEMORY_DISTILLER_MODEL=...        -> memory.distiller.model
        MEMORY_CARTOGRAPHER_TEMPERATURE=0 -> memory.cartographer.temperature

        MEMORY_PG0_NAME=mydb              -> memory.pg0.name
        MEMORY_PG0_PORT=5433              -> memory.pg0.port

    Env vars take precedence over YAML, matching the documented priority
    (Environment Variables > YAML Config > Defaults).


    Note: ``<SIGNATURE>_MEMORY_*`` vars are handled separately by
    ``apply_signature_env_overrides`` and are ignored here, since they never
    match a bare ``MEMORY_`` prefix.

    Args:
        config: The YAML-derived config dict to mutate.

    Returns:
        The same dict, with ``memory`` overrides applied.
    """
    from dotenv import dotenv_values

    from codespy.config_dspy import convert_env_value

    env_vars = {**dotenv_values(".env"), **os.environ}

    for key, value in env_vars.items():
        if value is None:
            continue
        key_upper = key.upper()
        if not key_upper.startswith("MEMORY_"):
            continue
        remainder = key_upper[len("MEMORY_") :]

        memory_config = config.setdefault("memory", {})
        if not isinstance(memory_config, dict):
            continue

        # Nested sub-model: MEMORY_<PREFIX><SETTING>
        # Checked before the flat lookup, since e.g. MEMORY_PG0_NAME has no entry in
        # MEMORY_ENV_SETTINGS and would otherwise be silently dropped.
        nested = next(
            (
                (field, settings_map, remainder[len(prefix):])
                for prefix, (field, settings_map) in NESTED_ENV_PREFIXES.items()
                if remainder.startswith(prefix)
            ),
            None,
        )
        if nested is not None:
            field, settings_map, setting = nested
            setting_field = settings_map.get(setting)
            if setting_field is None:
                continue
            sub_config = memory_config.setdefault(field, {})
            if not isinstance(sub_config, dict):
                continue
            sub_config[setting_field] = convert_env_value(value)
            continue

        field = MEMORY_ENV_SETTINGS.get(remainder)
        if field is None:
            continue
        memory_config[field] = convert_env_value(value)

    return config


# Cached singleton store. Avoids reconstructing the EpisodeStore's connection pool
# on every call.
_store: EpisodeStore | None = None
_store_built = False


def get_episode_store(settings: Settings) -> EpisodeStore | None:
    """Return the cached EpisodeStore for Hippocampus memory, or None if disabled.

    The store is built once and cached (module-level singleton). This matters
    for the connection pool setup.

    Call :func:`reset_episode_store` after changing settings (e.g. via
    ``reload_settings``) to force a rebuild on next access.

    Priority:
    1. If ``memory.postgres.host`` is set, use the built URI to connect.
    2. Else, try to auto-start pg0-embedded for local dev.
    3. If pg0 is not available, return None with a warning.

    Args:
        settings: Application settings.

    Returns:
        Cached EpisodeStore instance, or None if storage is not configured.
    """
    global _store, _store_built
    if _store_built:
        return _store

    mem = settings.memory
    bank_id = mem.bank_id or _generate_bank_id()
    schema = mem.postgres.schema  # "episodic" by default

    # Try external PostgreSQL first
    uri = mem.postgres.build_uri()
    if uri:
        from codespy.agents.memory.postgres import EpisodeStore

        _store = EpisodeStore(uri, bank_id, schema=schema)
        logger.info(f"EpisodeStore connected to external PostgreSQL (bank={bank_id}, schema={schema})")
    else:
        # Try pg0-embedded for local dev
        try:
            from codespy.agents.memory.pg0_manager import get_pg0_uri

            uri = get_pg0_uri(name=mem.pg0.name, port=mem.pg0.port, data_dir=mem.pg0.data_dir)
            from codespy.agents.memory.postgres import EpisodeStore

            _store = EpisodeStore(uri, bank_id, schema=schema)
            logger.info(f"EpisodeStore connected to pg0-embedded PostgreSQL (bank={bank_id}, schema={schema})")
        except ImportError:
            logger.warning(
                "Memory is enabled but no PostgreSQL is configured and pg0-embedded "
                "is not installed. Install with: pip install pg0-embedded\n"
                "Or set MEMORY_POSTGRES_HOST (+ credentials) to use an external PostgreSQL instance."
            )
            _store = None
        except Exception as e:
            logger.warning(f"Failed to start pg0-embedded: {e}")
            _store = None

    _store_built = True
    return _store


def reset_episode_store() -> None:
    """Clear the cached memory store so it is rebuilt on next access.

    Call this after reloading settings (e.g. ``reload_settings()``) so a
    changed ``memory`` configuration takes effect.
    """
    global _store, _store_built
    if _store is not None:
        try:
            _store.close()
        except Exception:
            pass
    _store = None
    _store_built = False


def verify_memory_access(settings: Settings) -> tuple[bool, str]:
    """Verify memory storage is accessible when memory is active.

    Returns:
        Tuple of (success, message). Success is True when memory is disabled
        (no active signatures use it) or when the storage backend responds.
    """
    from codespy.config_dspy import SIGNATURE_NAMES

    # Skip if no enabled signature uses memory
    if not any(
        settings.is_signature_enabled(sig) and settings.get_memory_enabled(sig)
        for sig in SIGNATURE_NAMES
    ):
        return True, "Memory disabled — skipping storage check"

    store = get_episode_store(settings)
    if store is None:
        return (
            False,
            "Memory is enabled but storage is not configured (set MEMORY_POSTGRES_HOST or install pg0-embedded)",
        )

    try:
        store.verify_access()
    except Exception as e:
        return False, f"Memory storage not accessible: {e}"

    return True, f"Memory storage verified (PostgreSQL, bank={settings.memory.bank_id or _generate_bank_id()})"


# Cached singleton Cerebral instance.
_cerebral: "Cerebral" | None = None
_cerebral_built = False


def _derive_cerebral_llm_params(settings: "Settings") -> tuple[str, str | None, str | None, str | None]:
    """Derive MemoryEngine LLM params from the cerebral model config + LLM credentials.

    Parses the litellm model string to extract the provider prefix and maps
    credentials from ``settings.llm``.

    Returns:
        ``(provider, model, api_key, base_url)``
    """
    llm_config = settings.get_llm_config("cerebral")
    model = llm_config.model  # e.g. "bedrock/converse/moonshotai.kimi-k2.5"

    # Parse litellm model string: "provider/model_path"
    parts = model.split("/", 1)
    provider = parts[0] if len(parts) > 1 else "openai"
    model_name = parts[1] if len(parts) > 1 else model

    # Map credentials from Settings.llm
    api_key: str | None = None
    base_url: str | None = None
    llm = settings.llm

    if provider == "bedrock":
        pass  # Uses AWS env vars (AWS_ACCESS_KEY_ID, etc.)
    elif provider == "openai":
        api_key = llm.openai_api_key.get_secret_value() if llm.openai_api_key else None
        base_url = llm.openai_api_base
    elif provider == "anthropic":
        api_key = llm.anthropic_api_key.get_secret_value() if llm.anthropic_api_key else None
    elif provider == "gemini":
        api_key = llm.gemini_api_key.get_secret_value() if llm.gemini_api_key else None
    elif provider in ("azure", "azure_ai"):
        api_key = llm.azure_api_key.get_secret_value() if llm.azure_api_key else None
        base_url = llm.azure_api_base
    else:
        # Unknown provider — pass model as-is, let MemoryEngine/litellm resolve
        provider = "litellm"
        model_name = model

    return provider, model_name, api_key, base_url


def get_cerebral(settings: "Settings") -> "Cerebral" | None:
    """Return the cached Cerebral instance, or None if unavailable.

    Cerebral activates unconditionally (like ``get_episode_store``).
    Agent-level ``get_memory_enabled(sig)`` handles per-signature gating.
    LLM parameters are auto-derived from the ``cerebral``
    ReflectionModuleConfig model string and ``settings.llm`` credentials.

    The store is built once and cached (module-level singleton).

    Call :func:`reset_cerebral` after changing settings (e.g. via
    ``reload_settings``) to force a rebuild on next access.

    Args:
        settings: Application settings.

    Returns:
        Cached Cerebral instance, or None when hindsight-api-slim is
        not installed or PostgreSQL is not available.
    """
    global _cerebral, _cerebral_built
    if _cerebral_built:
        return _cerebral

    try:
        from codespy.agents.memory.cerebral import Cerebral
    except ImportError:
        logger.warning(
            "Memory is enabled but hindsight-api-slim is not installed. "
            "Semantic memory disabled. Install with: pip install hindsight-api-slim"
        )
        _cerebral = None
        _cerebral_built = True
        return None

    pg_uri = settings.memory.postgres.build_uri()
    if not pg_uri:
        try:
            from codespy.agents.memory.pg0_manager import get_pg0_uri

            pg_uri = get_pg0_uri(
                name=settings.memory.pg0.name,
                port=settings.memory.pg0.port,
                data_dir=settings.memory.pg0.data_dir,
            )
        except Exception:
            logger.warning("Memory enabled but no PostgreSQL available for Cerebral")
            _cerebral = None
            _cerebral_built = True
            return None

    provider, model_name, api_key, base_url = _derive_cerebral_llm_params(settings)
    bank_id = settings.memory.bank_id or "codespy"
    embeddings_model = (
        settings.memory.hindsight.embeddings_model
        or EMBEDDING_MODELS.get(provider, "openai/text-embedding-3-small")
    )

    try:
        _cerebral = Cerebral(
            database_url=pg_uri,
            llm_provider=provider,
            llm_model=model_name,
            llm_api_key=api_key,
            llm_base_url=base_url,
            bank_id=bank_id,
            embeddings_model=embeddings_model,
        )
    except Exception:
        logger.error(
            "Cerebral initialization FAILED (bank=%s, schema=semantic). "
            "Semantic memory is disabled for this run.",
            bank_id, exc_info=True,
        )
        _cerebral = None
        _cerebral_built = True
        return None

    llm_config = settings.get_llm_config("cerebral")
    logger.info(
        "Cerebral initialized (bank=%s, model=%s, provider=%s, schema=semantic)",
        bank_id, llm_config.model, provider,
    )
    _cerebral_built = True
    return _cerebral


def reset_cerebral() -> None:
    """Clear the cached Cerebral instance so it is rebuilt on next access.

    Call this after reloading settings (e.g. ``reload_settings()``) so a
    changed ``memory`` configuration takes effect.
    """
    global _cerebral, _cerebral_built
    if _cerebral is not None:
        try:
            _cerebral.close()
        except Exception:
            pass
    _cerebral = None
    _cerebral_built = False
