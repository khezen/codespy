"""Memory (Hippocampus) configuration and storage factory."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from codespy.config_dspy import ReasoningEffort

if TYPE_CHECKING:
    from codespy.agents.memory.postgres import EpisodeStore
    from codespy.config import Settings

logger = logging.getLogger(__name__)


# Default embedding model per LLM provider prefix. Used by get_cerebral()
# when embeddings.model is None. litellm-sdk routes through litellm.
EMBEDDING_MODELS: dict[str, str] = {
    "bedrock": "bedrock/cohere.embed-multilingual-v3",
    "openai": "openai/text-embedding-3-small",
    "anthropic": "openai/text-embedding-3-small",
    "gemini": "gemini/text-embedding-004",
    "azure": "azure/text-embedding-3-small",
    "litellm": "openai/text-embedding-3-small",
}


class ReflectionModuleConfig(BaseModel):
    """LLM overrides for a single reflection module (Distiller / Cartographer).

    All fields are optional — ``None`` means "fall back to the corresponding
    top-level ``llm.default_*`` setting" (see ``codespy.config.Settings``).

    The reflection modules are compact summarize/curate tasks rather than deep
    analysis, so they are good candidates for a cheaper model tier than the
    one used for code review.
    """

    model: str | None = None
    reasoning_effort: ReasoningEffort | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_iters: int | None = 1
    max_llm_calls: int | None = 2


class HippocampusConfig(BaseModel):
    """Hippocampus (episodic memory) configuration."""

    # Whether to apply head+tail trajectory bounding before distillation.
    compact_trajectory: bool = True

    # Token budgets
    max_hippocampus_tokens: int = Field(default=16384)
    max_hippocampus_item_tokens: int = Field(default=512)
    max_trajectory_tokens: int | None = Field(default=16384)
    max_question_tokens: int | None = Field(default=8192)

    # Reflection modules
    distiller: ReflectionModuleConfig = Field(default_factory=ReflectionModuleConfig)
    cartographer: ReflectionModuleConfig = Field(default_factory=ReflectionModuleConfig)


class CerebralRetainConfig(BaseModel):
    """Cerebral (semantic memory) LLM configuration for fact extraction."""

    model: str | None = None  # Falls back to llm.default_model


class CerebralEmbeddingsConfig(BaseModel):
    """Cerebral (semantic memory) embeddings configuration."""

    model: str | None = None  # Auto-derived from provider if unset


class CerebralConfig(BaseModel):
    """Cerebral (semantic memory) configuration."""

    retain: CerebralRetainConfig = Field(default_factory=CerebralRetainConfig)
    embeddings: CerebralEmbeddingsConfig = Field(default_factory=CerebralEmbeddingsConfig)


class LLMSettings(BaseModel):
    """Fully resolved LLM settings for one named unit of work.

    Produced by ``Settings.get_llm_config()`` for either a signature
    (``review.<name>``) or a reflection module (``memory.hippocampus.<field>`` for
    ``memory_<field>``): every field is either the name-specific override or the
    corresponding top-level default, so consumers never re-apply fallback logic.
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

    host: str | None = None
    port: int = 5432
    user: str | None = None
    password: str | None = None
    database: str = "codespy"
    schema_name: str | None = Field(
        default="episodic", alias="schema"
    )

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

    name: str = "codespy"
    port: int | None = None
    data_dir: str | None = None


class MemoryConfig(BaseModel):
    """Global memory (Hippocampus + Cerebral) configuration.

    Controls where episodes are persisted and the memory knob applied to
    every agent. Per-signature ``memory:`` blocks override ``enabled``.
    """

    # PostgreSQL connection settings
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    pg0: Pg0Config = Field(default_factory=Pg0Config)
    bank_id: str | None = None

    # Master switch — overridable per-signature via review.<name>.memory.enabled
    enabled: bool = False

    # Hippocampus (episodic memory) configuration
    hippocampus: HippocampusConfig = Field(default_factory=HippocampusConfig)

    # Cerebral (semantic memory) configuration
    cerebral: CerebralConfig = Field(default_factory=CerebralConfig)


# Memory unit name prefix for LLM work units
MEMORY_UNIT_PREFIX = "memory_"

# Memory unit names (used for cost buckets, config lookup, and logging)
MEMORY_DISTILLER = "memory_distiller"
MEMORY_CARTOGRAPHER = "memory_cartographer"
MEMORY_RETAIN = "memory_retain"
MEMORY_EMBEDDINGS = "memory_embeddings"
MEMORY_OTHER = "memory_other"

# The reflection modules, derived from the HippocampusConfig fields that hold a
# ReflectionModuleConfig. Iterate this instead of hardcoding module names so
# adding a new reflection module only requires declaring its field above.
# Names are prefixed with MEMORY_UNIT_PREFIX to match the config/env naming.
REFLECTION_MODULES: tuple[str, ...] = tuple(
    f"{MEMORY_UNIT_PREFIX}{name}"
    for name, field in HippocampusConfig.model_fields.items()
    if field.annotation is ReflectionModuleConfig
)


def reflection_module_config(hippocampus: HippocampusConfig, name: str) -> ReflectionModuleConfig:
    """Get the ReflectionModuleConfig for a prefixed module name.

    Removes the MEMORY_UNIT_PREFIX and returns the corresponding config
    from the HippocampusConfig.

    Args:
        hippocampus: The HippocampusConfig instance.
        name: The prefixed module name (e.g., "memory_distiller").

    Returns:
        The ReflectionModuleConfig for that module.

    Raises:
        ValueError: If the name doesn't start with the expected prefix.
        AttributeError: If the module doesn't exist on HippocampusConfig.
    """
    if not name.startswith(MEMORY_UNIT_PREFIX):
        raise ValueError(f"Expected name to start with '{MEMORY_UNIT_PREFIX}', got: {name}")
    field_name = name[len(MEMORY_UNIT_PREFIX):]
    return getattr(hippocampus, field_name)


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
    bank_id = mem.bank_id or "codespy"
    schema = mem.postgres.schema_name  # "episodic" by default

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

    return True, f"Memory storage verified (PostgreSQL, bank={settings.memory.bank_id or 'codespy'})"


# Cached singleton Cerebral instance.
_cerebral: "Cerebral" | None = None
_cerebral_built = False


def _cerebral_litellm_params(settings: "Settings") -> tuple[str, str | None, str | None]:
    """Derive MemoryEngine litellm params from the cerebral model config + LLM credentials.

    All Cerebral LLM calls go through litellm. This function parses the litellm
    model string and extracts credentials based on the provider prefix.

    Returns:
        ``(model, api_key, base_url)`` - model is the full litellm string unchanged
    """
    llm_config = settings.get_llm_config(MEMORY_RETAIN)
    model = llm_config.model  # e.g. "bedrock/converse/moonshotai.kimi-k2.5"

    # Parse litellm model string: "provider/model_path"
    parts = model.split("/", 1)
    provider = parts[0] if len(parts) > 1 else "openai"

    # Map credentials from Settings.llm
    api_key: str | None = None
    base_url: str | None = None
    llm = settings.llm

    if provider == "bedrock":
        pass  # Uses AWS env vars (AWS_ACCESS_KEY_ID, etc.) - litellm reads these
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
        # Azure API version comes from env var AZURE_API_VERSION, same as DSPy
    # For any other provider, let litellm resolve from env/globals

    return model, api_key, base_url


def get_cerebral(settings: "Settings") -> "Cerebral" | None:
    """Return the cached Cerebral instance, or None if unavailable.

    Cerebral activates unconditionally (like ``get_episode_store``).
    Agent-level ``get_memory_enabled(sig)`` handles per-signature gating.
    LLM parameters are auto-derived from the ``cerebral.retain``
    config model string and ``settings.llm`` credentials.

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

    model, api_key, base_url = _cerebral_litellm_params(settings)
    bank_id = settings.memory.bank_id or "codespy"
    # Choose embedding default based on the provider prefix
    provider_prefix = model.split("/", 1)[0] if "/" in model else "openai"
    embeddings_model = (
        settings.memory.cerebral.embeddings.model
        or EMBEDDING_MODELS.get(provider_prefix, "openai/text-embedding-3-small")
    )

    try:
        _cerebral = Cerebral(
            database_url=pg_uri,
            llm_provider="litellm",
            llm_model=model,
            llm_api_key=api_key,
            llm_base_url=base_url,
            bank_id=bank_id,
            embeddings_model=embeddings_model,
        )
    except Exception:
        logger.error(
            "Cerebral initialization FAILED (bank=%s, schema=semantic). "
            "Semantic memory is disabled for this run.",
            bank_id,
            exc_info=True,
        )
        _cerebral = None
        _cerebral_built = True
        return None

    logger.info(
        "Cerebral initialized (bank=%s, provider=litellm, model=%s, schema=semantic)",
        bank_id,
        model,
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
