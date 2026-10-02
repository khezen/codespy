"""Memory (Hippocampus, Cerebral, Prefrontal) configuration and storage factory."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from codespy.config_dspy import ReasoningEffort

# Hindsight constraint: validate_retain_completion_token_budget requires
# retain_max_completion_tokens (default 64000) > retain_chunk_size.
# If chunk_size >= 64000, update_bank_config rejects the update and the bank
# silently falls back to defaults (chunk 3000, no mission).
_HINDSIGHT_RETAIN_MAX_COMPLETION_TOKENS = 64000

# Hindsight reads its reflect-loop iteration cap from this env var once, when
# its config is first built (at import). See get_cerebral().
HINDSIGHT_REFLECT_MAX_ITERATIONS_ENV = "HINDSIGHT_API_REFLECT_MAX_ITERATIONS"

# Hindsight reads its reflect context token cap from this env var once, when
# its config is first built. Also bounds briefing refreshes (LOW budget).
HINDSIGHT_REFLECT_MAX_CONTEXT_TOKENS_ENV = "HINDSIGHT_API_REFLECT_MAX_CONTEXT_TOKENS"

if TYPE_CHECKING:
    from codespy.agents.memory.cerebral import Cerebral
    from codespy.agents.memory.postgres import EpisodeStore
    from codespy.agents.memory.prefrontal import Prefrontal
    from codespy.config import Settings

logger = logging.getLogger(__name__)


# Default embedding model per LLM provider prefix. Used by get_cerebral()
# when embeddings.model is None. litellm-sdk routes through litellm.
EMBEDDING_MODELS: dict[str, str] = {
    "bedrock": "bedrock/cohere.embed-v4:0",
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

    model: str | None = "bedrock/converse/nvidia.nemotron-super-3-120b"
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
    distiller: ReflectionModuleConfig = Field(default_factory=lambda: ReflectionModuleConfig(model="bedrock/converse/nvidia.nemotron-super-3-120b"))
    cartographer: ReflectionModuleConfig = Field(default_factory=lambda: ReflectionModuleConfig(model="bedrock/converse/nvidia.nemotron-super-3-120b"))


class CerebralRetainConfig(BaseModel):
    """Cerebral (semantic memory) LLM configuration for fact extraction."""

    model: str | None = "bedrock/converse/nvidia.nemotron-super-3-120b"  # Falls back to llm.default_model
    # Value is in chars. The merged observations blob is bounded by
    # max_hippocampus_tokens (~65K chars worst case), so large blobs split
    # into several chunks. Must be < _HINDSIGHT_RETAIN_MAX_COMPLETION_TOKENS.
    chunk_size: int = Field(default=12288, gt=0, lt=_HINDSIGHT_RETAIN_MAX_COMPLETION_TOKENS)


class CerebralConsolidationConfig(BaseModel):
    """Cerebral (semantic memory) LLM configuration for consolidation.

    Consolidation merges facts after episode retention. Falls back to
    cerebral.retain.model → llm.default_model when unset.
    """

    model: str | None = "bedrock/converse/nvidia.nemotron-super-3-120b"  # Falls back to cerebral.retain.model → llm.default_model
    # Hindsight observation_scope_limits cap. -1 = unlimited, 0 = no new observations.
    # Applied per scope as: scope=[org:*, repo:*, project_scope:*], limit=N
    max_observations_per_scope: int = Field(default=100, ge=-1)


class MentalModelsConfig(BaseModel):
    """Mental-model (briefing) refresh LLM configuration.

    Mental-model refresh synthesizes briefings after consolidation.
    Falls back to prefrontal.recall.model → cerebral.retain.model → llm.default_model
    when unset. Only used when reflects > 0.

    Delta mode means nothing is lost - the refresh covers all new facts since
    the last successful refresh when the interval expires.
    """

    model: str | None = "bedrock/converse/nvidia.nemotron-super-3-120b"  # Falls back to prefrontal.recall.model → retain → default
    max_tokens: int = Field(default=2048, gt=0)  # Briefing size
    min_refresh_seconds: int = Field(default=0, ge=0)  # Minimum seconds between refreshes


class CerebralEmbeddingsConfig(BaseModel):
    """Cerebral (semantic memory) embeddings configuration."""

    model: str | None = "bedrock/cohere.embed-v4:0"  # Auto-derived from provider if unset
    max_input_chars: int | None = Field(
        default=None,
        ge=0,
        description="Maximum characters per embedding input. null=auto (2048 for Bedrock Cohere v3), 0=off, N=cap",
    )


class CerebralConfig(BaseModel):
    """Cerebral (semantic memory) configuration."""

    retain: CerebralRetainConfig = Field(default_factory=CerebralRetainConfig)
    consolidation: CerebralConsolidationConfig = Field(default_factory=CerebralConsolidationConfig)
    mental_models: MentalModelsConfig = Field(default_factory=MentalModelsConfig)
    embeddings: CerebralEmbeddingsConfig = Field(default_factory=CerebralEmbeddingsConfig)


class RecallConfig(BaseModel):
    """Prefrontal recall (read-time context injection) configuration.

    Controls the read-time recall into agent context. Only used when
    memory.prefrontal.reflects > 0.
    """

    # Model for Hindsight reflect (pre-call context and recall_memory tool).
    # Falls back to cerebral.retain.model → llm.default_model.
    # Unused when reflects=0 (no LLM at read time).
    model: str | None = "bedrock/converse/nvidia.nemotron-super-3-120b"
    # local: this scope/repo only. org: also other repos of the same owner.
    # bank: every repo in the bank (crosses organisations — opt-in only).
    reach: Literal["local", "org", "bank"] = "org"
    # Token budget for the pre-call context (local facets + remote facts).
    max_tokens: int = Field(default=8192, gt=0)
    # Token budget for recall_memory tool results.
    max_tool_tokens: int = Field(default=2048, gt=0)
    # recall_memory tool call limit (0 = tool disabled).
    max_tool_calls: int = Field(default=0, ge=0)


class PrefrontalConfig(BaseModel):
    """Prefrontal (semantic recall into agent context) configuration."""

    # 0: raw facts only, no LLM at read time and no briefings (no mental models).
    # N > 0: Hindsight reflect loop, capped at N iterations (both Prefrontal and
    # briefing refreshes). Uses Hindsight's LOW budget (0.5× multiplier), so the
    # global cap is set to 2× reflects to achieve exactly reflects iterations.
    reflects: int = Field(default=3, ge=0)
    # Recall settings (read-time context injection).
    recall: RecallConfig = Field(default_factory=RecallConfig)


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

    # Prefrontal (semantic recall into agent context) configuration
    prefrontal: PrefrontalConfig = Field(default_factory=PrefrontalConfig)


# Memory unit name prefix for LLM work units
MEMORY_UNIT_PREFIX = "memory_"

# Memory unit names (used for cost buckets, config lookup, and logging)
MEMORY_DISTILLER = "memory_distiller"
MEMORY_CARTOGRAPHER = "memory_cartographer"
MEMORY_RETAIN = "memory_retain"
MEMORY_EMBEDDINGS = "memory_embeddings"
MEMORY_OTHER = "memory_other"
MEMORY_PREFRONTAL = "memory_prefrontal"
MEMORY_CONSOLIDATION = "memory_consolidation"
MEMORY_MENTAL_MODELS = "memory_mental_models"

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


def _litellm_credentials(settings: "Settings", model: str) -> tuple[str | None, str | None]:
    """Derive API key and base_url from a litellm model string.

    All Cerebral LLM calls go through litellm. This function parses the litellm
    model string and extracts credentials based on the provider prefix.

    Args:
        settings: Application settings containing LLM credentials.
        model: Full litellm model string (e.g., "bedrock/converse/moonshotai.kimi-k2.5").

    Returns:
        ``(api_key, base_url)`` - None values indicate no special handling.
    """
    # Parse litellm model string: "provider/model_path"
    parts = model.split("/", 1)
    provider = parts[0] if len(parts) > 1 else "openai"

    llm = settings.llm
    api_key: str | None = None
    base_url: str | None = None

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

    return api_key, base_url


def _cerebral_litellm_params(settings: "Settings") -> tuple[str, str | None, str | None]:
    """Derive MemoryEngine litellm params from the cerebral model config + LLM credentials.

    Convenience wrapper around _litellm_credentials for the retain model.

    Returns:
        ``(model, api_key, base_url)`` - model is the full litellm string unchanged.
    """
    llm_config = settings.get_llm_config(MEMORY_RETAIN)
    model = llm_config.model
    api_key, base_url = _litellm_credentials(settings, model)
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

    # Prefrontal recall model (optional)
    recall_model = settings.memory.prefrontal.recall.model
    reflect_kwargs: dict[str, str | None] = {}
    if recall_model:
        recall_api_key, recall_base_url = _litellm_credentials(settings, recall_model)
        reflect_kwargs = {
            "reflect_llm_model": recall_model,
            "reflect_llm_api_key": recall_api_key,
            "reflect_llm_base_url": recall_base_url,
        }

    # Consolidation model (optional) - pass to Cerebral only when explicitly set
    # and no operator env override is present (env wins over config)
    cons_model = settings.memory.cerebral.consolidation.model
    consolidation_kwargs: dict[str, str | None] = {}
    if cons_model and not os.environ.get("HINDSIGHT_API_CONSOLIDATION_LLM_MODEL"):
        cons_api_key, cons_base_url = _litellm_credentials(settings, cons_model)
        consolidation_kwargs = {
            "consolidation_llm_model": cons_model,
            "consolidation_llm_api_key": cons_api_key,
            "consolidation_llm_base_url": cons_base_url,
        }

    # Apply Hindsight LLM defaults BEFORE Cerebral construction (these are read
    # during MemoryEngine.__init__). reflect_config is applied after construction.
    _apply_hindsight_llm_call_defaults(settings)

    # Apply mental-model refresh LLM config to Hindsight raw config BEFORE Cerebral construction
    _apply_mental_model_refresh_llm(settings)

    # Extraction model for two-step structured output fallback
    extraction_model = settings.get_llm_config(MEMORY_RETAIN).extraction_model
    extraction_kwargs: dict[str, str | None] = {}
    if extraction_model:
        extraction_api_key, extraction_base_url = _litellm_credentials(settings, extraction_model)
        extraction_kwargs = {
            "extraction_llm_model": extraction_model,
            "extraction_llm_api_key": extraction_api_key,
            "extraction_llm_base_url": extraction_base_url,
        }

    try:
        _cerebral = Cerebral(
            database_url=pg_uri,
            llm_provider="litellm",
            llm_model=model,
            llm_api_key=api_key,
            llm_base_url=base_url,
            bank_id=bank_id,
            embeddings_model=embeddings_model,
            retain_chunk_size=settings.memory.cerebral.retain.chunk_size,
            mental_models=settings.memory.prefrontal.reflects > 0,
            max_mental_model_tokens=settings.memory.cerebral.mental_models.max_tokens,
            embeddings_max_input_chars=settings.memory.cerebral.embeddings.max_input_chars,
            min_mental_model_refresh_seconds=settings.memory.cerebral.mental_models.min_refresh_seconds,
            max_observations_per_scope=settings.memory.cerebral.consolidation.max_observations_per_scope,
            **reflect_kwargs,
            **consolidation_kwargs,
            **extraction_kwargs,
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

    reflect_model_str = f", reflect_model={recall_model}" if recall_model else ""
    consolidation_model_str = (
        f", consolidation_model={consolidation_kwargs.get('consolidation_llm_model')}"
        if consolidation_kwargs.get("consolidation_llm_model")
        else ""
    )
    mental_models_model_str = (
        f", mental_models_model={settings.memory.cerebral.mental_models.model}"
        if settings.memory.cerebral.mental_models.model and settings.memory.prefrontal.reflects > 0
        else ""
    )
    logger.info(
        "Cerebral initialized (bank=%s, provider=litellm, model=%s%s%s%s, schema=semantic)",
        bank_id,
        model,
        reflect_model_str,
        consolidation_model_str,
        mental_models_model_str,
    )
    # Apply reflect caps programmatically (after Cerebral construction, before first use)
    _apply_reflect_config(settings.memory.prefrontal)
    _cerebral_built = True
    return _cerebral


# Inverse of Hindsight's LOW budget multiplier (0.5) to double the raw
# iteration cap so both Prefrontal and briefing refresh get exactly
# ``reflects`` iterations: max(1, int(2*reflects * 0.5)) = reflects.
# See memory_engine.py:15134-15138 for the multiplier logic.
_LOW_BUDGET_MULTIPLIER_INVERSE = 2


def _apply_hindsight_llm_call_defaults(settings: "Settings") -> None:
    """Apply Hindsight LLM timeout and retry settings from codespy config.

    Sets global ``llm_timeout`` and ``llm_max_retries`` so all Hindsight operations
    (retain, consolidation, reflect, mental-model refresh) inherit codespy's
    ``llm.timeout`` / ``llm.retries``. Per-operation env vars still win automatically
    because Hindsight prefers non-None per-op values.

    Also sets ``reflect_llm_timeout`` / ``reflect_llm_max_retries`` to override
    Hindsight's 30s reflect default when no operator env is set.

    Operator env (HINDSIGHT_API_LLM_TIMEOUT, HINDSIGHT_API_LLM_MAX_RETRIES,
    HINDSIGHT_API_REFLECT_LLM_TIMEOUT, HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES) wins.

    Called in get_cerebral() after Cerebral import but before Cerebral() construction.
    This must run before Cerebral is built because the defaults are resolved in
    MemoryEngine.__init__.

    Args:
        settings: Application settings.
    """
    try:
        import hindsight_api.config as ha_cfg

        raw = ha_cfg._get_raw_config()
    except Exception as e:
        logger.warning("Could not access Hindsight raw config for LLM defaults: %s", e)
        return

    # --- Global settings (retain, consolidation, reflect, mental-model refresh) ---

    # Determine global timeout: env wins, then settings.llm.timeout
    global_timeout: float | None = None
    global_env_timeout = os.environ.get("HINDSIGHT_API_LLM_TIMEOUT")
    if global_env_timeout is not None:
        try:
            global_timeout = float(global_env_timeout)
            logger.info("Using env HINDSIGHT_API_LLM_TIMEOUT=%s (operator override)", global_timeout)
        except ValueError:
            pass
    if global_timeout is None and hasattr(raw, "llm_timeout"):
        global_timeout = float(settings.llm.timeout)
        raw.llm_timeout = global_timeout

    # Determine global retries: env wins, then settings.llm.retries
    global_retries: int | None = None
    global_env_retries = os.environ.get("HINDSIGHT_API_LLM_MAX_RETRIES")
    if global_env_retries is not None:
        try:
            global_retries = int(global_env_retries)
            logger.info("Using env HINDSIGHT_API_LLM_MAX_RETRIES=%s (operator override)", global_retries)
        except ValueError:
            pass
    if global_retries is None and hasattr(raw, "llm_max_retries"):
        global_retries = int(settings.llm.retries)
        raw.llm_max_retries = global_retries

    # --- Reflect-specific settings (override Hindsight's 30s default) ---

    # Determine reflect timeout: env wins, then settings.llm.timeout
    reflect_timeout: float | None = None
    reflect_env_timeout = os.environ.get("HINDSIGHT_API_REFLECT_LLM_TIMEOUT") or os.environ.get("HINDSIGHT_API_LLM_TIMEOUT")
    if reflect_env_timeout is not None:
        try:
            reflect_timeout = float(reflect_env_timeout)
            # Log the actual env var that was used
            used_env = "HINDSIGHT_API_REFLECT_LLM_TIMEOUT" if os.environ.get("HINDSIGHT_API_REFLECT_LLM_TIMEOUT") else "HINDSIGHT_API_LLM_TIMEOUT"
            logger.info("Using env %s=%s (operator override)", used_env, reflect_timeout)
        except ValueError:
            pass
    if reflect_timeout is None and hasattr(raw, "reflect_llm_timeout"):
        reflect_timeout = float(settings.llm.timeout)
        raw.reflect_llm_timeout = reflect_timeout

    # Determine reflect retries: env wins, then settings.llm.retries
    reflect_retries: int | None = None
    reflect_env_retries = os.environ.get("HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES") or os.environ.get("HINDSIGHT_API_LLM_MAX_RETRIES")
    if reflect_env_retries is not None:
        try:
            reflect_retries = int(reflect_env_retries)
            # Log the actual env var that was used
            used_env = "HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES" if os.environ.get("HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES") else "HINDSIGHT_API_LLM_MAX_RETRIES"
            logger.info("Using env %s=%s (operator override)", used_env, reflect_retries)
        except ValueError:
            pass
    if reflect_retries is None and hasattr(raw, "reflect_llm_max_retries"):
        reflect_retries = int(settings.llm.retries)
        raw.reflect_llm_max_retries = reflect_retries

    # Summary log
    global_timeout_str = str(global_timeout) if global_timeout is not None else "-"
    global_retries_str = str(global_retries) if global_retries is not None else "-"
    reflect_timeout_str = str(reflect_timeout) if reflect_timeout is not None else "-"
    reflect_retries_str = str(reflect_retries) if reflect_retries is not None else "-"

    if global_timeout is not None or global_retries is not None:
        logger.info(
            "Hindsight LLM defaults: timeout=%s, max_retries=%s (reflect timeout=%s, max_retries=%s)",
            global_timeout_str,
            global_retries_str,
            reflect_timeout_str,
            reflect_retries_str,
        )


def _apply_reflect_config(cfg: PrefrontalConfig) -> dict[str, Any]:
    """Apply reflect caps to Hindsight config programmatically.

    Called in get_cerebral() after Cerebral import. Hindsight re-reads
    reflect_max_iterations at call time (not in _CONFIGURABLE_FIELDS), so
    setting it on _get_raw_config() after import takes effect.

    The global cap is set to 2 × reflects (when reflects > 0) so that with
    Hindsight's LOW budget multiplier (0.5×), both Prefrontal and briefing
    refresh get exactly ``reflects`` iterations.

    Args:
        cfg: PrefrontalConfig with reflects.

    Returns:
        Dict of effective values applied (iterations). Empty dict when reflects == 0.
    """
    # Set a minimal cap of 2 even when reflects == 0 to bound any Hindsight reflect
    raw_cap = max(2 * max(cfg.reflects, 1), 1)

    try:
        import hindsight_api.config as ha_cfg

        raw = ha_cfg._get_raw_config()
    except Exception as e:
        logger.warning("Could not access Hindsight raw config: %s", e)
        return {}

    applied: dict[str, Any] = {}

    # Apply iteration cap: env wins as-is (operator raw cap), else double reflects
    env_iters = os.environ.get(HINDSIGHT_REFLECT_MAX_ITERATIONS_ENV)
    if env_iters is not None:
        try:
            applied["iterations"] = int(env_iters)
            applied["budget"] = "low"
            logger.info("Using env %s=%s (operator override)", HINDSIGHT_REFLECT_MAX_ITERATIONS_ENV, env_iters)
        except ValueError:
            pass
    elif cfg.reflects > 0:
        if hasattr(raw, "reflect_max_iterations"):
            raw.reflect_max_iterations = raw_cap
            applied["iterations"] = raw_cap
            applied["budget"] = "low"

    # Hindsight's reflect_max_context_tokens default is 100000; we no longer
    # override it. The default is used which allows full context without
    # triggering split synthesis.

    if applied:
        effective_iters = applied.get("iterations", 0)
        # With LOW budget (0.5×), iterations = max(1, int(cap * 0.5))
        effective_reflects = max(1, int(effective_iters * 0.5))
        logger.info(
            "Prefrontal reflect caps: iterations=%d (hindsight cap=%d, budget=low) context_tokens=100000",
            effective_reflects,
            effective_iters,
        )

    return applied


def get_prefrontal(
    settings: Settings,
    task_name: str,
    repo_full_name: str | None,
    scope_topic_ids: Sequence[str] | None = None,
    include_repo: bool = False,
) -> Prefrontal | None:
    """Return a Prefrontal for one agent call, or None when inactive.

    Active only when ``get_memory_enabled(task_name)`` is true, a repo is
    known and ``get_cerebral`` returns a Cerebral. Otherwise the agent
    behaves exactly as without Prefrontal. Never raises.
    """
    try:
        if not settings.get_memory_enabled(task_name) or not repo_full_name:
            return None
        cerebral = get_cerebral(settings)
        if cerebral is None:
            return None
        from codespy.agents.memory.prefrontal import Prefrontal

        return Prefrontal(
            settings.memory.prefrontal,
            cerebral,
            task_name,
            repo_full_name,
            scope_topic_ids=scope_topic_ids,
            include_repo=include_repo,
        )
    except Exception:
        logger.warning("Prefrontal unavailable for %s", task_name, exc_info=True)
        return None


# Consumer signatures that share the run-level Prefrontal load
_RUN_PREFRONTAL_CONSUMERS = ("summary", "code_review", "doc", "supply_chain", "audit")


def get_run_prefrontal(
    settings: Settings,
    repo_full_name: str,
    scope_topic_ids: Sequence[str],
) -> Prefrontal | None:
    """Return a Prefrontal for the shared run-level recall, or None when inactive.

    Active only when:
    - A repo is known
    - Cerebral is available
    - At least one consumer (summary, code_review, doc, supply_chain, audit)
      has both is_signature_enabled and get_memory_enabled True

    When active, returns a Prefrontal with task_name="review", include_repo=True,
    and all scope_topic_ids. The run-level recall is shared by all consumer agents.

    Never raises (same soft-fail behavior as get_prefrontal).
    """
    try:
        if not repo_full_name:
            return None

        # Check if any consumer has memory enabled
        from codespy.config_dspy import SIGNATURE_NAMES

        has_memory_consumer = any(
            settings.is_signature_enabled(sig) and settings.get_memory_enabled(sig)
            for sig in SIGNATURE_NAMES
            if sig in _RUN_PREFRONTAL_CONSUMERS
        )
        if not has_memory_consumer:
            return None

        cerebral = get_cerebral(settings)
        if cerebral is None:
            return None

        from codespy.agents.memory.prefrontal import Prefrontal

        return Prefrontal(
            settings.memory.prefrontal,
            cerebral,
            task_name="review",
            repo_full_name=repo_full_name,
            scope_topic_ids=scope_topic_ids,
            include_repo=True,
        )
    except Exception:
        logger.warning("Run Prefrontal unavailable for %s", repo_full_name, exc_info=True)
        return None


# Snapshot of raw config mental_model_refresh fields before mutation,
# restored by reset_cerebral() to avoid leaking settings across rebuilds.
_mental_model_refresh_snapshot: dict[str, Any] | None = None


def _apply_mental_model_refresh_llm(settings: "Settings") -> None:
    """Apply mental-model refresh LLM config to Hindsight raw config programmatically.

    Hindsight reads mental_model_refresh_llm_* from raw config (populated from
    HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_* env vars). This function sets those
    fields when codespy's config has an explicit mental_models.model and no env
    override is present.

    The snapshot is stored in _mental_model_refresh_snapshot so reset_cerebral()
    can restore the original values.

    Args:
        settings: Application settings.
    """
    global _mental_model_refresh_snapshot

    # Only apply when reflects > 0 and a mental_models model is configured
    # reflects gates mental models; the model itself lives in cerebral.mental_models
    if settings.memory.prefrontal.reflects == 0:
        return
    mm_cfg = settings.memory.cerebral.mental_models
    if not mm_cfg.model:
        return

    # Operator env wins - if any HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_* is set,
    # do not touch the raw config (the env values are already there)
    env_vars = [
        "HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_PROVIDER",
        "HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_MODEL",
        "HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_API_KEY",
        "HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_BASE_URL",
    ]
    if any(os.environ.get(v) for v in env_vars):
        logger.info("Using HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_* env vars (operator override)")
        return

    try:
        import hindsight_api.config as ha_cfg

        raw = ha_cfg._get_raw_config()
    except Exception as e:
        logger.warning("Could not access Hindsight raw config for mental-model refresh: %s", e)
        return

    # Snapshot current values before mutation
    _mental_model_refresh_snapshot = {}
    for field in ("mental_model_refresh_llm_provider", "mental_model_refresh_llm_model",
                  "mental_model_refresh_llm_api_key", "mental_model_refresh_llm_base_url"):
        if hasattr(raw, field):
            _mental_model_refresh_snapshot[field] = getattr(raw, field)

    # Get credentials for the configured model
    model = mm_cfg.model
    api_key, base_url = _litellm_credentials(settings, model)

    try:
        # Set the raw config fields (guarded by hasattr for forward compatibility)
        if hasattr(raw, "mental_model_refresh_llm_provider"):
            raw.mental_model_refresh_llm_provider = "litellm"
        if hasattr(raw, "mental_model_refresh_llm_model"):
            raw.mental_model_refresh_llm_model = model
        if hasattr(raw, "mental_model_refresh_llm_api_key"):
            raw.mental_model_refresh_llm_api_key = api_key
        if hasattr(raw, "mental_model_refresh_llm_base_url"):
            # Use "" for no base URL (not None), so Hindsight doesn't inherit reflect's base URL
            raw.mental_model_refresh_llm_base_url = base_url or ""

        logger.info("Mental-model refresh LLM: model=%s", model)
    except Exception as e:
        logger.warning("Failed to set mental-model refresh LLM config: %s", e)


def reset_cerebral() -> None:
    """Clear the cached Cerebral instance so it is rebuilt on next access.

    Call this after reloading settings (e.g. ``reload_settings()``) so a
    changed ``memory`` configuration takes effect.

    Also restores the Hindsight raw config mental_model_refresh fields from
    the snapshot taken by _apply_mental_model_refresh_llm to avoid leaking
    settings across rebuilds.
    """
    global _cerebral, _cerebral_built, _mental_model_refresh_snapshot

    if _cerebral is not None:
        try:
            _cerebral.close()
        except Exception:
            pass
    _cerebral = None
    _cerebral_built = False

    # Restore raw config snapshot if present
    if _mental_model_refresh_snapshot:
        try:
            import hindsight_api.config as ha_cfg

            raw = ha_cfg._get_raw_config()
            for field, value in _mental_model_refresh_snapshot.items():
                if hasattr(raw, field):
                    setattr(raw, field, value)
        except Exception:
            pass
        _mental_model_refresh_snapshot = None
