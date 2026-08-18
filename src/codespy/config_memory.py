"""Memory (Hippocampus) configuration and storage factory."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from codespy.config_dspy import ReasoningEffort
from codespy.tools.storage.base import Storage

if TYPE_CHECKING:
    from codespy.config import Settings


MemoryBackend = Literal["filesystem", "s3"]


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


class MemoryConfig(BaseModel):
    """Global memory (Hippocampus) configuration.

    Controls where episodes are persisted and the default reflection knobs
    applied to every agent. Per-signature ``memory:`` blocks override the
    ``default_*`` values.
    """

    # Storage backend

    backend: MemoryBackend = "filesystem"  # MEMORY_BACKEND
    root: str = "~/.cache/codespy/memory"  # MEMORY_ROOT (filesystem backend)
    s3_bucket: str | None = None  # MEMORY_S3_BUCKET (s3 backend)
    s3_region: str | None = None  # MEMORY_S3_REGION (falls back to aws_region)
    s3_endpoint_url: str | None = None  # MEMORY_S3_ENDPOINT_URL (MinIO/S3-compatible)

    # Reflection defaults — overridable per-signature
    default_enabled: bool = False  # MEMORY_DEFAULT_ENABLED
    default_max_reflects: int = Field(default=0)  # MEMORY_DEFAULT_MAX_REFLECTS

    # Ceiling on the rendered ContextMemory. This is the *persisted* artifact and it
    # is prepended to every predictor of the wrapped agent, so it is re-sent on
    # every ReAct iteration (~default_max_iters times per scope) plus once per
    # reflection call. Easily the most cost-sensitive of the three budgets.
    # Approximate item capacity is default_max_context_memory_tokens divided by
    # default_max_context_item_tokens (8192 / 410 ~= 19 items).
    # MEMORY_DEFAULT_MAX_CONTEXT_MEMORY_TOKENS
    default_max_context_memory_tokens: int = Field(default=8192)

    # Per-item ceiling handed to the Distiller/Cartographer as a prompt input, so
    # they keep each context-memory item compact instead of spending the whole memory
    # budget on one verbose entry. Soft limit: it is expressed to the LLM rather
    # than enforced in code (truncating an item could corrupt an exact constant).
    # The hard, memory-wide limit is default_max_context_memory_tokens, enforced by the
    # Evictor. MEMORY_DEFAULT_MAX_CONTEXT_ITEM_TOKENS
    default_max_context_item_tokens: int = Field(default=410)

    # Head+tail cap on the agent trajectory fed to the Distiller. Without it a
    # single tool-heavy scope can produce a 100k+ token trajectory; TwoStepAdapter
    # then sends it twice. 8192 is ~5% of a 128k window and preserves both the
    # orientation steps (60% head) and the conclusions (40% tail).
    default_max_trajectory_tokens: int | None = 8192  # MEMORY_DEFAULT_MAX_TRAJECTORY_TOKENS

    # Head+tail cap on the serialized agent inputs used as the Distiller/Cartographer
    # "question". Only applies when the caller passes no 'question': otherwise
    # every input field is serialized, which for code review means the full patch
    # of every changed file. See Hippocampus.max_question_tokens.
    default_max_question_tokens: int | None = 2048  # MEMORY_DEFAULT_MAX_QUESTION_TOKENS

    # Per-module LLM overrides for the reflection pipeline.
    # Unset fields fall back to the top-level ``default_*`` settings.
    distiller: ReflectionModuleConfig = Field(
        default_factory=ReflectionModuleConfig
    )  # MEMORY_DISTILLER_*
    cartographer: ReflectionModuleConfig = Field(
        default_factory=ReflectionModuleConfig
    )  # MEMORY_CARTOGRAPHER_*


# Env var name (without the MEMORY_ prefix) -> MemoryConfig field name.
# ``memory`` is a nested model and ``Settings`` does not set
# ``env_nested_delimiter``, so pydantic-settings cannot populate these fields
# from the environment on its own. apply_memory_env_overrides() bridges the gap.
MEMORY_ENV_SETTINGS = {
    "BACKEND": "backend",
    "ROOT": "root",
    "S3_BUCKET": "s3_bucket",
    "S3_REGION": "s3_region",
    "S3_ENDPOINT_URL": "s3_endpoint_url",
    "DEFAULT_ENABLED": "default_enabled",
    "DEFAULT_MAX_REFLECTS": "default_max_reflects",
    "DEFAULT_MAX_CONTEXT_MEMORY_TOKENS": "default_max_context_memory_tokens",
    "DEFAULT_MAX_CONTEXT_ITEM_TOKENS": "default_max_context_item_tokens",
    "DEFAULT_MAX_TRAJECTORY_TOKENS": "default_max_trajectory_tokens",
    "DEFAULT_MAX_QUESTION_TOKENS": "default_max_question_tokens",
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

# Env var prefix (after MEMORY_) -> MemoryConfig field holding the nested model.
REFLECTION_MODULE_PREFIXES = {f"{name.upper()}_": name for name in REFLECTION_MODULES}


def apply_memory_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply ``MEMORY_*`` environment variable overrides to the ``memory`` block.

    Maps flat env vars onto the nested ``memory`` config, e.g.::

        MEMORY_BACKEND=s3                          -> memory.backend
        MEMORY_DEFAULT_ENABLED=true                -> memory.default_enabled
        MEMORY_DEFAULT_MAX_CONTEXT_MEMORY_TOKENS=512  -> memory.default_max_context_memory_tokens

    Reflection module overrides use a second level of nesting::

        MEMORY_DISTILLER_MODEL=...        -> memory.distiller.model
        MEMORY_CARTOGRAPHER_TEMPERATURE=0 -> memory.cartographer.temperature

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

        # Reflection module settings: MEMORY_<MODULE>_<SETTING>. Checked before
        # the flat lookup, since e.g. MEMORY_DISTILLER_MODEL has no entry in
        # MEMORY_ENV_SETTINGS and would otherwise be silently dropped.
        module_field = next(
            (
                (field, remainder[len(prefix) :])
                for prefix, field in REFLECTION_MODULE_PREFIXES.items()
                if remainder.startswith(prefix)
            ),
            None,
        )
        if module_field is not None:
            field, setting = module_field
            module_setting = REFLECTION_MODULE_ENV_SETTINGS.get(setting)
            if module_setting is None:
                continue
            module_config = memory_config.setdefault(field, {})
            if not isinstance(module_config, dict):
                continue
            module_config[module_setting] = convert_env_value(value)
            continue

        field = MEMORY_ENV_SETTINGS.get(remainder)
        if field is None:
            continue
        memory_config[field] = convert_env_value(value)

    return config


# Cached singleton store. Avoids reconstructing an S3Client's boto3 client
# (credential resolution + connection pool setup) on every call — see
# get_memory_store() for details. Filesystem stores are cheap to build but
# there's no reason not to reuse them too.
_store: Storage | None = None
_store_built = False


def get_memory_store(settings: Settings) -> Storage | None:
    """Return the cached Storage backend for Hippocampus memory, or None if disabled.

    The store is built once and cached (module-level singleton). This matters
    most for the S3 backend: constructing ``S3Client`` creates a boto3 client,
    which resolves credentials and sets up a connection pool — work we don't
    want repeated on every scope/signature call. Filesystem stores are cheap
    to build, but caching them too keeps the function's behaviour uniform.

    Call :func:`reset_memory_store` after changing settings (e.g. via
    ``reload_settings``) to force a rebuild on next access.

    Filesystem backend: creates a ``FileSystem`` rooted at the resolved
    ``memory.root`` path (``~`` is expanded).

    S3 backend: creates an ``S3Client`` pointing at ``memory.s3_bucket`` with
    optional region / endpoint overrides. Returns None if no bucket is configured.

    Args:
        settings: Application settings.

    Returns:
        Cached Storage instance, or None if storage is not configured.
    """
    global _store, _store_built
    if _store_built:
        return _store

    mem = settings.memory

    if mem.backend == "s3":
        if not mem.s3_bucket:
            _store = None
        else:
            from codespy.tools.storage.s3.client import S3Client

            _store = S3Client(
                bucket=mem.s3_bucket,
                region=mem.s3_region or settings.aws_region,
                endpoint_url=mem.s3_endpoint_url or None,
            )
    else:
        # Filesystem (default)
        from pathlib import Path

        from codespy.tools.storage.filesystem.client import FileSystem

        root = str(Path(mem.root).expanduser().resolve())
        _store = FileSystem(root)

    _store_built = True
    return _store


def reset_memory_store() -> None:
    """Clear the cached memory store so it is rebuilt on next access.

    Call this after reloading settings (e.g. ``reload_settings()``) so a
    changed ``memory`` configuration takes effect.
    """
    global _store, _store_built
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

    store = get_memory_store(settings)
    if store is None:
        return False, "Memory is enabled but storage is not configured (missing S3 bucket?)"

    try:
        store.verify_access()
    except Exception as e:
        return False, f"Memory storage not accessible: {e}"

    return True, f"Memory storage verified ({settings.memory.backend})"
