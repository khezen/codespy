"""Lightweight config utilities. No internal imports to avoid circular deps."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, SecretStr


def secret_value(s: SecretStr | None) -> str:
    """Extract the plain-text value from a SecretStr, or return empty string.

    Use at API boundaries where a raw string is needed (library calls,
    environment variables, URL interpolation). The empty-string return for
    None/empty ensures callers can use standard truthiness checks:

        token = secret_value(settings.github_token)
        if token:
            # token is configured and non-empty
    """
    if s is None:
        return ""
    return s.get_secret_value()


def convert_env_value(value: str) -> Any:
    """Convert environment variable string to appropriate Python type.

    Conversion rules:
    - Empty string: treated as unset (return None to signal skip)
    - Literal "null": returns Python None (explicit reset)
    - JSON array/object: parsed via json.loads
    - "true"/"false": converted to bool
    - Other values: returned as raw string (let pydantic coerce)
    """
    # Empty string means unset - skip this value
    if value == "":
        return None  # Signal to skip

    # Literal null means explicit reset to None
    if value.lower() == "null":
        return None

    # JSON arrays/objects
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    # Boolean values
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # Return raw string - pydantic will coerce int/float as needed
    return value


def build_env_map(
    sections: dict[str, type[BaseModel]],
    *,
    collapsed_paths: set[tuple[str, ...]] | None = None,
    full_name_paths: set[tuple[str, ...]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Build a mapping of env var name -> config path tuple.

    Args:
        sections: Dict of section name -> BaseModel subclass (e.g., {"llm": LLMConfig})
        collapsed_paths: Set of path prefixes to collapse (drop the prefix segments).
            For example, {("llm",)} drops "LLM_" from all descendants of llm.
        full_name_paths: Set of full paths that keep their full name (not collapsed).
            These take precedence over collapsed_paths. For example,
            {("llm", "retries")} keeps "LLM_RETRIES" instead of "RETRIES".

    Returns:
        Dict mapping env var name to tuple path (e.g., {"DEFAULT_MODEL": ("llm", "default_model")})

    Examples:
        >>> build_env_map(
        ...     sections={"llm": LLMConfig, "review": ReviewConfig},
        ...     collapsed_paths={("llm",), ("memory", "hippocampus"), ("memory", "cerebral")},
        ...     full_name_paths={("llm", "retries"), ("llm", "timeout")},
        ... )
        # Produces:
        # {
        #     "DEFAULT_MODEL": ("llm", "default_model"),  # llm prefix dropped
        #     "LLM_RETRIES": ("llm", "retries"),          # full_name_paths keeps LLM_
        #     "OPENAI_API_KEY": ("llm", "openai_api_key"),  # credentials are bare
        #     "REVIEW_SCOPE_ENABLED": ("review", "scope", "enabled"),  # review prefix kept
        #     "MEMORY_DISTILLER_MODEL": ("memory", "hippocampus", "distiller", "model"),
        # }
    """
    env_map: dict[str, tuple[str, ...]] = {}
    collapsed_paths = collapsed_paths or set()
    full_name_paths = full_name_paths or set()

    for section_name, model_cls in sections.items():
        _build_env_map_for_model(
            model_cls,
            prefix=(section_name,),
            collapsed_paths=collapsed_paths,
            full_name_paths=full_name_paths,
            env_map=env_map,
        )

    return env_map


def _build_env_map_for_model(
    model_cls: type[BaseModel],
    prefix: tuple[str, ...],
    collapsed_paths: set[tuple[str, ...]],
    full_name_paths: set[tuple[str, ...]],
    env_map: dict[str, tuple[str, ...]],
) -> None:
    """Recursively build env map for a model and its nested models."""
    for field_name, field_info in model_cls.model_fields.items():
        annotation = field_info.annotation

        # Use alias if available, otherwise field name
        seg = field_info.alias or field_name
        p = prefix + (seg,)

        # Determine the env var name
        if p in full_name_paths:
            # Use full path (keep all segments)
            env_name = "_".join(p).upper()
        else:
            # Apply collapsing: drop each segment for which the path up to that segment is in collapsed_paths
            segments = []
            for i, segment in enumerate(p):
                if p[: i + 1] not in collapsed_paths:
                    segments.append(segment)
            env_name = "_".join(segments).upper()

        # Check if this is a nested BaseModel
        origin = getattr(annotation, "__origin__", None)
        if origin is not None:
            # Handle Optional[X] or List[X] etc - unwrap
            args = getattr(annotation, "__args__", ())
            if args:
                annotation = args[0]

        # Check if the unwrapped annotation is a BaseModel subclass
        try:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                # Recurse into nested model, but don't add to env_map directly
                _build_env_map_for_model(
                    annotation,
                    prefix=p,
                    collapsed_paths=collapsed_paths,
                    full_name_paths=full_name_paths,
                    env_map=env_map,
                )
                continue
        except TypeError:
            pass

        # This is a leaf field - add to map with duplicate check
        if env_name in env_map:
            old_path = env_map[env_name]
            if old_path != p:
                raise ValueError(f"Duplicate env var {env_name}: {old_path} vs {p}")
            # Same path, already added - skip
        else:
            env_map[env_name] = p


def apply_env_overrides(config: dict[str, Any], env_map: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Apply environment variable overrides to a config dict.

    Args:
        config: The YAML-derived config dict to mutate.
        env_map: Mapping of env var name -> config path tuple.

    Returns:
        The same dict, with env overrides applied.
    """
    # Load .env file first, then actual env vars (env vars win)
    env_vars = {**dotenv_values(".env"), **os.environ}

    for env_name, path in env_map.items():
        value = env_vars.get(env_name)
        if value is None:
            continue

        converted = convert_env_value(value)
        if converted is None and value != "null":
            # Empty string - skip
            continue

        # Navigate/create the nested dict structure
        target = config
        for key in path[:-1]:
            if key not in target or target[key] is None:
                target[key] = {}
            target = target[key]

        # Set the value
        target[path[-1]] = converted

    return config
