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
    bare_fields: dict[str, set[str]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Build a mapping of env var name -> config path tuple.

    Args:
        sections: Dict of section name -> BaseModel subclass (e.g., {"llm": LLMConfig})
        bare_fields: Optional dict of section name -> set of field names that
            should use bare (non-prefixed) env var names (e.g., {"llm": {"openai_api_key"}})

    Returns:
        Dict mapping env var name to tuple path (e.g., {"LLM_DEFAULT_MODEL": ("llm", "default_model")})
    """
    env_map: dict[str, tuple[str, ...]] = {}
    bare_fields = bare_fields or {}

    for section_name, model_cls in sections.items():
        section_bare = bare_fields.get(section_name, set())
        _build_env_map_for_model(
            model_cls,
            prefix=(section_name,),
            bare_fields=section_bare,
            env_map=env_map,
        )

    # Validate uniqueness
    seen: dict[str, str] = {}
    for env_name, path in env_map.items():
        if env_name in seen:
            raise ValueError(
                f"Duplicate env var name '{env_name}' from paths {seen[env_name]} and {path}"
            )
        seen[env_name] = str(path)

    return env_map


def _build_env_map_for_model(
    model_cls: type[BaseModel],
    prefix: tuple[str, ...],
    bare_fields: set[str],
    env_map: dict[str, tuple[str, ...]],
) -> None:
    """Recursively build env map for a model and its nested models."""
    for field_name, field_info in model_cls.model_fields.items():
        annotation = field_info.annotation

        # Check if this field is a nested BaseModel (recurse) or a leaf
        is_bare = field_name in bare_fields and len(prefix) == 1  # Only direct children

        if is_bare:
            # Use bare field name (e.g., OPENAI_API_KEY instead of LLM_OPENAI_API_KEY)
            env_name = field_name.upper()
        else:
            # Use full path (e.g., LLM_DEFAULT_MODEL)
            env_name = "_".join(prefix + (field_name,)).upper()

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
                    prefix=prefix + (field_name,),
                    bare_fields=set(),  # Nested models don't use bare names
                    env_map=env_map,
                )
                continue
        except TypeError:
            pass

        # This is a leaf field - add to map
        env_map[env_name] = prefix + (field_name,)


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
