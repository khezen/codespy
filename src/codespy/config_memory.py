"""Memory (Hippocampus) configuration and storage factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from codespy.tools.storage.base import Storage

if TYPE_CHECKING:
    from codespy.config import Settings


MemoryBackend = Literal["filesystem", "s3"]


class MemoryConfig(BaseModel):
    """Global memory (Hippocampus) configuration.

    Controls where episodes are persisted and the default reflection knobs
    applied to every agent. Per-signature ``memory:`` blocks override the
    ``default_*`` values.
    """

    # Storage backend
    backend: MemoryBackend = "filesystem"  # MEMORY_BACKEND
    root: str = "~/.cache/codespy/memory"  # MEMORY_ROOT (filesystem backend)
    s3_bucket: str | None = None           # MEMORY_S3_BUCKET (s3 backend)
    s3_region: str | None = None           # MEMORY_S3_REGION (falls back to aws_region)
    s3_endpoint_url: str | None = None     # MEMORY_S3_ENDPOINT_URL (MinIO/S3-compatible)

    # Reflection defaults — overridable per-signature
    default_enabled: bool = False                  # MEMORY_DEFAULT_ENABLED
    default_max_reflects: int = Field(default=0)   # MEMORY_DEFAULT_MAX_REFLECTS
    default_token_budget: int = Field(default=1024) # MEMORY_DEFAULT_TOKEN_BUDGET
    default_max_trajectory_tokens: int | None = None  # MEMORY_DEFAULT_MAX_TRAJECTORY_TOKENS


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
