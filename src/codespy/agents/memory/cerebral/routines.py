"""Maintenance routine repair for Hindsight semantic memory.

Upstream #2638: the maintenance routines (banks_needing_consolidation,
mental_models_with_cron, schemas_with_expired_rows, schemas_with_expired_operations)
install into and are called from `database_schema`. Cerebral sets its schema only
on the tenant extension, so `HINDSIGHT_API_DATABASE_SCHEMA` defaults to `public`,
but migrations run with `target_schema="semantic"`. With `database_schema=public`
and `target_schema=semantic`, the install migrations skip creating the routines,
and the maintenance loop warns that `public.<routine>() does not exist`.

This module provides a one-time idempotent repair for existing databases.
Fresh databases get the routines via the normal migration path once the env
var is set (see cerebral.py).
"""

from __future__ import annotations

import glob
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hindsight_api import MemoryEngine

logger = logging.getLogger(__name__)

# The four maintenance routines that must exist in the configured schema
ROUTINE_NAMES = [
    "banks_needing_consolidation",
    "schemas_with_expired_rows",
    "mental_models_with_cron",
    "schemas_with_expired_operations",
]

# Upstream migration that defines the current routine bodies
REVISION_ID = "c8b4e2a71f95"


def _get_migration_path() -> Path | None:
    """Find the migration file by revision ID in the hindsight_api package."""
    try:
        import hindsight_api
    except ImportError:
        return None

    pkg_dir = Path(hindsight_api.__file__).parent
    versions_dir = pkg_dir / "alembic" / "versions"
    if not versions_dir.exists():
        return None

    pattern = str(versions_dir / f"{REVISION_ID}_*.py")
    matches = glob.glob(pattern)
    if len(matches) == 1:
        return Path(matches[0])
    return None


def collect_routine_sql(schema: str) -> dict[str, str]:
    """Extract SQL for all four maintenance routines from the upstream migration.

    Returns a mapping from routine name -> CREATE OR REPLACE FUNCTION SQL.
    The SQL is qualified with the given schema (e.g., "semantic".routine_name).

    Raises RuntimeError if the migration file is missing or if the configured
    schema does not match the requested schema (meaning HINDSIGHT_API_DATABASE_SCHEMA
    was overridden to a different value).
    """
    import hindsight_api
    from hindsight_api.alembic._owned import externally_owned

    migration_path = _get_migration_path()
    if migration_path is None:
        raise RuntimeError(
            f"Migration {REVISION_ID} not found in hindsight_api. "
            "The repair cannot proceed."
        )

    # Load the migration module without registering in sys.modules permanently
    module_name = f"_codespy_hs_{REVISION_ID}"
    spec = importlib.util.spec_from_file_location(module_name, migration_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {migration_path}")

    module = importlib.util.module_from_spec(spec)
    # Temporarily register to allow the loader to work
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[module_name]

    # Prepare the collector that intercepts execute_unless_owned calls
    captured: dict[str, str] = {}

    def collector(name: str, sql: str) -> None:
        # Skip if externally owned (operator is managing this routine)
        if externally_owned(name):
            logger.info(
                "Skipping routine %r: named in HINDSIGHT_API_EXTERNALLY_OWNED_ROUTINES",
                name,
            )
            return
        captured[name] = sql

    # Mock the alembic context so _pg_upgrade can run without an active env
    # The migration only uses context.config.get_main_option("target_schema")
    module.context = SimpleNamespace(
        config=SimpleNamespace(
            get_main_option=lambda k: schema if k == "target_schema" else None
        )
    )
    # Replace execute_unless_owned with our collector
    module.execute_unless_owned = collector

    # Run the PostgreSQL upgrade function directly (not upgrade(), which needs op.get_bind)
    module._pg_upgrade()

    # Validate results
    if not captured:
        # _pg_upgrade returned early because configured schema != target_schema
        raise RuntimeError(
            f"Migration {REVISION_ID} produced no SQL: configured database_schema "
            f"does not match requested schema {schema!r}. "
            "Is HINDSIGHT_API_DATABASE_SCHEMA set to a different value?"
        )

    # Verify all expected routines are present (unless externally owned)
    missing = set(ROUTINE_NAMES) - set(captured.keys())
    if missing:
        # Some routines may be externally owned, that's fine
        for name in missing:
            if externally_owned(name):
                continue
            raise RuntimeError(
                f"Migration {REVISION_ID} did not produce SQL for routine {name!r}"
            )

    # Verify SQL contains the expected schema qualification
    for name, sql in captured.items():
        expected = f'"{schema}".{name}'
        if expected not in sql:
            raise RuntimeError(
                f"Routine {name!r} SQL does not contain expected qualification {expected!r}"
            )

    return captured


async def ensure_maintenance_routines(engine: MemoryEngine, schema: str) -> list[str]:
    """Ensure all four maintenance routines exist in the given schema.

    Returns a list of routine names that were installed (may be empty if all
    already exist). This is idempotent: running twice is a no-op on the second
    run (the presence check returns all four).

    On any failure (migration file missing, permissions error, etc.), logs a
    warning and returns an empty list. Cerebral stays usable; the maintenance
    warnings will continue as the visible symptom.
    """
    from hindsight_api.engine.db_utils import acquire_with_retry

    try:
        sql_map = collect_routine_sql(schema)
    except RuntimeError as exc:
        logger.warning(
            "Failed to collect maintenance routine SQL (revision %s): %s",
            REVISION_ID,
            exc,
            exc_info=True,
        )
        return []

    # Check which routines are already present
    try:
        async with acquire_with_retry(engine._backend, max_retries=1) as conn:
            rows = await conn.fetch(
                """
                SELECT p.proname
                FROM pg_proc p
                JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = $1
                  AND p.proname = ANY($2::text[])
                """,
                schema,
                ROUTINE_NAMES,
            )
            present = {row["proname"] for row in rows}
    except Exception as exc:
        logger.warning(
            "Failed to check existing routines: %s",
            exc,
            exc_info=True,
        )
        return []

    missing = [name for name in ROUTINE_NAMES if name not in present and name in sql_map]
    if not missing:
        logger.debug("All maintenance routines already present in schema %r", schema)
        return []

    # Install missing routines in one transaction
    installed: list[str] = []
    try:
        async with acquire_with_retry(engine._backend, max_retries=1) as conn:
            async with conn.transaction():
                for name in missing:
                    sql = sql_map[name]
                    await conn.execute(sql)
                    installed.append(name)
    except Exception as exc:
        logger.warning(
            "Failed to install maintenance routines %s: %s",
            missing,
            exc,
            exc_info=True,
        )
        # Concurrent installs can fail with "tuple concurrently updated";
        # next boot's presence check will succeed if another process won.
        return []

    logger.info(
        "Installed maintenance routines in schema %r: %s",
        schema,
        installed,
    )
    return installed
