"""Orphaned async operation repair for Hindsight semantic memory.

When SyncTaskBackend runs tasks inline, failures leave rows in `pending` status
because there's no worker poller to mark them failed or reschedule them. This
module provides startup repair to mark old pending operations as failed so that
deduplication no longer blocks new submissions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hindsight_api import MemoryEngine

logger = logging.getLogger(__name__)

# Operation types that can be orphaned by inline SyncTaskBackend execution
_ORPHAN_OPERATION_TYPES = ["refresh_mental_model", "consolidation"]

# Default age threshold for considering an operation orphaned (1 hour)
_DEFAULT_OLDER_THAN_SECONDS = 3600

# Max error message length for the error_message column
_MAX_ERROR_MESSAGE_LENGTH = 5000


async def repair_orphaned_operations(
    engine: MemoryEngine,
    schema: str,
    bank_id: str,
    older_than_s: int = _DEFAULT_OLDER_THAN_SECONDS,
) -> int:
    """Mark orphaned pending operations as failed.

    Pending operations older than the threshold are marked failed with a
    diagnostic message. This allows new submissions to proceed (the refresh
    deduplication only blocks when a pending row exists).

    Args:
        engine: The MemoryEngine instance.
        schema: The schema name (e.g., "semantic").
        bank_id: The bank ID to repair operations for.
        older_than_s: Age threshold in seconds (default 3600 = 1 hour).

    Returns:
        Number of rows marked as failed.
    """
    from hindsight_api.engine.db_utils import acquire_with_retry

    error_message = (
        "orphaned: left pending by inline SyncTaskBackend (repaired by codespy)"
    )
    # Truncate if needed (though this message is well under the limit)
    if len(error_message) > _MAX_ERROR_MESSAGE_LENGTH:
        error_message = error_message[:_MAX_ERROR_MESSAGE_LENGTH]

    try:
        async with acquire_with_retry(engine._backend, max_retries=1) as conn:
            rows = await conn.fetch(
                f"""
                UPDATE "{schema}".async_operations
                SET status = 'failed',
                    error_message = $2,
                    updated_at = NOW()
                WHERE bank_id = $1
                  AND status = 'pending'
                  AND operation_type = ANY($3::text[])
                  AND updated_at < NOW() - MAKE_INTERVAL(SECS => $4)
                RETURNING operation_id, operation_type
                """,
                bank_id,
                error_message,
                _ORPHAN_OPERATION_TYPES,
                older_than_s,
            )

        if rows:
            by_type: dict[str, int] = {}
            for row in rows:
                op_type = row["operation_type"]
                by_type[op_type] = by_type.get(op_type, 0) + 1

            logger.info(
                "Repaired %d orphaned operation(s) in bank %r: %s",
                len(rows),
                bank_id,
                ", ".join(f"{count} {op_type}" for op_type, count in by_type.items()),
            )
        else:
            logger.debug(
                "No orphaned operations found in bank %r (older than %ds)",
                bank_id,
                older_than_s,
            )

        return len(rows)

    except Exception as exc:
        logger.warning(
            "Failed to repair orphaned operations in bank %r: %s",
            bank_id,
            exc,
            exc_info=True,
        )
        return 0
