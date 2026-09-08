"""pg0-embedded lifecycle management for local PostgreSQL development."""

from __future__ import annotations

import atexit
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pg0 import Pg0

_pg0_instance: Pg0 | None = None
_pg0_uri: str | None = None


def _wait_for_ready(
    instance,
    timeout: float = 5.0,
) -> str | None:
    """Poll info() until pg0 reports a healthy instance with a URI.

    The pg0 binary's ``info`` command runs ``psql -c 'SELECT 1'``
    internally and only returns a URI when the healthcheck passes.
    Polling this accounts for slow first-run initialization (initdb,
    pgvector installation) without reimplementing the healthcheck.

    Returns:
        Connection URI string, or None on timeout.
    """
    deadline = time.monotonic() + timeout
    last_info = None
    last_exc = None
    polls = 0
    while time.monotonic() < deadline:
        polls += 1
        try:
            last_info = instance.info()
            last_exc = None
            if last_info and last_info.uri:
                return last_info.uri
            logger.debug(
                "pg0 poll #%d: running=%s, pid=%s, port=%s, uri=%s",
                polls,
                getattr(last_info, "running", None),
                getattr(last_info, "pid", None),
                getattr(last_info, "port", None),
                bool(getattr(last_info, "uri", None)),
            )
        except Exception as exc:
            last_exc = exc
            logger.debug("pg0 poll #%d: info() raised %s: %s", polls, type(exc).__name__, exc)
        time.sleep(0.5)

    # Timeout — log summary
    if last_exc is not None:
        logger.debug(
            "pg0 readiness timed out after %.0fs (%d polls); "
            "last info() error: %s: %s",
            timeout, polls, type(last_exc).__name__, last_exc,
        )
    elif last_info is not None:
        logger.debug(
            "pg0 readiness timed out after %.0fs (%d polls); "
            "last state: running=%s, pid=%s, port=%s, data_dir=%s",
            timeout, polls,
            getattr(last_info, "running", None),
            getattr(last_info, "pid", None),
            getattr(last_info, "port", None),
            getattr(last_info, "data_dir", None),
        )
    else:
        logger.debug(
            "pg0 readiness timed out after %.0fs (%d polls); "
            "info() never returned a result",
            timeout, polls,
        )
    return None


def _get_pg0_logs(instance, lines: int = 30) -> str:
    """Try to capture PostgreSQL log output from the pg0 instance."""
    try:
        log_output = instance.logs(lines=lines)
        if log_output and log_output.strip():
            return log_output.strip()
    except Exception as exc:
        logger.debug("Could not retrieve pg0 logs: %s", exc)
    return "(no pg0 logs available)"


def _try_direct_healthcheck(instance, name: str) -> str | None:
    """Verify PostgreSQL via psycopg when pg0's bundled-psql health check fails.

    Constructs a URI from the Pg0 instance's known parameters and the port
    reported by info(), then runs SELECT 1 through psycopg — the same driver
    EpisodeStore will use.

    Returns the URI on success, None on failure.
    """
    try:
        info = instance.info()
        port = getattr(info, "port", None)
        if not port:
            return None

        username = getattr(instance, "username", "postgres")
        password = getattr(instance, "password", "postgres")
        database = getattr(instance, "database", name)
        uri = f"postgresql://{username}:{password}@127.0.0.1:{port}/{database}"

        import psycopg
        with psycopg.connect(uri, connect_timeout=5) as conn:
            conn.execute("SELECT 1")

        logger.info(
            "pg0 psycopg healthcheck passed on port %d (pg0 bundled-psql unavailable)",
            port,
        )
        return uri
    except Exception as exc:
        logger.debug("Direct psycopg healthcheck failed: %s", exc)
        return None


def _start_round(
    instance, name: str, timeout: float, Pg0AlreadyRunningError: type
) -> str | None:
    """Start pg0 instance and wait for a healthy URI.

    Tries pg0 native health check first, then psycopg fallback.
    Returns URI on success, None on failure.
    """
    try:
        start_info = instance.start()
        logger.debug(
            "pg0 start: running=%s, pid=%s, port=%s, uri=%s",
            getattr(start_info, "running", None),
            getattr(start_info, "pid", None),
            getattr(start_info, "port", None),
            bool(getattr(start_info, "uri", None)),
        )
        if start_info and start_info.uri:
            return start_info.uri
    except Pg0AlreadyRunningError:
        logger.info("pg0 instance already running, reusing it")

    uri = _wait_for_ready(instance, timeout=timeout)
    if not uri:
        uri = _try_direct_healthcheck(instance, name)
    return uri


def get_pg0_uri(name: str = "codespy", port: int | None = None, data_dir: str | None = None) -> str:
    """Start pg0 (if not running) and return its connection URI.

    Strategy:
      Round 1 — start + poll info() for readiness (15s timeout).
      Round 2 — stop + restart without data loss (15s timeout).
      If both fail, raise RuntimeError with manual recovery instructions.
      Data is never dropped automatically.

    The pg0 binary's ``info`` command includes a ``SELECT 1`` healthcheck;
    a URI is only returned when the database is genuinely healthy.

    Args:
        name: Database/instance name to use.
        port: Port to use (auto-detected if None).

    Returns:
        PostgreSQL connection URI string.

    Raises:
        ImportError: If pg0-embedded is not installed.
        RuntimeError: If both rounds fail to produce a healthy instance
            (data is preserved; see error message for manual recovery).
    """
    global _pg0_instance, _pg0_uri

    if _pg0_instance is not None and _pg0_uri is not None:
        return _pg0_uri

    from pg0 import Pg0, Pg0AlreadyRunningError  # raises ImportError

    # Default data_dir to ~/.cache/codespy/pg0 so data persists on the
    # Docker-mounted codespy-cache volume (and locally in the home dir).
    # NOTE: cannot reuse ~/.cache/codespy/memory — old JSON episodes live there.
    if data_dir is None:
        data_dir = str(Path.home() / ".cache" / "codespy" / "pg0")

    logger.info("Starting pg0-embedded PostgreSQL%s...", f" on port {port}" if port else "")
    _pg0_instance = Pg0(port=port, name=name, database=name, data_dir=data_dir)

    try:
        # Round 1
        uri = _start_round(_pg0_instance, name, 15.0, Pg0AlreadyRunningError)
        if uri:
            atexit.register(stop_pg0)
            _pg0_uri = uri
            logger.info("pg0-embedded PostgreSQL ready at %s", uri)
            try:
                _ready_info = _pg0_instance.info()
                _data_dir = getattr(_ready_info, "data_dir", None)
                if _data_dir:
                    logger.info("pg0 data_dir: %s (exists=%s)", _data_dir, Path(_data_dir).is_dir())
            except Exception:
                pass  # non-critical diagnostic
            return uri

        # Round 2: stop + restart (data preserved)
        logger.warning(
            "pg0 not healthy after Round 1; stopping and retrying (data preserved)..."
        )
        try:
            _pg0_instance.stop()
        except Exception:
            pass
        time.sleep(2)

        _pg0_instance = Pg0(port=port, name=name, database=name, data_dir=data_dir)
        uri = _start_round(_pg0_instance, name, 15.0, Pg0AlreadyRunningError)
        if uri:
            atexit.register(stop_pg0)
            _pg0_uri = uri
            logger.info("pg0-embedded PostgreSQL ready (after restart) at %s", uri)
            try:
                _ready_info = _pg0_instance.info()
                _data_dir = getattr(_ready_info, "data_dir", None)
                if _data_dir:
                    logger.info("pg0 data_dir: %s (exists=%s)", _data_dir, Path(_data_dir).is_dir())
            except Exception:
                pass  # non-critical diagnostic
            return uri

        # Both failed
        pg_logs = _get_pg0_logs(_pg0_instance)
        raise RuntimeError(
            f"pg0-embedded failed to start PostgreSQL instance '{name}' "
            f"after two attempts (data preserved).\n"
            f"PostgreSQL logs:\n{pg_logs}\n\n"
            f"Troubleshooting:\n"
            f"  1. Check if another PostgreSQL is using the same port\n"
            f"  2. Check pg0 status: pg0 info --name {name}\n"
            f"  3. View logs: pg0 logs --name {name}\n"
            f"  4. If data is corrupt, manually reset: "
            f"pg0 drop --name {name} && rm -rf ~/.pg0/instances/{name}"
        )

    except Exception:
        if _pg0_instance is not None:
            try:
                _pg0_instance.stop()
            except Exception:
                pass
            _pg0_instance = None
            _pg0_uri = None
        raise


def stop_pg0() -> None:
    """Stop the cached pg0 instance (if running)."""
    global _pg0_instance, _pg0_uri

    if _pg0_instance is not None:
        logger.info("Stopping pg0-embedded PostgreSQL...")
        try:
            _pg0_instance.stop()
        except Exception as e:
            logger.warning(f"Error stopping pg0: {e}")
        finally:
            _pg0_instance = None
            _pg0_uri = None
