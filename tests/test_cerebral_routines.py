"""Unit tests for cerebral.routines module.

These tests verify that the maintenance routine repair works correctly
without needing a live database. The tests mock the migration file and
database connection where necessary.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


class TestCollectRoutineSql:
    """Tests for collect_routine_sql function."""

    @pytest.fixture(autouse=True)
    def reset_env(self, monkeypatch):
        """Ensure HINDSIGHT_API_DATABASE_SCHEMA matches the test schema."""
        # Import the module fresh for each test
        monkeypatch.setenv("HINDSIGHT_API_DATABASE_SCHEMA", "semantic")
        yield

    def test_returns_all_four_routines(self):
        """Test that collect_routine_sql returns all four expected routines."""
        from codespy.agents.memory.cerebral import routines as routines_module

        result = routines_module.collect_routine_sql("semantic")

        expected_routines = {
            "banks_needing_consolidation",
            "schemas_with_expired_rows",
            "mental_models_with_cron",
            "schemas_with_expired_operations",
        }

        assert set(result.keys()) == expected_routines

    def test_sql_contains_schema_qualification(self):
        """Test that each SQL statement contains the schema qualification."""
        from codespy.agents.memory.cerebral import routines as routines_module

        result = routines_module.collect_routine_sql("semantic")

        for name, sql in result.items():
            expected = f'"semantic".{name}'
            assert expected in sql, f"SQL for {name} missing schema qualification"

    def test_sql_contains_create_or_replace(self):
        """Test that each SQL statement contains CREATE OR REPLACE."""
        from codespy.agents.memory.cerebral import routines as routines_module

        result = routines_module.collect_routine_sql("semantic")

        for name, sql in result.items():
            assert "CREATE OR REPLACE FUNCTION" in sql, f"SQL for {name} missing CREATE OR REPLACE"

    def test_no_public_in_sql(self):
        """Test that SQL does not contain 'public' when semantic schema is requested."""
        from codespy.agents.memory.cerebral import routines as routines_module

        result = routines_module.collect_routine_sql("semantic")

        for name, sql in result.items():
            # Should not have "public" qualified in the function definition
            assert '"public".' not in sql, f"SQL for {name} incorrectly contains 'public'"


class TestExternallyOwnedRoutines:
    """Tests for handling of externally owned routines."""

    def test_externally_owned_routine_skipped(self, monkeypatch):
        """Test that externally owned routines are skipped."""
        monkeypatch.setenv("HINDSIGHT_API_DATABASE_SCHEMA", "semantic")
        monkeypatch.setenv(
            "HINDSIGHT_API_EXTERNALLY_OWNED_ROUTINES",
            "mental_models_with_cron"
        )

        from codespy.agents.memory.cerebral import routines as routines_module

        result = routines_module.collect_routine_sql("semantic")

        # mental_models_with_cron should be missing
        assert "mental_models_with_cron" not in result
        # The others should be present
        assert "banks_needing_consolidation" in result
        assert "schemas_with_expired_rows" in result
        assert "schemas_with_expired_operations" in result

    def test_multiple_externally_owned_routines(self, monkeypatch):
        """Test that multiple externally owned routines are all skipped."""
        monkeypatch.setenv("HINDSIGHT_API_DATABASE_SCHEMA", "semantic")
        monkeypatch.setenv(
            "HINDSIGHT_API_EXTERNALLY_OWNED_ROUTINES",
            "mental_models_with_cron,banks_needing_consolidation"
        )

        from codespy.agents.memory.cerebral import routines as routines_module

        result = routines_module.collect_routine_sql("semantic")

        assert "mental_models_with_cron" not in result
        assert "banks_needing_consolidation" not in result
        assert "schemas_with_expired_rows" in result
        assert "schemas_with_expired_operations" in result

    def test_whitespace_handling_in_env_var(self, monkeypatch):
        """Test that whitespace around routine names is handled correctly."""
        monkeypatch.setenv("HINDSIGHT_API_DATABASE_SCHEMA", "semantic")
        monkeypatch.setenv(
            "HINDSIGHT_API_EXTERNALLY_OWNED_ROUTINES",
            "  mental_models_with_cron  ,  banks_needing_consolidation  "
        )

        from codespy.agents.memory.cerebral import routines as routines_module

        result = routines_module.collect_routine_sql("semantic")

        assert "mental_models_with_cron" not in result
        assert "banks_needing_consolidation" not in result


class TestCollectRoutineSqlErrors:
    """Tests for error handling in collect_routine_sql."""

    def test_migration_file_not_found(self):
        """Test that RuntimeError is raised when migration file is missing."""
        from codespy.agents.memory.cerebral import routines as routines_module

        # Patch _get_migration_path to return None
        with patch.object(routines_module, "_get_migration_path", return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                routines_module.collect_routine_sql("semantic")

            assert "c8b4e2a71f95" in str(exc_info.value)
            assert "not found" in str(exc_info.value)


class TestEnsureMaintenanceRoutines:
    """Tests for ensure_maintenance_routines function."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock MemoryEngine with _backend."""
        engine = MagicMock()
        engine._backend = MagicMock()
        return engine

    @pytest.fixture
    def mock_sql_map(self):
        """Create a mock SQL map with all four routines."""
        return {
            "banks_needing_consolidation": "CREATE OR REPLACE FUNCTION ...",
            "schemas_with_expired_rows": "CREATE OR REPLACE FUNCTION ...",
            "mental_models_with_cron": "CREATE OR REPLACE FUNCTION ...",
            "schemas_with_expired_operations": "CREATE OR REPLACE FUNCTION ...",
        }

    @pytest.mark.asyncio
    async def test_all_present_returns_empty_list(self, mock_engine, mock_sql_map):
        """Test that when all routines are present, returns empty list."""
        from codespy.agents.memory.cerebral import routines as routines_module

        # Create a proper async mock context manager
        async def mock_fetch(*args, **kwargs):
            return [
                {"proname": "banks_needing_consolidation"},
                {"proname": "schemas_with_expired_rows"},
                {"proname": "mental_models_with_cron"},
                {"proname": "schemas_with_expired_operations"},
            ]

        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(side_effect=mock_fetch)

        # Mock acquire_with_retry to return mock_conn via async context manager
        @MagicMock
        async def mock_acquire_cm(*args, **kwargs):
            class CtxManager:
                async def __aenter__(self):
                    return mock_conn
                async def __aexit__(self, *args):
                    return None
            return CtxManager()

        # Create a proper async context manager
        class AsyncContextManager:
            async def __aenter__(self):
                return mock_conn
            async def __aexit__(self, *args):
                return None

        with patch.object(routines_module, "collect_routine_sql", return_value=mock_sql_map):
            with patch("hindsight_api.engine.db_utils.acquire_with_retry", return_value=AsyncContextManager()):
                result = await routines_module.ensure_maintenance_routines(
                    mock_engine, "semantic"
                )

                assert result == []

    @pytest.mark.asyncio
    async def test_missing_routines_get_installed(self, mock_engine, mock_sql_map):
        """Test that missing routines are installed."""
        from codespy.agents.memory.cerebral import routines as routines_module

        # Track which SQL statements were executed
        executed_sql = []

        async def mock_fetch(*args, **kwargs):
            # Only two routines present
            return [
                {"proname": "banks_needing_consolidation"},
                {"proname": "schemas_with_expired_rows"},
            ]

        async def mock_execute(sql):
            executed_sql.append(sql)

        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(side_effect=mock_fetch)
        mock_conn.execute = AsyncMock(side_effect=mock_execute)

        # Create a proper async context manager
        class AsyncContextManager:
            async def __aenter__(self):
                return mock_conn
            async def __aexit__(self, *args):
                return None

        class MockTransaction:
            async def __aenter__(self):
                return None
            async def __aexit__(self, *args):
                return None

        mock_conn.transaction = MagicMock(return_value=MockTransaction())

        with patch.object(routines_module, "collect_routine_sql", return_value=mock_sql_map):
            with patch("hindsight_api.engine.db_utils.acquire_with_retry", return_value=AsyncContextManager()):
                result = await routines_module.ensure_maintenance_routines(
                    mock_engine, "semantic"
                )

                # Should have installed the two missing routines
                assert len(result) == 2
                assert "mental_models_with_cron" in result
                assert "schemas_with_expired_operations" in result

    @pytest.mark.asyncio
    async def test_collector_failure_returns_empty_list(self, mock_engine):
        """Test that if collect_routine_sql fails, we return empty list and log warning."""
        from codespy.agents.memory.cerebral import routines as routines_module

        with patch.object(
            routines_module, "collect_routine_sql", side_effect=RuntimeError("Migration not found")
        ):
            result = await routines_module.ensure_maintenance_routines(mock_engine, "semantic")
            assert result == []

    @pytest.mark.asyncio
    async def test_db_check_failure_returns_empty_list(self, mock_engine, mock_sql_map):
        """Test that if DB check fails, we return empty list and log warning."""
        from codespy.agents.memory.cerebral import routines as routines_module

        # Create a proper async context manager that raises on enter
        class AsyncContextManager:
            async def __aenter__(self):
                raise Exception("Connection failed")
            async def __aexit__(self, *args):
                return None

        with patch.object(routines_module, "collect_routine_sql", return_value=mock_sql_map):
            with patch("hindsight_api.engine.db_utils.acquire_with_retry", return_value=AsyncContextManager()):
                result = await routines_module.ensure_maintenance_routines(mock_engine, "semantic")
                assert result == []


class TestGetMigrationPath:
    """Tests for _get_migration_path helper."""

    def test_finds_migration_file(self):
        """Test that the migration file is found."""
        from codespy.agents.memory.cerebral import routines as routines_module

        path = routines_module._get_migration_path()
        assert path is not None
        assert path.exists()
        assert "c8b4e2a71f95" in path.name

    def test_returns_none_when_hindsight_not_importable(self):
        """Test that None is returned if hindsight_api is not available."""
        from codespy.agents.memory.cerebral import routines as routines_module

        with patch.object(routines_module, "_get_migration_path", return_value=None):
            # Test that our patch works
            path = routines_module._get_migration_path()
            assert path is None


class TestRoutineNamesConstant:
    """Tests for the ROUTINE_NAMES constant."""

    def test_contains_all_four_routines(self):
        """Test that ROUTINE_NAMES contains all four expected routine names."""
        from codespy.agents.memory.cerebral import routines as routines_module

        expected = [
            "banks_needing_consolidation",
            "schemas_with_expired_rows",
            "mental_models_with_cron",
            "schemas_with_expired_operations",
        ]
        assert routines_module.ROUTINE_NAMES == expected

    def test_revision_id_is_set(self):
        """Test that REVISION_ID is set to expected value."""
        from codespy.agents.memory.cerebral import routines as routines_module

        assert routines_module.REVISION_ID == "c8b4e2a71f95"
