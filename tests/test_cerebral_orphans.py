"""Tests for the orphaned async operation repair module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("hindsight_api")

from codespy.agents.memory.cerebral.orphans import (
    _ORPHAN_OPERATION_TYPES,
    repair_orphaned_operations,
)


class TestRepairOrphanedOperations:
    """Tests for repair_orphaned_operations function."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock MemoryEngine with a mock _backend."""
        engine = MagicMock()
        engine._backend = MagicMock()
        return engine

    @pytest.fixture
    def mock_conn(self):
        """Create a mock async connection."""
        conn = AsyncMock()
        return conn

    @pytest.mark.asyncio
    async def test_marks_pending_operations_as_failed(self, mock_engine, mock_conn):
        """Test that pending operations older than threshold are marked failed."""
        # Set up mock rows to return (simulating 2 orphaned operations)
        mock_rows = [
            {"operation_id": "op-1", "operation_type": "refresh_mental_model"},
            {"operation_id": "op-2", "operation_type": "consolidation"},
        ]
        mock_conn.fetch.return_value = mock_rows

        with patch(
            "codespy.agents.memory.cerebral.orphans.acquire_with_retry"
        ) as mock_acquire:
            mock_acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_acquire.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await repair_orphaned_operations(
                mock_engine, "semantic", "test-bank", older_than_s=3600
            )

        assert result == 2

        # Verify the SQL was called with correct parameters
        call_args = mock_conn.fetch.call_args
        assert call_args[0][1] == "test-bank"  # bank_id
        assert "orphaned: left pending by inline SyncTaskBackend" in call_args[0][2]
        assert call_args[0][3] == _ORPHAN_OPERATION_TYPES
        assert call_args[0][4] == 3600  # older_than_s

    @pytest.mark.asyncio
    async def test_no_orphans_found(self, mock_engine, mock_conn):
        """Test when no orphaned operations are found."""
        mock_conn.fetch.return_value = []

        with patch(
            "codespy.agents.memory.cerebral.orphans.acquire_with_retry"
        ) as mock_acquire:
            mock_acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_acquire.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await repair_orphaned_operations(
                mock_engine, "semantic", "test-bank"
            )

        assert result == 0

    @pytest.mark.asyncio
    async def test_db_error_returns_zero(self, mock_engine):
        """Test that database errors are caught and return 0."""
        with patch(
            "codespy.agents.memory.cerebral.orphans.acquire_with_retry"
        ) as mock_acquire:
            mock_acquire.side_effect = Exception("connection failed")

            result = await repair_orphaned_operations(
                mock_engine, "semantic", "test-bank"
            )

        assert result == 0

    @pytest.mark.asyncio
    async def test_uses_default_age_threshold(self, mock_engine, mock_conn):
        """Test that default age threshold (3600s) is used when not specified."""
        mock_conn.fetch.return_value = []

        with patch(
            "codespy.agents.memory.cerebral.orphans.acquire_with_retry"
        ) as mock_acquire:
            mock_acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_acquire.return_value.__aexit__ = AsyncMock(return_value=None)

            await repair_orphaned_operations(mock_engine, "semantic", "test-bank")

            call_args = mock_conn.fetch.call_args
            assert call_args[0][4] == 3600  # Default older_than_s

    @pytest.mark.asyncio
    async def test_custom_age_threshold(self, mock_engine, mock_conn):
        """Test that custom age threshold is passed through."""
        mock_conn.fetch.return_value = []

        with patch(
            "codespy.agents.memory.cerebral.orphans.acquire_with_retry"
        ) as mock_acquire:
            mock_acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_acquire.return_value.__aexit__ = AsyncMock(return_value=None)

            await repair_orphaned_operations(
                mock_engine, "semantic", "test-bank", older_than_s=7200
            )

            call_args = mock_conn.fetch.call_args
            assert call_args[0][4] == 7200

    @pytest.mark.asyncio
    async def test_only_affects_specified_bank(self, mock_engine, mock_conn):
        """Test that only operations for the specified bank are affected."""
        mock_conn.fetch.return_value = []

        with patch(
            "codespy.agents.memory.cerebral.orphans.acquire_with_retry"
        ) as mock_acquire:
            mock_acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_acquire.return_value.__aexit__ = AsyncMock(return_value=None)

            await repair_orphaned_operations(
                mock_engine, "semantic", "specific-bank"
            )

            call_args = mock_conn.fetch.call_args
            assert call_args[0][1] == "specific-bank"


class TestOrphanOperationTypes:
    """Tests for the operation types constant."""

    def test_includes_refresh_mental_model(self):
        """Test that refresh_mental_model is included."""
        assert "refresh_mental_model" in _ORPHAN_OPERATION_TYPES

    def test_includes_consolidation(self):
        """Test that consolidation is included."""
        assert "consolidation" in _ORPHAN_OPERATION_TYPES

    def test_has_exactly_two_types(self):
        """Test that there are exactly two operation types."""
        assert len(_ORPHAN_OPERATION_TYPES) == 2
