"""Tests for memory storage configuration and access verification."""

from unittest.mock import MagicMock, patch

import pytest

from codespy.config_memory import (
    _generate_bank_id,
    apply_memory_env_overrides,
    get_episode_store,
    reset_episode_store,
    verify_memory_access,
)


class TestGenerateBankId:
    """Tests for _generate_bank_id function."""

    def test_generate_bank_id_returns_codespy(self):
        """Should return 'codespy' as the default bank_id."""
        result = _generate_bank_id()
        assert result == "codespy"


class TestApplyMemoryEnvOverrides:
    """Tests for apply_memory_env_overrides function."""

    def test_override_postgres_uri(self, monkeypatch):
        """MEMORY_POSTGRES_URI should set memory.postgres_uri."""
        monkeypatch.setenv("MEMORY_POSTGRES_URI", "postgresql://localhost:5432/test")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["postgres_uri"] == "postgresql://localhost:5432/test"

    def test_override_bank_id(self, monkeypatch):
        """MEMORY_BANK_ID should set memory.bank_id."""
        monkeypatch.setenv("MEMORY_BANK_ID", "my-agent")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["bank_id"] == "my-agent"

    def test_override_pg0_name(self, monkeypatch):
        """MEMORY_PG0_NAME should set memory.pg0_name."""
        monkeypatch.setenv("MEMORY_PG0_NAME", "custom_name")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["pg0_name"] == "custom_name"

    def test_override_pg0_port(self, monkeypatch):
        """MEMORY_PG0_PORT should set memory.pg0_port as int."""
        monkeypatch.setenv("MEMORY_PG0_PORT", "5433")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["pg0_port"] == 5433

    def test_override_default_enabled(self, monkeypatch):
        """MEMORY_DEFAULT_ENABLED should set memory.default_enabled as bool."""
        monkeypatch.setenv("MEMORY_DEFAULT_ENABLED", "true")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["default_enabled"] is True

    def test_override_reflection_module(self, monkeypatch):
        """MEMORY_DISTILLER_MODEL should set memory.distiller.model."""
        monkeypatch.setenv("MEMORY_DISTILLER_MODEL", "claude-3-sonnet")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["distiller"]["model"] == "claude-3-sonnet"

    def test_no_memory_prefix_ignored(self, monkeypatch):
        """Non-MEMORY_ env vars should be ignored."""
        monkeypatch.setenv("OTHER_POSTGRES_URI", "postgresql://localhost/db")
        config = {}
        result = apply_memory_env_overrides(config)
        assert "memory" not in result or "postgres_uri" not in result.get("memory", {})


class TestGetEpisodeStore:
    """Tests for get_episode_store function."""

    def test_returns_none_when_no_config(self):
        """Should return None when postgres_uri not set and pg0 not available."""
        settings = MagicMock()
        settings.memory.postgres_uri = None
        settings.memory.bank_id = "test-bank"
        settings.memory.pg0_name = "codespy"
        settings.memory.pg0_port = None

        # Simulate pg0 not being available
        with patch("codespy.config_memory._store", None):
            with patch("codespy.config_memory._store_built", False):
                with patch(
                    "codespy.agents.memory.pg0_manager.get_pg0_uri",
                    side_effect=ImportError("pg0 not installed"),
                ):
                    result = get_episode_store(settings)

        assert result is None

    def test_uses_postgres_uri_when_set(self):
        """Should use external PostgreSQL when MEMORY_POSTGRES_URI is set."""
        settings = MagicMock()
        settings.memory.postgres_uri = "postgresql://localhost:5432/codespy"
        settings.memory.bank_id = "test-bank"

        mock_store = MagicMock()

        with patch("codespy.config_memory._store", None):
            with patch("codespy.config_memory._store_built", False):
                with patch(
                    "codespy.agents.memory.postgres.EpisodeStore",
                    return_value=mock_store,
                ) as mock_episode_store:
                    result = get_episode_store(settings)

        mock_episode_store.assert_called_once_with(
            "postgresql://localhost:5432/codespy", "test-bank"
        )
        assert result == mock_store

    def test_caching_behavior(self):
        """Should cache the store after first call."""
        settings = MagicMock()
        settings.memory.postgres_uri = "postgresql://localhost:5432/codespy"
        settings.memory.bank_id = "test-bank"

        mock_store = MagicMock()

        with patch("codespy.config_memory._store", mock_store):
            with patch("codespy.config_memory._store_built", True):
                result = get_episode_store(settings)

        # Should return cached store without creating new one
        assert result == mock_store


class TestResetEpisodeStore:
    """Tests for reset_episode_store function."""

    def test_closes_existing_store(self):
        """Should close existing store and reset cache."""
        mock_store = MagicMock()

        with patch("codespy.config_memory._store", mock_store):
            with patch("codespy.config_memory._store_built", True):
                reset_episode_store()

        mock_store.close.assert_called_once()

    def test_handles_close_exception(self):
        """Should handle exceptions during close gracefully."""
        mock_store = MagicMock()
        mock_store.close.side_effect = Exception("Close failed")

        with patch("codespy.config_memory._store", mock_store):
            with patch("codespy.config_memory._store_built", True):
                # Should not raise
                reset_episode_store()

        mock_store.close.assert_called_once()


class TestVerifyMemoryAccess:
    """Tests for verify_memory_access function."""

    def test_verify_memory_access_all_disabled(self):
        """When all signatures disabled, returns success with skip message."""
        settings = MagicMock()
        settings.is_signature_enabled.return_value = False
        settings.get_memory_enabled.return_value = False

        success, message = verify_memory_access(settings)

        assert success is True
        assert "Memory disabled" in message

    def test_verify_memory_access_enabled_sig_disabled(self):
        """When signature has memory enabled but signature itself is disabled."""
        settings = MagicMock()

        def is_enabled(sig):
            return False  # All signatures disabled

        def memory_enabled(sig):
            return True  # But memory is configured

        settings.is_signature_enabled.side_effect = is_enabled
        settings.get_memory_enabled.side_effect = memory_enabled

        success, message = verify_memory_access(settings)

        assert success is True
        assert "Memory disabled" in message

    def test_verify_memory_access_store_none(self):
        """When memory active but store is None (no PostgreSQL configured)."""
        settings = MagicMock()

        def is_enabled(sig):
            return sig == "summary"  # Only summary enabled

        def memory_enabled(sig):
            return sig == "summary"  # Memory enabled for summary

        settings.is_signature_enabled.side_effect = is_enabled
        settings.get_memory_enabled.side_effect = memory_enabled

        with patch("codespy.config_memory.get_episode_store", return_value=None):
            success, message = verify_memory_access(settings)

        assert success is False
        assert "not configured" in message

    def test_verify_memory_access_postgres_ok(self):
        """When memory active with valid PostgreSQL store."""
        settings = MagicMock()
        settings.memory.bank_id = "test-bank"

        def is_enabled(sig):
            return sig == "summary"

        def memory_enabled(sig):
            return sig == "summary"

        settings.is_signature_enabled.side_effect = is_enabled
        settings.get_memory_enabled.side_effect = memory_enabled

        # Mock EpisodeStore that verifies successfully
        mock_store = MagicMock()
        mock_store.verify_access.return_value = None

        with patch("codespy.config_memory.get_episode_store", return_value=mock_store):
            success, message = verify_memory_access(settings)

        assert success is True
        assert "verified" in message
        assert "PostgreSQL" in message
        mock_store.verify_access.assert_called_once()

    def test_verify_memory_access_raises(self):
        """When store's verify_access raises an exception."""
        settings = MagicMock()
        settings.memory.bank_id = "test-bank"

        def is_enabled(sig):
            return sig == "summary"

        def memory_enabled(sig):
            return sig == "summary"

        settings.is_signature_enabled.side_effect = is_enabled
        settings.get_memory_enabled.side_effect = memory_enabled

        # Mock store that raises on verify_access
        mock_store = MagicMock()
        mock_store.verify_access.side_effect = ConnectionError("Cannot connect")

        with patch("codespy.config_memory.get_episode_store", return_value=mock_store):
            success, message = verify_memory_access(settings)

        assert success is False
        assert "not accessible" in message
        assert "Cannot connect" in message
