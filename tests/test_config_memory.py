"""Tests for memory storage configuration and access verification."""

from unittest.mock import MagicMock, patch

import pytest

from codespy.config_memory import (
    PostgresConfig,
    Pg0Config,
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

    def test_override_postgres_host(self, monkeypatch):
        """MEMORY_POSTGRES_HOST should set memory.postgres.host."""
        monkeypatch.setenv("MEMORY_POSTGRES_HOST", "myhost.example.com")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["postgres"]["host"] == "myhost.example.com"

    def test_override_postgres_port(self, monkeypatch):
        """MEMORY_POSTGRES_PORT should set memory.postgres.port as int."""
        monkeypatch.setenv("MEMORY_POSTGRES_PORT", "5433")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["postgres"]["port"] == 5433

    def test_override_postgres_user(self, monkeypatch):
        """MEMORY_POSTGRES_USER should set memory.postgres.user."""
        monkeypatch.setenv("MEMORY_POSTGRES_USER", "admin")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["postgres"]["user"] == "admin"

    def test_override_postgres_password(self, monkeypatch):
        """MEMORY_POSTGRES_PASSWORD should set memory.postgres.password."""
        monkeypatch.setenv("MEMORY_POSTGRES_PASSWORD", "secret123")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["postgres"]["password"] == "secret123"

    def test_override_postgres_database(self, monkeypatch):
        """MEMORY_POSTGRES_DATABASE should set memory.postgres.database."""
        monkeypatch.setenv("MEMORY_POSTGRES_DATABASE", "mydb")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["postgres"]["database"] == "mydb"

    def test_override_postgres_schema(self, monkeypatch):
        """MEMORY_POSTGRES_SCHEMA should set memory.postgres.schema."""
        monkeypatch.setenv("MEMORY_POSTGRES_SCHEMA", "public")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["postgres"]["schema"] == "public"

    def test_override_bank_id(self, monkeypatch):
        """MEMORY_BANK_ID should set memory.bank_id."""
        monkeypatch.setenv("MEMORY_BANK_ID", "my-agent")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["bank_id"] == "my-agent"

    def test_override_pg0_name(self, monkeypatch):
        """MEMORY_PG0_NAME should set memory.pg0.name."""
        monkeypatch.setenv("MEMORY_PG0_NAME", "custom_name")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["pg0"]["name"] == "custom_name"

    def test_override_pg0_port(self, monkeypatch):
        """MEMORY_PG0_PORT should set memory.pg0.port as int."""
        monkeypatch.setenv("MEMORY_PG0_PORT", "5433")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["pg0"]["port"] == 5433

    def test_override_pg0_data_dir(self, monkeypatch):
        """MEMORY_PG0_DATA_DIR should set memory.pg0.data_dir."""
        monkeypatch.setenv("MEMORY_PG0_DATA_DIR", "/custom/path")
        config = {}
        result = apply_memory_env_overrides(config)
        assert result["memory"]["pg0"]["data_dir"] == "/custom/path"

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
        monkeypatch.setenv("OTHER_POSTGRES_HOST", "myhost.example.com")
        config = {}
        result = apply_memory_env_overrides(config)
        assert "memory" not in result or "postgres" not in result.get("memory", {})


class TestPostgresConfigBuildUri:
    """Tests for PostgresConfig.build_uri() method."""

    def test_build_uri_returns_none_when_no_host(self):
        """Should return None when host is unset."""
        config = PostgresConfig()
        assert config.build_uri() is None

    def test_build_uri_basic(self):
        """Should build basic URI with all fields."""
        config = PostgresConfig(
            host="db.example.com",
            user="u",
            password="p",
            database="codespy"
        )
        result = config.build_uri()
        assert result == "postgresql://u:p@db.example.com:5432/codespy"

    def test_build_uri_no_password(self):
        """Should not include password segment when password is None."""
        config = PostgresConfig(
            host="db.example.com",
            user="u"
        )
        result = config.build_uri()
        assert result == "postgresql://u@db.example.com:5432/codespy"

    def test_build_uri_no_user_no_password(self):
        """Should default to 'postgres' user when user is None."""
        config = PostgresConfig(host="db.example.com")
        result = config.build_uri()
        assert result == "postgresql://postgres@db.example.com:5432/codespy"

    def test_build_uri_special_chars_in_password(self):
        """Should URL-encode special characters in password."""
        config = PostgresConfig(
            host="db.example.com",
            user="u",
            password="p@ss:w/rd"
        )
        result = config.build_uri()
        assert "p%40ss%3Aw%2Frd" in result
        assert result == "postgresql://u:p%40ss%3Aw%2Frd@db.example.com:5432/codespy"

    def test_build_uri_with_schema(self):
        """Should append search_path option when schema is set."""
        config = PostgresConfig(
            host="db.example.com",
            schema="my_schema"
        )
        result = config.build_uri()
        assert result == "postgresql://postgres@db.example.com:5432/codespy?options=-csearch_path%3Dmy_schema"

    def test_build_uri_custom_port_and_database(self):
        """Should use custom port and database."""
        config = PostgresConfig(
            host="db.example.com",
            port=5433,
            database="mydb"
        )
        result = config.build_uri()
        assert result == "postgresql://postgres@db.example.com:5433/mydb"

    def test_build_uri_empty_password_treated_as_none(self):
        """Empty string password should be treated as no password."""
        config = PostgresConfig(
            host="db.example.com",
            user="u",
            password=""
        )
        result = config.build_uri()
        assert result == "postgresql://u@db.example.com:5432/codespy"


class TestGetEpisodeStore:
    """Tests for get_episode_store function."""

    def test_returns_none_when_no_config(self):
        """Should return None when postgres.host not set and pg0 not available."""
        settings = MagicMock()
        settings.memory.postgres = MagicMock()
        settings.memory.postgres.build_uri.return_value = None
        settings.memory.pg0 = MagicMock()
        settings.memory.pg0.name = "codespy"
        settings.memory.pg0.port = None
        settings.memory.pg0.data_dir = None
        settings.memory.bank_id = "test-bank"

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
        """Should use external PostgreSQL when postgres.build_uri() returns a URI."""
        settings = MagicMock()
        settings.memory.postgres = MagicMock()
        settings.memory.postgres.build_uri.return_value = "postgresql://u:p@host:5432/codespy"
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
            "postgresql://u:p@host:5432/codespy", "test-bank"
        )
        assert result == mock_store

    def test_caching_behavior(self):
        """Should cache the store after first call."""
        settings = MagicMock()
        settings.memory.postgres = MagicMock()
        settings.memory.postgres.build_uri.return_value = "postgresql://localhost:5432/codespy"
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
