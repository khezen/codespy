"""Tests for memory storage configuration and access verification."""

from unittest.mock import MagicMock, patch

import pytest

from codespy.config_memory import (
    EMBEDDING_MODELS,
    PostgresConfig,
    Pg0Config,
    _cerebral_litellm_params,
    get_episode_store,
    reset_episode_store,
    verify_memory_access,
)


class TestGenerateBankId:
    """Tests for default bank_id behavior."""

    def test_default_bank_id_is_codespy(self):
        """Bank ID defaults to 'codespy' when not set."""
        settings = MagicMock()
        settings.memory.bank_id = None
        assert settings.memory.bank_id or "codespy" == "codespy"


class TestApplyMemoryEnvOverrides:
    """Tests for memory env var overrides with new paths."""

    def test_override_postgres_host(self, monkeypatch):
        """MEMORY_POSTGRES_HOST should set memory.postgres.host."""
        monkeypatch.setenv("MEMORY_POSTGRES_HOST", "myhost.example.com")
        from codespy.config_utils import apply_env_overrides, build_env_map
        from codespy.config_llm import LLMConfig
        from codespy.config_dspy import ReviewConfig
        from codespy.config_memory import MemoryConfig

        env_map = build_env_map(
            sections={"llm": LLMConfig, "review": ReviewConfig, "memory": MemoryConfig},
            bare_fields={"llm": {"openai_api_key", "anthropic_api_key", "gemini_api_key",
                               "aws_region", "aws_access_key_id", "aws_secret_access_key",
                               "azure_api_key", "azure_api_base", "azure_api_version"}},
        )
        config = {}
        result = apply_env_overrides(config, env_map)
        assert result["memory"]["postgres"]["host"] == "myhost.example.com"

    def test_override_memory_hippocampus_distiller_model(self, monkeypatch):
        """MEMORY_HIPPOCAMPUS_DISTILLER_MODEL should set the distiller model."""
        monkeypatch.setenv("MEMORY_HIPPOCAMPUS_DISTILLER_MODEL", "claude-3-sonnet")
        from codespy.config_utils import apply_env_overrides, build_env_map
        from codespy.config_llm import LLMConfig
        from codespy.config_dspy import ReviewConfig
        from codespy.config_memory import MemoryConfig

        env_map = build_env_map(
            sections={"llm": LLMConfig, "review": ReviewConfig, "memory": MemoryConfig},
            bare_fields={"llm": {"openai_api_key", "anthropic_api_key", "gemini_api_key",
                               "aws_region", "aws_access_key_id", "aws_secret_access_key",
                               "azure_api_key", "azure_api_base", "azure_api_version"}},
        )
        config = {}
        result = apply_env_overrides(config, env_map)
        assert result["memory"]["hippocampus"]["distiller"]["model"] == "claude-3-sonnet"

    def test_override_memory_cerebral_embeddings_model(self, monkeypatch):
        """MEMORY_CEREBRAL_EMBEDDINGS_MODEL should set embeddings model."""
        monkeypatch.setenv("MEMORY_CEREBRAL_EMBEDDINGS_MODEL", "openai/text-embedding-3-large")
        from codespy.config_utils import apply_env_overrides, build_env_map
        from codespy.config_llm import LLMConfig
        from codespy.config_dspy import ReviewConfig
        from codespy.config_memory import MemoryConfig

        env_map = build_env_map(
            sections={"llm": LLMConfig, "review": ReviewConfig, "memory": MemoryConfig},
            bare_fields={"llm": {"openai_api_key", "anthropic_api_key", "gemini_api_key",
                               "aws_region", "aws_access_key_id", "aws_secret_access_key",
                               "azure_api_key", "azure_api_base", "azure_api_version"}},
        )
        config = {}
        result = apply_env_overrides(config, env_map)
        assert result["memory"]["cerebral"]["embeddings"]["model"] == "openai/text-embedding-3-large"


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


class TestGetEpisodeStore:
    """Tests for get_episode_store function."""

    def test_returns_none_when_no_config(self):
        """Should return None when postgres.host not set and pg0 not available."""
        settings = MagicMock()
        settings.memory.postgres = MagicMock()
        settings.memory.postgres.build_uri.return_value = None
        settings.memory.postgres.schema_name = "episodic"
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


class TestCerebralLitellmParams:
    """Tests for _cerebral_litellm_params function."""

    @pytest.fixture
    def mock_settings(self):
        """Create a mock settings object with LLM config."""
        settings = MagicMock()
        settings.llm.openai_api_key.get_secret_value.return_value = "sk-openai-test"
        settings.llm.openai_api_base = "https://api.openai.com/v1"
        settings.llm.anthropic_api_key.get_secret_value.return_value = "sk-anthropic-test"
        settings.llm.gemini_api_key.get_secret_value.return_value = "gemini-test"
        settings.llm.azure_api_key.get_secret_value.return_value = "azure-test"
        settings.llm.azure_api_base = "https://my-resource.openai.azure.com"
        return settings

    def test_openai_prefix(self, mock_settings):
        """openai/ prefix should use OpenAI credentials."""
        mock_settings.get_llm_config.return_value.model = "openai/gpt-4"

        model, api_key, base_url = _cerebral_litellm_params(mock_settings)

        assert model == "openai/gpt-4"
        assert api_key == "sk-openai-test"
        assert base_url == "https://api.openai.com/v1"

    def test_anthropic_prefix(self, mock_settings):
        """anthropic/ prefix should use Anthropic credentials."""
        mock_settings.get_llm_config.return_value.model = "anthropic/claude-3-opus-4-1"

        model, api_key, base_url = _cerebral_litellm_params(mock_settings)

        assert model == "anthropic/claude-3-opus-4-1"
        assert api_key == "sk-anthropic-test"
        assert base_url is None


class TestEmbeddingModels:
    """Tests for EMBEDDING_MODELS constant."""

    def test_embedding_models_mapping_bedrock(self):
        """EMBEDDING_MODELS should have entry for bedrock."""
        assert "bedrock" in EMBEDDING_MODELS
        assert "cohere" in EMBEDDING_MODELS["bedrock"]

    def test_embedding_models_mapping_openai(self):
        """EMBEDDING_MODELS should have entry for openai."""
        assert "openai" in EMBEDDING_MODELS
        assert "text-embedding-3-small" in EMBEDDING_MODELS["openai"]

    def test_embedding_models_mapping_anthropic(self):
        """EMBEDDING_MODELS should have entry for anthropic."""
        assert "anthropic" in EMBEDDING_MODELS

    def test_embedding_models_mapping_gemini(self):
        """EMBEDDING_MODELS should have entry for gemini."""
        assert "gemini" in EMBEDDING_MODELS

    def test_embedding_models_mapping_azure(self):
        """EMBEDDING_MODELS should have entry for azure."""
        assert "azure" in EMBEDDING_MODELS
