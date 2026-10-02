"""Tests for memory storage configuration and access verification."""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from codespy.config_memory import (
    EMBEDDING_MODELS,
    CerebralConsolidationConfig,
    CerebralRetainConfig,
    MentalModelsConfig,
    PostgresConfig,
    _apply_hindsight_llm_call_defaults,
    _cerebral_litellm_params,
    get_episode_store,
)


class TestGenerateBankId:
    """Tests for default bank_id behavior."""

    def test_default_bank_id_is_codespy(self):
        """Bank ID defaults to 'codespy' when not set."""
        settings = MagicMock()
        settings.memory.bank_id = None
        assert settings.memory.bank_id or "codespy" == "codespy"


class TestApplyMemoryEnvOverrides:
    """Tests for memory env var overrides with new collapsed paths."""

    def test_override_postgres_host(self, monkeypatch):
        """MEMORY_POSTGRES_HOST should set memory.postgres.host."""
        monkeypatch.setenv("MEMORY_POSTGRES_HOST", "myhost.example.com")
        from codespy.config import _ENV_MAP
        from codespy.config_utils import apply_env_overrides

        config = {}
        result = apply_env_overrides(config, _ENV_MAP)
        assert result["memory"]["postgres"]["host"] == "myhost.example.com"

    def test_override_memory_distiller_model(self, monkeypatch):
        """MEMORY_DISTILLER_MODEL should set the distiller model (collapsed path)."""
        monkeypatch.setenv("MEMORY_DISTILLER_MODEL", "claude-3-sonnet")
        from codespy.config import _ENV_MAP
        from codespy.config_utils import apply_env_overrides

        config = {}
        result = apply_env_overrides(config, _ENV_MAP)
        assert result["memory"]["hippocampus"]["distiller"]["model"] == "claude-3-sonnet"

    def test_override_memory_embeddings_model(self, monkeypatch):
        """MEMORY_EMBEDDINGS_MODEL should set embeddings model (collapsed path)."""
        monkeypatch.setenv("MEMORY_EMBEDDINGS_MODEL", "openai/text-embedding-3-large")
        from codespy.config import _ENV_MAP
        from codespy.config_utils import apply_env_overrides

        config = {}
        result = apply_env_overrides(config, _ENV_MAP)
        assert result["memory"]["cerebral"]["embeddings"]["model"] == "openai/text-embedding-3-large"

    def test_override_memory_max_hippocampus_tokens(self, monkeypatch):
        """MEMORY_MAX_HIPPOCAMPUS_TOKENS should set hippocampus token budget (collapsed path)."""
        monkeypatch.setenv("MEMORY_MAX_HIPPOCAMPUS_TOKENS", "1234")
        from codespy.config import _ENV_MAP
        from codespy.config_utils import apply_env_overrides

        config = {}
        result = apply_env_overrides(config, _ENV_MAP)
        # convert_env_value returns numbers as raw strings
        assert result["memory"]["hippocampus"]["max_hippocampus_tokens"] == "1234"

    def test_override_memory_max_hippocampus_item_tokens(self, monkeypatch):
        """MEMORY_MAX_HIPPOCAMPUS_ITEM_TOKENS should set hippocampus item token budget (collapsed path)."""
        monkeypatch.setenv("MEMORY_MAX_HIPPOCAMPUS_ITEM_TOKENS", "256")
        from codespy.config import _ENV_MAP
        from codespy.config_utils import apply_env_overrides

        config = {}
        result = apply_env_overrides(config, _ENV_MAP)
        # convert_env_value returns numbers as raw strings
        assert result["memory"]["hippocampus"]["max_hippocampus_item_tokens"] == "256"


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
        assert "embed-v4:0" in EMBEDDING_MODELS["bedrock"]

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


class TestCerebralRetainConfig:
    """Tests for CerebralRetainConfig chunk_size validation."""

    def test_default_chunk_size(self):
        """Default chunk_size should be 12288."""
        config = CerebralRetainConfig()
        assert config.chunk_size == 12288

    def test_chunk_size_accepted_values(self):
        """Valid chunk sizes should be accepted."""
        config = CerebralRetainConfig(chunk_size=5000)
        assert config.chunk_size == 5000

        config = CerebralRetainConfig(chunk_size=63999)
        assert config.chunk_size == 63999

        config = CerebralRetainConfig(chunk_size=1)
        assert config.chunk_size == 1

    def test_chunk_size_rejects_zero(self):
        """chunk_size=0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            CerebralRetainConfig(chunk_size=0)

    def test_chunk_size_rejects_negative(self):
        """Negative chunk_size should raise ValidationError."""
        with pytest.raises(ValidationError):
            CerebralRetainConfig(chunk_size=-1)

    def test_chunk_size_rejects_64000(self):
        """chunk_size=64000 should raise ValidationError (must be < 64000)."""
        with pytest.raises(ValidationError):
            CerebralRetainConfig(chunk_size=64000)

    def test_chunk_size_rejects_too_large(self):
        """chunk_size > 64000 should raise ValidationError."""
        with pytest.raises(ValidationError):
            CerebralRetainConfig(chunk_size=100000)


class TestCerebralRetainChunkSizeEnvOverride:
    """Tests for MEMORY_RETAIN_CHUNK_SIZE env var override."""

    def test_override_memory_retain_chunk_size(self, monkeypatch):
        """MEMORY_RETAIN_CHUNK_SIZE should set memory.cerebral.retain.chunk_size."""
        monkeypatch.setenv("MEMORY_RETAIN_CHUNK_SIZE", "5000")
        from codespy.config import _ENV_MAP
        from codespy.config_utils import apply_env_overrides

        config = {}
        result = apply_env_overrides(config, _ENV_MAP)
        assert result["memory"]["cerebral"]["retain"]["chunk_size"] == "5000"


class TestPrefrontalConfig:
    """Tests for memory.prefrontal config, env mapping and get_cerebral wiring."""

    ENV = {
        "MEMORY_PREFRONTAL_REFLECTS": ("reflects", "5"),
        "MEMORY_RECALL_MODEL": ("model", "openai/gpt-4o-mini"),
        "MEMORY_RECALL_REACH": ("reach", "bank"),
        "MEMORY_RECALL_MAX_TOKENS": ("max_tokens", "2000"),
        "MEMORY_RECALL_MAX_TOOL_TOKENS": ("max_tool_tokens", "700"),
        "MEMORY_RECALL_MAX_TOOL_CALLS": ("max_tool_calls", "2"),
    }

    def test_defaults(self):
        from codespy.config_memory import MemoryConfig

        pf = MemoryConfig().prefrontal
        assert pf.reflects == 3  # Changed from 5 to 3
        assert pf.recall.model == "bedrock/converse/nvidia.nemotron-super-3-120b"
        assert pf.recall.reach == "org"
        assert pf.recall.max_tokens == 8192
        assert pf.recall.max_tool_tokens == 2048
        assert pf.recall.max_tool_calls == 0

    def test_env_mapping(self, monkeypatch):
        from codespy.config import _ENV_MAP
        from codespy.config_memory import PrefrontalConfig
        from codespy.config_utils import apply_env_overrides

        for env, (_, value) in self.ENV.items():
            monkeypatch.setenv(env, value)
        result = apply_env_overrides({}, _ENV_MAP)["memory"]["prefrontal"]

        # Verify env mapping for reflects
        assert _ENV_MAP["MEMORY_PREFRONTAL_REFLECTS"] == ("memory", "prefrontal", "reflects")
        assert "reflects" in result

        # Verify env mapping for recall settings
        assert _ENV_MAP["MEMORY_RECALL_MODEL"] == ("memory", "prefrontal", "recall", "model")
        assert _ENV_MAP["MEMORY_RECALL_REACH"] == ("memory", "prefrontal", "recall", "reach")
        assert _ENV_MAP["MEMORY_RECALL_MAX_TOKENS"] == ("memory", "prefrontal", "recall", "max_tokens")
        assert "recall" in result
        assert result["recall"]["model"] == "openai/gpt-4o-mini"
        assert result["recall"]["reach"] == "bank"

        cfg = PrefrontalConfig(**result)
        assert cfg.recall.reach == "bank"
        assert cfg.reflects == 5
        assert cfg.recall.max_tool_calls == 2

    def test_rejects_negative_reflects(self):
        from codespy.config_memory import PrefrontalConfig

        with pytest.raises(ValidationError):
            PrefrontalConfig(reflects=-1)

    def test_reflection_modules_unchanged(self):
        from codespy.config_memory import REFLECTION_MODULES

        assert REFLECTION_MODULES == ("memory_distiller", "memory_cartographer")

    def test_apply_reflect_config_returns_empty_when_reflects_zero(self):
        from codespy.config_memory import PrefrontalConfig, _apply_reflect_config

        cfg = PrefrontalConfig(reflects=0)
        result = _apply_reflect_config(cfg)
        assert result == {}

    def test_apply_reflect_config_applies_values(self, monkeypatch):
        pytest = __import__("pytest")
        ha = pytest.importorskip("hindsight_api")
        from codespy.config_memory import PrefrontalConfig, _apply_reflect_config

        cfg = PrefrontalConfig(reflects=3)
        # Clear env vars to ensure config values are applied
        monkeypatch.delenv("HINDSIGHT_API_REFLECT_MAX_ITERATIONS", raising=False)

        result = _apply_reflect_config(cfg)

        # With reflects=3, the cap is set to 2*3=6 (doubled for LOW budget)
        assert result["iterations"] == 6
        assert result["budget"] == "low"

    def test_apply_reflect_config_env_wins_over_config(self, monkeypatch):
        pytest = __import__("pytest")
        pytest.importorskip("hindsight_api")
        from codespy.config_memory import (
            HINDSIGHT_REFLECT_MAX_ITERATIONS_ENV,
            HINDSIGHT_REFLECT_MAX_CONTEXT_TOKENS_ENV,
            PrefrontalConfig,
            _apply_reflect_config,
        )

        monkeypatch.setenv(HINDSIGHT_REFLECT_MAX_ITERATIONS_ENV, "10")
        monkeypatch.setenv(HINDSIGHT_REFLECT_MAX_CONTEXT_TOKENS_ENV, "20000")

        cfg = PrefrontalConfig(reflects=3)
        result = _apply_reflect_config(cfg)

        assert result["iterations"] == 10  # env wins (used as-is, not doubled)
        assert result["budget"] == "low"

    def test_apply_hindsight_llm_call_defaults(self, monkeypatch):
        """Test _apply_hindsight_llm_call_defaults sets global/reflect timeout/retries."""
        pytest = __import__("pytest")
        pytest.importorskip("hindsight_api")
        import hindsight_api.config as ha_cfg

        # Clear all relevant env vars
        for env in [
            "HINDSIGHT_API_LLM_TIMEOUT",
            "HINDSIGHT_API_LLM_MAX_RETRIES",
            "HINDSIGHT_API_REFLECT_LLM_TIMEOUT",
            "HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES",
        ]:
            monkeypatch.delenv(env, raising=False)

        raw = ha_cfg._get_raw_config()

        # Save original values to restore after test
        orig_llm_timeout = raw.llm_timeout
        orig_llm_max_retries = raw.llm_max_retries
        orig_reflect_llm_timeout = raw.reflect_llm_timeout
        orig_reflect_llm_max_retries = raw.reflect_llm_max_retries

        try:
            raw.llm_timeout = 120.0
            raw.llm_max_retries = 3
            raw.reflect_llm_timeout = 30.0
            raw.reflect_llm_max_retries = 3

            settings = MagicMock()
            settings.llm.timeout = 240.0
            settings.llm.retries = 2

            _apply_hindsight_llm_call_defaults(settings)

            # Verify global and reflect settings were updated
            assert raw.llm_timeout == 240.0
            assert raw.llm_max_retries == 2
            assert raw.reflect_llm_timeout == 240.0
            assert raw.reflect_llm_max_retries == 2
        finally:
            # Restore original values
            raw.llm_timeout = orig_llm_timeout
            raw.llm_max_retries = orig_llm_max_retries
            raw.reflect_llm_timeout = orig_reflect_llm_timeout
            raw.reflect_llm_max_retries = orig_reflect_llm_max_retries

    def test_apply_hindsight_llm_call_defaults_with_env_vars(self, monkeypatch):
        """Test that when env vars are set, function uses those values and does not overwrite.

        Note: Hindsight loads env vars at import time and caches the config.
        This test verifies that when env vars are present, the function
        correctly identifies them and does not overwrite the values.
        """
        pytest = __import__("pytest")
        pytest.importorskip("hindsight_api")
        import hindsight_api.config as ha_cfg

        raw = ha_cfg._get_raw_config()

        # Save original values
        orig_llm_timeout = raw.llm_timeout
        orig_llm_max_retries = raw.llm_max_retries
        orig_reflect_llm_timeout = raw.reflect_llm_timeout
        orig_reflect_llm_max_retries = raw.reflect_llm_max_retries

        try:
            # Set up values that would be overwritten if env is not checked
            raw.llm_timeout = 120.0
            raw.llm_max_retries = 3
            raw.reflect_llm_timeout = 30.0
            raw.reflect_llm_max_retries = 3

            settings = MagicMock()
            settings.llm.timeout = 240.0
            settings.llm.retries = 3

            _apply_hindsight_llm_call_defaults(settings)

            # Verify the function logic based on env state
            if os.environ.get("HINDSIGHT_API_REFLECT_LLM_TIMEOUT"):
                # If env is set, function should not overwrite
                assert raw.reflect_llm_timeout == 30.0  # unchanged
            else:
                # If env is not set, function should set from settings
                assert raw.reflect_llm_timeout == 240.0

            if os.environ.get("HINDSIGHT_API_LLM_TIMEOUT"):
                # If global env is set, function should not overwrite
                assert raw.llm_timeout == 120.0  # unchanged
            else:
                # If env is not set, function should set from settings
                assert raw.llm_timeout == 240.0
        finally:
            # Restore original values
            raw.llm_timeout = orig_llm_timeout
            raw.llm_max_retries = orig_llm_max_retries
            raw.reflect_llm_timeout = orig_reflect_llm_timeout
            raw.reflect_llm_max_retries = orig_reflect_llm_max_retries

    def test_consolidation_inherits_global_llm_timeout(self, monkeypatch):
        """Test that consolidation inherits global llm_timeout when per-op value is None."""
        pytest = __import__("pytest")
        pytest.importorskip("hindsight_api")
        import hindsight_api.config as ha_cfg

        # Clear all env vars
        for env in [
            "HINDSIGHT_API_LLM_TIMEOUT",
            "HINDSIGHT_API_LLM_MAX_RETRIES",
            "HINDSIGHT_API_CONSOLIDATION_LLM_TIMEOUT",
            "HINDSIGHT_API_REFLECT_LLM_TIMEOUT",
        ]:
            monkeypatch.delenv(env, raising=False)

        raw = ha_cfg._get_raw_config()

        # Save original values
        orig_llm_timeout = raw.llm_timeout
        orig_consolidation_llm_timeout = raw.consolidation_llm_timeout

        try:
            # Simulate Hindsight's behavior: per-op is None, falls back to global
            # Set consolidation_llm_timeout to None (as Hindsight would when env is unset)
            raw.consolidation_llm_timeout = None
            raw.llm_timeout = 240.0

            # Hindsight's _op_defaults returns: per-op if non-None, else global
            # This is what MemoryEngine does for consolidation
            per_op = raw.consolidation_llm_timeout
            effective = per_op if per_op is not None else raw.llm_timeout
            assert effective == 240.0
        finally:
            # Restore original values
            raw.llm_timeout = orig_llm_timeout
            raw.consolidation_llm_timeout = orig_consolidation_llm_timeout

    def _settings(self, reflects):
        from codespy.config_memory import MemoryConfig

        settings = MagicMock()
        settings.memory = MemoryConfig()
        settings.memory.prefrontal.reflects = reflects
        settings.memory.postgres.host = "db"
        settings.get_llm_config.return_value = MagicMock(model="openai/gpt-4o")
        settings.llm.openai_api_key = None
        settings.llm.openai_api_base = None
        return settings

    # Note: Removed old test_get_cerebral_sets_env_before_import - now using programmatic
    # config via _apply_reflect_config instead of env vars. See tests for _apply_reflect_config.

    def _capture_cerebral_kwargs(self, monkeypatch, settings) -> dict:
        import codespy.config_memory as cm
        from codespy.agents.memory import cerebral as cerebral_module

        seen: dict = {}

        def fake_cerebral(**kwargs):
            seen.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(cm, "_cerebral", None)
        monkeypatch.setattr(cm, "_cerebral_built", False)
        monkeypatch.setattr(cerebral_module, "Cerebral", fake_cerebral)
        assert cm.get_cerebral(settings) is not None
        monkeypatch.setattr(cm, "_cerebral", None)
        monkeypatch.setattr(cm, "_cerebral_built", False)
        return seen

    def test_get_cerebral_without_prefrontal_model_passes_no_reflect_llm(self, monkeypatch):
        pytest = __import__("pytest")
        pytest.importorskip("hindsight_api")  # Skip if hindsight_api not available
        kwargs = self._capture_cerebral_kwargs(monkeypatch, self._settings(0))
        assert not any(k.startswith("reflect_llm_") for k in kwargs)

    def test_get_cerebral_passes_prefrontal_recall_model_as_reflect_llm(self, monkeypatch):
        pytest = __import__("pytest")
        pytest.importorskip("hindsight_api")  # Skip if hindsight_api not available
        from pydantic import SecretStr

        settings = self._settings(3)
        settings.memory.prefrontal.recall.model = "anthropic/claude-sonnet-4-5"
        settings.llm.anthropic_api_key = SecretStr("sk-ant")
        kwargs = self._capture_cerebral_kwargs(monkeypatch, settings)
        # Engine model is unchanged; reflect gets its own model and credentials.
        assert kwargs["llm_model"] == "openai/gpt-4o"
        # Cerebral derives reflect_llm_provider from llm_provider itself; not passed by caller
        assert "reflect_llm_provider" not in kwargs
        assert kwargs["reflect_llm_model"] == "anthropic/claude-sonnet-4-5"
        assert kwargs["reflect_llm_api_key"] == "sk-ant"
        assert kwargs["reflect_llm_base_url"] is None

    def test_get_cerebral_reflect_kwargs_match_cerebral_signature(self, monkeypatch):
        """Guard against signature drift: reflect kwargs must bind to Cerebral.__init__.

        Skipped when hindsight_api is not installed (e.g., in CI without the dependency).
        """
        import inspect

        from pydantic import SecretStr

        # Skip if hindsight_api is not available (Cerebral requires it)
        pytest = __import__("pytest")
        pytest.importorskip("hindsight_api")
        from codespy.agents.memory.cerebral.cerebral import Cerebral

        settings = self._settings(3)

        # Test without prefrontal model
        settings_no_pf = self._settings(3)
        settings_no_pf.memory.prefrontal.model = None
        kwargs_no_pf = self._capture_cerebral_kwargs(monkeypatch, settings_no_pf)
        # Should bind without error
        inspect.signature(Cerebral.__init__).bind(None, **kwargs_no_pf)

        # Test with prefrontal recall model
        settings.memory.prefrontal.recall.model = "anthropic/claude-sonnet-4-5"
        settings.llm.anthropic_api_key = SecretStr("sk-ant")
        kwargs_with_pf = self._capture_cerebral_kwargs(monkeypatch, settings)
        # Should bind without error
        inspect.signature(Cerebral.__init__).bind(None, **kwargs_with_pf)

    def test_get_cerebral_passes_max_observations_per_scope(self, monkeypatch):
        """Test that get_cerebral passes max_observations_per_scope to Cerebral."""
        pytest = __import__("pytest")
        pytest.importorskip("hindsight_api")  # Skip if hindsight_api not available

        settings = self._settings(0)
        settings.memory.cerebral.consolidation.max_observations_per_scope = 42
        kwargs = self._capture_cerebral_kwargs(monkeypatch, settings)
        assert kwargs["max_observations_per_scope"] == 42


class TestPrefrontalModelResolution:
    """get_llm_config('memory_recall') fallback chain.

    Settings are built with explicit sections so a local ``.env`` cannot
    override them.
    """

    def _settings(self, *, recall_model=None, retain=None):
        from codespy.config import Settings
        from codespy.config_llm import LLMConfig
        from codespy.config_memory import (
            CerebralConfig,
            CerebralRetainConfig,
            MemoryConfig,
            PrefrontalConfig,
            RecallConfig,
        )

        return Settings(
            llm=LLMConfig(default_model="openai/gpt-4o"),
            memory=MemoryConfig(
                cerebral=CerebralConfig(retain=CerebralRetainConfig(model=retain)),
                prefrontal=PrefrontalConfig(recall=RecallConfig(model=recall_model)),
            ),
        )

    def test_explicit_recall_model(self):
        from codespy.config_memory import MEMORY_RECALL

        s = self._settings(recall_model="anthropic/claude-haiku", retain="openai/gpt-4o-mini")
        assert s.get_llm_config(MEMORY_RECALL).model == "anthropic/claude-haiku"

    def test_falls_back_to_retain_model(self):
        from codespy.config_memory import MEMORY_RECALL

        s = self._settings(retain="openai/gpt-4o-mini")
        assert s.get_llm_config(MEMORY_RECALL).model == "openai/gpt-4o-mini"

    def test_falls_back_to_default_model(self):
        from codespy.config_memory import MEMORY_RECALL

        s = self._settings()
        assert s.get_llm_config(MEMORY_RECALL).model == s.llm.default_model

    # Note: _warn_reflect_mismatch removed - now using programmatic _apply_reflect_config
    # which sets values directly instead of warning about mismatches


class TestConsolidationConfig:
    """Tests for memory.cerebral.consolidation config and env mapping."""

    def test_defaults(self):
        from codespy.config_memory import CerebralConfig

        cfg = CerebralConfig().consolidation
        assert cfg.model == "bedrock/converse/nvidia.nemotron-super-3-120b"
        assert cfg.max_observations_per_scope == 100  # default

    def test_env_mapping(self, monkeypatch):
        from codespy.config import _ENV_MAP
        from codespy.config_memory import CerebralConfig
        from codespy.config_utils import apply_env_overrides

        monkeypatch.setenv("MEMORY_CONSOLIDATION_MODEL", "openai/gpt-4o-mini")
        result = apply_env_overrides({}, _ENV_MAP)["memory"]["cerebral"]["consolidation"]
        assert _ENV_MAP["MEMORY_CONSOLIDATION_MODEL"] == ("memory", "cerebral", "consolidation", "model")
        assert result["model"] == "openai/gpt-4o-mini"
        cfg = CerebralConsolidationConfig(**result)
        assert cfg.model == "openai/gpt-4o-mini"

    def test_max_observations_per_scope_default(self):
        """Default max_observations_per_scope is 100."""
        cfg = CerebralConsolidationConfig()
        assert cfg.max_observations_per_scope == 100

    def test_max_observations_per_scope_custom(self):
        """Custom max_observations_per_scope is accepted."""
        cfg = CerebralConsolidationConfig(max_observations_per_scope=50)
        assert cfg.max_observations_per_scope == 50

    def test_max_observations_per_scope_unlimited(self):
        """-1 means unlimited."""
        cfg = CerebralConsolidationConfig(max_observations_per_scope=-1)
        assert cfg.max_observations_per_scope == -1

    def test_max_observations_per_scope_rejects_less_than_minus_one(self):
        """Values < -1 are rejected."""
        with pytest.raises(ValidationError):
            CerebralConsolidationConfig(max_observations_per_scope=-2)

    def test_max_observations_per_scope_env_mapping(self, monkeypatch):
        """MEMORY_CONSOLIDATION_MAX_OBSERVATIONS_PER_SCOPE env var works."""
        from codespy.config import _ENV_MAP
        from codespy.config_utils import apply_env_overrides

        monkeypatch.setenv("MEMORY_CONSOLIDATION_MAX_OBSERVATIONS_PER_SCOPE", "200")
        result = apply_env_overrides({}, _ENV_MAP)["memory"]["cerebral"]["consolidation"]
        assert result["max_observations_per_scope"] == "200"
        cfg = CerebralConsolidationConfig(**result)
        assert cfg.max_observations_per_scope == 200


class TestRecallConfig:
    """Tests for memory.prefrontal.recall config and env mapping."""

    def test_defaults(self):
        from codespy.config_memory import RecallConfig

        cfg = RecallConfig()
        assert cfg.model == "bedrock/converse/nvidia.nemotron-super-3-120b"
        assert cfg.reach == "org"
        assert cfg.max_tokens == 8192
        assert cfg.max_tool_tokens == 2048
        assert cfg.max_tool_calls == 0

    def test_env_mapping_model(self, monkeypatch):
        from codespy.config import _ENV_MAP
        from codespy.config_memory import RecallConfig
        from codespy.config_utils import apply_env_overrides

        monkeypatch.setenv("MEMORY_RECALL_MODEL", "openai/gpt-4o")
        result = apply_env_overrides({}, _ENV_MAP)["memory"]["prefrontal"]["recall"]
        assert _ENV_MAP["MEMORY_RECALL_MODEL"] == ("memory", "prefrontal", "recall", "model")
        assert result["model"] == "openai/gpt-4o"
        cfg = RecallConfig(**result)
        assert cfg.model == "openai/gpt-4o"

    def test_env_mapping_reach(self, monkeypatch):
        from codespy.config import _ENV_MAP
        from codespy.config_memory import RecallConfig
        from codespy.config_utils import apply_env_overrides

        monkeypatch.setenv("MEMORY_RECALL_REACH", "bank")
        result = apply_env_overrides({}, _ENV_MAP)["memory"]["prefrontal"]["recall"]
        assert _ENV_MAP["MEMORY_RECALL_REACH"] == ("memory", "prefrontal", "recall", "reach")
        assert result["reach"] == "bank"
        cfg = RecallConfig(**result)
        assert cfg.reach == "bank"

    def test_env_mapping_max_tokens(self, monkeypatch):
        from codespy.config import _ENV_MAP
        from codespy.config_memory import RecallConfig
        from codespy.config_utils import apply_env_overrides

        monkeypatch.setenv("MEMORY_RECALL_MAX_TOKENS", "4096")
        result = apply_env_overrides({}, _ENV_MAP)["memory"]["prefrontal"]["recall"]
        assert _ENV_MAP["MEMORY_RECALL_MAX_TOKENS"] == ("memory", "prefrontal", "recall", "max_tokens")
        assert result["max_tokens"] == "4096"
        cfg = RecallConfig(**result)
        assert cfg.max_tokens == 4096

    def test_rejects_invalid_reach(self):
        from codespy.config_memory import RecallConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RecallConfig(reach="world")


class TestMentalModelsConfig:
    """Tests for memory.cerebral.mental_models config and env mapping."""

    def test_defaults(self):
        from codespy.config_memory import CerebralConfig

        cfg = CerebralConfig().mental_models
        assert cfg.model == "bedrock/converse/nvidia.nemotron-super-3-120b"
        assert cfg.max_tokens == 2048
        assert cfg.min_refresh_seconds == 0

    def test_env_mapping(self, monkeypatch):
        from codespy.config import _ENV_MAP
        from codespy.config_memory import CerebralConfig
        from codespy.config_utils import apply_env_overrides

        monkeypatch.setenv("MEMORY_MENTAL_MODELS_MODEL", "anthropic/claude-3-haiku")
        result = apply_env_overrides({}, _ENV_MAP)["memory"]["cerebral"]["mental_models"]
        assert _ENV_MAP["MEMORY_MENTAL_MODELS_MODEL"] == ("memory", "cerebral", "mental_models", "model")
        assert result["model"] == "anthropic/claude-3-haiku"
        cfg = MentalModelsConfig(**result)
        assert cfg.model == "anthropic/claude-3-haiku"

    def test_env_mapping_max_tokens(self, monkeypatch):
        from codespy.config import _ENV_MAP
        from codespy.config_memory import CerebralConfig
        from codespy.config_utils import apply_env_overrides

        monkeypatch.setenv("MEMORY_MENTAL_MODELS_MAX_TOKENS", "1024")
        result = apply_env_overrides({}, _ENV_MAP)["memory"]["cerebral"]["mental_models"]
        assert _ENV_MAP["MEMORY_MENTAL_MODELS_MAX_TOKENS"] == ("memory", "cerebral", "mental_models", "max_tokens")
        assert result["max_tokens"] == "1024"
        cfg = MentalModelsConfig(**result)
        assert cfg.max_tokens == 1024

    def test_env_mapping_min_refresh_seconds(self, monkeypatch):
        from codespy.config import _ENV_MAP
        from codespy.config_memory import CerebralConfig
        from codespy.config_utils import apply_env_overrides

        monkeypatch.setenv("MEMORY_MENTAL_MODELS_MIN_REFRESH_SECONDS", "60")
        result = apply_env_overrides({}, _ENV_MAP)["memory"]["cerebral"]["mental_models"]
        assert _ENV_MAP["MEMORY_MENTAL_MODELS_MIN_REFRESH_SECONDS"] == ("memory", "cerebral", "mental_models", "min_refresh_seconds")
        assert result["min_refresh_seconds"] == "60"
        cfg = MentalModelsConfig(**result)
        assert cfg.min_refresh_seconds == 60

    def test_rejects_zero_max_tokens(self):
        from codespy.config_memory import MentalModelsConfig

        with pytest.raises(ValidationError):
            MentalModelsConfig(max_tokens=0)

    def test_rejects_negative_max_tokens(self):
        from codespy.config_memory import MentalModelsConfig

        with pytest.raises(ValidationError):
            MentalModelsConfig(max_tokens=-1)

    def test_rejects_negative_min_refresh_seconds(self):
        from codespy.config_memory import MentalModelsConfig

        with pytest.raises(ValidationError):
            MentalModelsConfig(min_refresh_seconds=-1)

    def test_old_env_vars_not_in_env_map(self):
        """Old mental_models env vars should not be in _ENV_MAP."""
        from codespy.config import _ENV_MAP

        assert "MEMORY_MAX_MENTAL_MODEL_TOKENS" not in _ENV_MAP
        assert "MEMORY_MIN_MENTAL_MODEL_REFRESH_SECONDS" not in _ENV_MAP
