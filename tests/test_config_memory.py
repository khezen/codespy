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
    _cerebral_litellm_params,
    _hindsight_env_overlay,
    get_episode_store,
)


class TestGenerateBankId:
    """Tests for default bank_id behavior."""

    def test_default_bank_id_is_codebase(self):
        """Bank ID defaults to 'codebase' when not set."""
        from codespy.config_memory import DEFAULT_BANK_ID

        settings = MagicMock()
        settings.memory.bank_id = None
        assert DEFAULT_BANK_ID == "codebase"
        assert (settings.memory.bank_id or DEFAULT_BANK_ID) == "codebase"


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


_OVERLAY_ENVS = (
    "HINDSIGHT_API_LLM_TIMEOUT",
    "HINDSIGHT_API_LLM_MAX_RETRIES",
    "HINDSIGHT_API_REFLECT_LLM_TIMEOUT",
    "HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES",
    "HINDSIGHT_API_REFLECT_MAX_ITERATIONS",
    "HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_PROVIDER",
    "HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_MODEL",
    "HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_API_KEY",
    "HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_BASE_URL",
    "HINDSIGHT_API_LLM_SUPPORTS_MAX_ITEMS",
)


class TestHindsightEnvOverlay:
    """codespy settings reach Hindsight through a temporary HINDSIGHT_API_* env overlay."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for env in _OVERLAY_ENVS:
            monkeypatch.delenv(env, raising=False)

    def _settings(self, *, reflects=3, model="openai/gpt-4o", mm_model="openai/gpt-4o-mini"):
        from pydantic import SecretStr

        from codespy.config import Settings
        from codespy.config_llm import LLMConfig
        from codespy.config_memory import (
            CerebralConfig,
            CerebralConsolidationConfig,
            CerebralRetainConfig,
            MemoryConfig,
            MentalModelsConfig,
            PrefrontalConfig,
            RecallConfig,
        )

        return Settings(
            llm=LLMConfig(
                default_model=model,
                extraction_model=model,
                timeout=200,
                retries=4,
                openai_api_key=SecretStr("sk-openai"),
            ),
            memory=MemoryConfig(
                postgres=PostgresConfig(host="db"),
                cerebral=CerebralConfig(
                    retain=CerebralRetainConfig(model=model),
                    consolidation=CerebralConsolidationConfig(model=model),
                    mental_models=MentalModelsConfig(model=mm_model),
                ),
                prefrontal=PrefrontalConfig(reflects=reflects, recall=RecallConfig(model=model)),
            ),
        )

    def test_values_from_settings(self):
        overlay = _hindsight_env_overlay(self._settings(reflects=3))
        assert overlay["HINDSIGHT_API_LLM_TIMEOUT"] == "200.0"
        assert overlay["HINDSIGHT_API_LLM_MAX_RETRIES"] == "4"
        assert overlay["HINDSIGHT_API_REFLECT_LLM_TIMEOUT"] == "200.0"
        assert overlay["HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES"] == "4"
        # LOW budget halves the cap, so 2 x reflects gives exactly reflects iterations
        assert overlay["HINDSIGHT_API_REFLECT_MAX_ITERATIONS"] == "6"
        assert overlay["HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_PROVIDER"] == "litellm"
        assert overlay["HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_MODEL"] == "openai/gpt-4o-mini"
        assert overlay["HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_API_KEY"] == "sk-openai"
        assert overlay["HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_BASE_URL"] == ""
        assert "HINDSIGHT_API_LLM_SUPPORTS_MAX_ITEMS" not in overlay

    def test_reflects_zero_keeps_minimal_cap_and_no_refresh_llm(self):
        overlay = _hindsight_env_overlay(self._settings(reflects=0))
        assert overlay["HINDSIGHT_API_REFLECT_MAX_ITERATIONS"] == "2"
        assert not any("MENTAL_MODEL_REFRESH" in k for k in overlay)

    def test_operator_env_wins(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_LLM_TIMEOUT", "99")
        monkeypatch.setenv("HINDSIGHT_API_REFLECT_MAX_ITERATIONS", "10")
        overlay = _hindsight_env_overlay(self._settings())
        assert "HINDSIGHT_API_LLM_TIMEOUT" not in overlay
        assert "HINDSIGHT_API_REFLECT_MAX_ITERATIONS" not in overlay
        # Reflect inherits the operator's explicit global timeout inside Hindsight
        assert "HINDSIGHT_API_REFLECT_LLM_TIMEOUT" not in overlay
        assert overlay["HINDSIGHT_API_LLM_MAX_RETRIES"] == "4"

    def test_refresh_group_all_or_nothing(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_MODEL", "openai/other")
        overlay = _hindsight_env_overlay(self._settings())
        assert not any("MENTAL_MODEL_REFRESH" in k for k in overlay)

    def test_bedrock_disables_max_items(self):
        overlay = _hindsight_env_overlay(
            self._settings(model="bedrock/converse/nvidia.nemotron-super-3-120b")
        )
        assert overlay["HINDSIGHT_API_LLM_SUPPORTS_MAX_ITEMS"] == "false"

    def test_bedrock_refresh_model_alone_disables_max_items(self):
        overlay = _hindsight_env_overlay(
            self._settings(mm_model="bedrock/converse/nvidia.nemotron-super-3-120b")
        )
        assert overlay["HINDSIGHT_API_LLM_SUPPORTS_MAX_ITEMS"] == "false"

    def test_get_cerebral_builds_config_and_restores_env(self, monkeypatch):
        pytest.importorskip("hindsight_api")
        import hindsight_api.config as ha_cfg

        import codespy.config_memory as cm
        from codespy.agents.memory import cerebral as cerebral_module

        settings = self._settings(reflects=3)
        seen_env: dict[str, str | None] = {}

        def fake_cerebral(**kwargs):
            # MemoryEngine.__init__ runs here: the overlay must be visible
            seen_env.update({k: os.environ.get(k) for k in _OVERLAY_ENVS})
            return MagicMock()

        monkeypatch.setattr(cm, "_cerebral", None)
        monkeypatch.setattr(cm, "_cerebral_built", False)
        monkeypatch.setattr(cerebral_module, "Cerebral", fake_cerebral)
        try:
            assert cm.get_cerebral(settings) is not None
            assert seen_env["HINDSIGHT_API_LLM_TIMEOUT"] == "200.0"

            config = ha_cfg.get_config()
            assert config.llm_timeout == settings.llm.timeout
            assert config.llm_max_retries == settings.llm.retries
            assert config.reflect_llm_timeout == settings.llm.timeout
            assert config.reflect_max_iterations == 2 * settings.memory.prefrontal.reflects
            assert config.mental_model_refresh_llm_model == "openai/gpt-4o-mini"
            # Nothing is left behind for subprocesses
            assert not any(k in os.environ for k in _OVERLAY_ENVS)
        finally:
            monkeypatch.setattr(cm, "_cerebral", None)
            monkeypatch.setattr(cm, "_cerebral_built", False)
            ha_cfg.clear_config_cache()

    def test_reset_cerebral_clears_config_cache(self, monkeypatch):
        pytest.importorskip("hindsight_api")
        import hindsight_api.config as ha_cfg

        import codespy.config_memory as cm

        ha_cfg.get_config()
        assert ha_cfg._config_cache is not None
        monkeypatch.setattr(cm, "_cerebral", None)
        cm.reset_cerebral()
        assert ha_cfg._config_cache is None
