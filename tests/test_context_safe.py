"""Tests for ContextSafe proactive RLM fallback thresholds."""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from codespy.agents.context_safe import ContextSafe
from codespy.config_dspy import (
    RLMFallbackConfig,
    apply_rlm_fallback_env_overrides,
)


class TestRLMFallbackConfig:
    """Tests for RLMFallbackConfig model."""

    def test_default_values(self):
        """Test default threshold values."""
        config = RLMFallbackConfig()
        assert config.enabled is True
        assert config.react_threshold == 0.30
        assert config.chain_of_thought_threshold == 0.40
        assert config.predict_threshold == 0.50

    def test_threshold_bounds(self):
        """Test that thresholds are bounded 0.0-1.0."""
        # Valid values
        config = RLMFallbackConfig(react_threshold=0.5)
        assert config.react_threshold == 0.5

        # Invalid: below 0
        with pytest.raises(ValidationError):
            RLMFallbackConfig(react_threshold=-0.1)

        # Invalid: above 1
        with pytest.raises(ValidationError):
            RLMFallbackConfig(react_threshold=1.1)


class TestApplyRLMFallbackEnvOverrides:
    """Tests for apply_rlm_fallback_env_overrides function."""

    def test_react_threshold_override(self):
        """Test RLM_FALLBACK_REACT_THRESHOLD env var."""
        config = {}
        with patch.dict(os.environ, {"RLM_FALLBACK_REACT_THRESHOLD": "0.25"}):
            result = apply_rlm_fallback_env_overrides(config)
        assert result["rlm_fallback"]["react_threshold"] == "0.25"

    def test_enabled_override(self):
        """Test RLM_FALLBACK_ENABLED env var."""
        config = {}
        with patch.dict(os.environ, {"RLM_FALLBACK_ENABLED": "false"}):
            result = apply_rlm_fallback_env_overrides(config)
        assert result["rlm_fallback"]["enabled"] is False

    def test_all_thresholds(self):
        """Test all threshold env vars."""
        config = {}
        env_vars = {
            "RLM_FALLBACK_REACT_THRESHOLD": "0.35",
            "RLM_FALLBACK_CHAIN_OF_THOUGHT_THRESHOLD": "0.45",
            "RLM_FALLBACK_PREDICT_THRESHOLD": "0.55",
        }
        with patch.dict(os.environ, env_vars):
            result = apply_rlm_fallback_env_overrides(config)
        assert result["rlm_fallback"]["react_threshold"] == "0.35"
        assert result["rlm_fallback"]["chain_of_thought_threshold"] == "0.45"
        assert result["rlm_fallback"]["predict_threshold"] == "0.55"

    def test_unrelated_env_vars_ignored(self):
        """Test that unrelated env vars are ignored."""
        config = {}
        with patch.dict(os.environ, {"OTHER_VAR": "value"}):
            result = apply_rlm_fallback_env_overrides(config)
        assert "rlm_fallback" not in result

    def test_existing_config_preserved(self):
        """Test that existing config is preserved."""
        config = {"rlm_fallback": {"enabled": False, "react_threshold": 0.20}}
        with patch.dict(os.environ, {"RLM_FALLBACK_REACT_THRESHOLD": "0.35"}):
            result = apply_rlm_fallback_env_overrides(config)
        assert result["rlm_fallback"]["enabled"] is False
        assert result["rlm_fallback"]["react_threshold"] == "0.35"


class MockSignature:
    """Mock signature class with __name__ attribute."""

    __name__ = "MockSignature"


class TestGetRLMThreshold:
    """Tests for Settings.get_rlm_threshold() method."""

    def test_returns_correct_threshold_per_type(self):
        """Test that correct threshold is returned for each module type."""
        from codespy.config import Settings

        settings = Settings()
        settings.rlm_fallback.enabled = True
        settings.rlm_fallback.react_threshold = 0.30
        settings.rlm_fallback.chain_of_thought_threshold = 0.40
        settings.rlm_fallback.predict_threshold = 0.50

        assert settings.get_rlm_threshold("react") == 0.30
        assert settings.get_rlm_threshold("chain_of_thought") == 0.40
        assert settings.get_rlm_threshold("predict") == 0.50

    def test_returns_1_0_when_disabled(self):
        """Test that 1.0 is returned when rlm_fallback is disabled."""
        from codespy.config import Settings

        settings = Settings()
        settings.rlm_fallback.enabled = False

        assert settings.get_rlm_threshold("react") == 1.0
        assert settings.get_rlm_threshold("chain_of_thought") == 1.0
        assert settings.get_rlm_threshold("predict") == 1.0

    def test_returns_1_0_for_unknown_type(self):
        """Test that 1.0 is returned for unknown module types."""
        from codespy.config import Settings

        settings = Settings()
        settings.rlm_fallback.enabled = True

        assert settings.get_rlm_threshold("unknown_type") == 1.0


class TestContextSafeShouldUseRLM:
    """Tests for ContextSafe._should_use_rlm() method."""

    def _create_mock_lm(self, model="anthropic/claude-opus-4-6", max_tokens=64000):
        """Create a mock LM for testing."""
        lm = MagicMock()
        lm.model = model
        lm.kwargs = {"max_tokens": max_tokens}
        return lm

    @patch("codespy.agents.context_safe.litellm")
    def test_proactive_threshold_triggered(self, mock_litellm):
        """Test that proactive threshold triggers when input exceeds threshold."""
        # Mock model info: 200k max input tokens
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 200000}
        # Mock token counter to return 70k tokens (above 30% of 200k = 60k, above 8192 floor)
        mock_litellm.token_counter.return_value = 70000

        # Create ContextSafe with 0.30 threshold
        module = MagicMock()
        cs = ContextSafe(module, MockSignature(), rlm_threshold=0.30)

        # Set up dspy.settings.lm
        with patch("codespy.agents.context_safe.dspy.settings") as mock_settings:
            mock_settings.lm = self._create_mock_lm()
            should_fallback, reason = cs._should_use_rlm({"input": "test"})

        assert should_fallback is True
        assert "context rot threshold exceeded" in reason
        assert "70000" in reason

    @patch("codespy.agents.context_safe.litellm")
    def test_proactive_threshold_not_triggered_below_threshold(self, mock_litellm):
        """Test that proactive threshold does not trigger below threshold."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 200000}
        # 50k tokens is below 30% of 200k (60k)
        mock_litellm.token_counter.return_value = 50000

        module = MagicMock()
        cs = ContextSafe(module, MockSignature(), rlm_threshold=0.30)

        with patch("codespy.agents.context_safe.dspy.settings") as mock_settings:
            mock_settings.lm = self._create_mock_lm()
            should_fallback, reason = cs._should_use_rlm({"input": "test"})

        assert should_fallback is False

    @patch("codespy.agents.context_safe.litellm")
    def test_proactive_threshold_not_triggered_below_floor(self, mock_litellm):
        """Test that proactive threshold does not trigger below 8192 floor."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 200000}
        # 4000 tokens is above 30% threshold (would be 6000) but below 8192 floor
        mock_litellm.token_counter.return_value = 4000

        module = MagicMock()
        cs = ContextSafe(module, MockSignature(), rlm_threshold=0.30)

        with patch("codespy.agents.context_safe.dspy.settings") as mock_settings:
            mock_settings.lm = self._create_mock_lm()
            should_fallback, reason = cs._should_use_rlm({"input": "test"})

        assert should_fallback is False

    @patch("codespy.agents.context_safe.litellm")
    def test_proactive_threshold_disabled_at_1_0(self, mock_litellm):
        """Test that proactive threshold is disabled when rlm_threshold=1.0."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 300000}
        # With 150k tokens, proactive would trigger at threshold < 0.75
        # but with threshold=1.0, proactive is disabled
        # Hard overflow: 150k + 64k + 4k = 218k < 300k, so no overflow either
        mock_litellm.token_counter.return_value = 150000

        module = MagicMock()
        cs = ContextSafe(module, MockSignature(), rlm_threshold=1.0)

        with patch("codespy.agents.context_safe.dspy.settings") as mock_settings:
            mock_settings.lm = self._create_mock_lm()
            should_fallback, reason = cs._should_use_rlm({"input": "test"})

        # Should not trigger proactive fallback (disabled at 1.0)
        # and should not trigger hard overflow (218k < 300k)
        assert should_fallback is False

    @patch("codespy.agents.context_safe.litellm")
    def test_hard_overflow_still_works(self, mock_litellm):
        """Test that hard overflow check still works with rlm_threshold=1.0."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 100000}
        # Input + max_tokens (64000) + safety_margin (4096) > max_input
        mock_litellm.token_counter.return_value = 50000

        module = MagicMock()
        cs = ContextSafe(module, MockSignature(), rlm_threshold=1.0)

        with patch("codespy.agents.context_safe.dspy.settings") as mock_settings:
            mock_settings.lm = self._create_mock_lm(max_tokens=64000)
            should_fallback, reason = cs._should_use_rlm({"input": "test"})

        # 50000 + 64000 + 4096 = 118096 > 100000
        assert should_fallback is True
        assert "context overflow predicted" in reason

    @patch("codespy.agents.context_safe.litellm")
    def test_model_not_in_litellm_db(self, mock_litellm):
        """Test behavior when model is not in litellm's database."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 0}

        module = MagicMock()
        cs = ContextSafe(module, MockSignature(), rlm_threshold=0.30)

        with patch("codespy.agents.context_safe.dspy.settings") as mock_settings:
            mock_settings.lm = self._create_mock_lm()
            should_fallback, reason = cs._should_use_rlm({"input": "test"})

        # Should skip proactive check when max_input_tokens is 0
        assert should_fallback is False

    def test_no_lm_configured(self):
        """Test behavior when no LM is configured."""
        module = MagicMock()
        cs = ContextSafe(module, MockSignature(), rlm_threshold=0.30)

        with patch("codespy.agents.context_safe.dspy.settings") as mock_settings:
            mock_settings.lm = None
            should_fallback, reason = cs._should_use_rlm({"input": "test"})

        assert should_fallback is False


class TestContextSafeThreeLayerDefense:
    """Tests demonstrating the three-layer defense strategy."""

    @patch("codespy.agents.context_safe.litellm")
    def test_layer1_proactive_triggers_first(self, mock_litellm):
        """Test that Layer 1 (proactive) triggers before Layer 2 (hard overflow)."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 200000}
        # At 70k tokens with 0.30 threshold: proactive triggers (70k > 60k)
        # But hard overflow: 70k + 64k + 4k = 138k < 200k, so no overflow
        mock_litellm.token_counter.return_value = 70000

        module = MagicMock()
        cs = ContextSafe(module, MockSignature(), rlm_threshold=0.30)

        with patch("codespy.agents.context_safe.dspy.settings") as mock_settings:
            mock_settings.lm = MagicMock()
            mock_settings.lm.model = "anthropic/claude-opus-4-6"
            mock_settings.lm.kwargs = {"max_tokens": 64000}
            should_fallback, reason = cs._should_use_rlm({"input": "test"})

        assert should_fallback is True
        # Should be proactive, not overflow
        assert "context rot threshold exceeded" in reason

    @patch("codespy.agents.context_safe.litellm")
    def test_layer2_hard_overflow_as_fallback(self, mock_litellm):
        """Test that Layer 2 (hard overflow) triggers when Layer 1 doesn't."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 80000}
        # At 20k tokens with 0.30 threshold: proactive doesn't trigger (20k < 24k threshold)
        # But hard overflow: 20k + 64k + 4k = 88k > 80k
        mock_litellm.token_counter.return_value = 20000

        module = MagicMock()
        cs = ContextSafe(module, MockSignature(), rlm_threshold=0.30)

        with patch("codespy.agents.context_safe.dspy.settings") as mock_settings:
            mock_settings.lm = MagicMock()
            mock_settings.lm.model = "anthropic/claude-opus-4-6"
            mock_settings.lm.kwargs = {"max_tokens": 64000}
            should_fallback, reason = cs._should_use_rlm({"input": "test"})

        # 20k + 64k + 4k = 88k > 80k
        assert should_fallback is True
        assert "context overflow predicted" in reason
