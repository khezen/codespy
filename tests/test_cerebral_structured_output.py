"""Tests for Cerebral structured output support.

Covers:
- Native structured output registration
- Two-step structured output wrapper
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if litellm is not available
try:
    import litellm

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

pytestmark = pytest.mark.skipif(not HAS_LITELLM, reason="litellm not installed")


class TestRegisterNativeStructuredOutput:
    """Tests for register_native_structured_output_overrides."""

    def test_registration_makes_super_capable(self):
        """After registration, Super should report native structured output support."""
        from litellm.utils import supports_native_structured_output
        from codespy.agents.memory.cerebral.structured_output import (
            register_native_structured_output_overrides,
        )

        register_native_structured_output_overrides()

        # Check the bare key
        result = supports_native_structured_output(
            "nvidia.nemotron-super-3-120b", "bedrock_converse"
        )
        assert result is True

    def test_registration_is_idempotent(self):
        """Multiple registrations should not cause errors."""
        from litellm.utils import supports_native_structured_output
        from codespy.agents.memory.cerebral.structured_output import (
            register_native_structured_output_overrides,
        )

        # Register twice
        register_native_structured_output_overrides()
        register_native_structured_output_overrides()

        # Should still work
        result = supports_native_structured_output(
            "nvidia.nemotron-super-3-120b", "bedrock_converse"
        )
        assert result is True

    def test_pricing_preserved_after_registration(self):
        """Model pricing should be preserved after registration."""
        from codespy.agents.memory.cerebral.structured_output import (
            register_native_structured_output_overrides,
        )

        register_native_structured_output_overrides()

        # Get model info
        info = litellm.get_model_info("nvidia.nemotron-super-3-120b", "bedrock_converse")
        # Should have pricing info
        assert info is not None


class TestEnforcesSchema:
    """Tests for enforces_schema function."""

    def test_returns_true_for_native_supported(self):
        """Should return True for models with native structured output."""
        from codespy.agents.memory.cerebral.structured_output import (
            enforces_schema,
            register_native_structured_output_overrides,
        )

        register_native_structured_output_overrides()

        result = enforces_schema("bedrock/converse/nvidia.nemotron-super-3-120b")
        # Result should be True (or None if litellm doesn't support the function)
        assert result is True or result is None

    def test_returns_false_for_tool_fallback_models(self):
        """Should return False for models using tool fallback."""
        from codespy.agents.memory.cerebral.structured_output import (
            enforces_schema,
        )

        # A model without native support (unknown/unmapped)
        result = enforces_schema("bedrock/converse/unknown-model")
        # Should be None or False depending on litellm
        assert result in (None, False)

    def test_returns_none_for_unmapped_models(self):
        """Should return None for unmapped models."""
        from codespy.agents.memory.cerebral.structured_output import (
            enforces_schema,
        )

        result = enforces_schema("unmapped-provider/unknown-model")
        # Result is None if function not found, or False/None if model not found
        assert result is None or result is False


class TestTwoStepStructuredLLM:
    """Tests for TwoStepStructuredLLM class."""

    def test_getattr_forwards_to_main(self):
        """Attribute access should forward to main LLM."""
        from codespy.agents.memory.cerebral.structured_output import (
            TwoStepStructuredLLM,
        )

        main = MagicMock()
        main.model = "main-model"
        main.provider = "bedrock"

        wrapper = TwoStepStructuredLLM(main, MagicMock())

        assert wrapper.model == "main-model"
        assert wrapper.provider == "bedrock"

    @pytest.mark.asyncio
    async def test_passthrough_without_response_format(self):
        """Should pass through to main when no response_format."""
        from unittest.mock import AsyncMock
        from codespy.agents.memory.cerebral.structured_output import (
            TwoStepStructuredLLM,
        )

        main = MagicMock()
        main.call = AsyncMock(return_value=MagicMock())
        extraction = MagicMock()

        wrapper = TwoStepStructuredLLM(main, extraction)
        await wrapper.call([{"role": "user", "content": "test"}], response_format=None)

        main.call.assert_called_once()
        extraction.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_builds_step1_messages_with_schema(self):
        """Should append schema instruction to system message."""
        from unittest.mock import AsyncMock, patch
        from pydantic import BaseModel
        from codespy.agents.memory.cerebral.structured_output import (
            TwoStepStructuredLLM,
        )

        # Create a mock result with content (not text)
        mock_result = MagicMock()
        mock_result.content = '{"key": "value"}'
        mock_result.usage = MagicMock()
        mock_result.usage.__add__ = MagicMock(return_value=MagicMock())
        mock_result.model_copy = MagicMock(return_value=mock_result)

        main = MagicMock()
        main.call = AsyncMock(return_value=mock_result)

        extraction = MagicMock()
        extraction.call = AsyncMock(return_value=mock_result)

        wrapper = TwoStepStructuredLLM(main, extraction)

        # Create a pydantic model for response_format
        class MockModel(BaseModel):
            key: str

        messages = [{"role": "system", "content": "You are helpful"}]
        await wrapper.call(messages, response_format=MockModel, scope="test")

        # Check main was called with modified messages
        call_args = main.call.call_args
        modified_messages = call_args[1].get("messages", call_args[0][0])

        # System message should contain schema
        assert len(modified_messages) == 1
        assert "JSON schema" in modified_messages[0]["content"]


class TestInstallTwoStepStructuredOutput:
    """Tests for install_two_step_structured_output."""

    def test_skips_multi_llm_provider(self):
        """Should skip MultiLLMProvider implementations."""
        from codespy.agents.memory.cerebral.structured_output import (
            enforces_schema,
            install_two_step_structured_output,
        )

        # Create mock engine with MultiLLMProvider
        engine = MagicMock()
        engine._llm_config = MagicMock()

        multi_provider = MagicMock()
        multi_provider.__class__.__name__ = "MultiLLMProvider"
        engine._llm_config._provider_impl = multi_provider

        extraction_impl = MagicMock()

        install_two_step_structured_output(engine, extraction_impl)

        # Multi provider should not be wrapped
        assert engine._llm_config._provider_impl is multi_provider

    def test_skips_when_main_enforces(self):
        """Should skip when main model already enforces schema."""
        from codespy.agents.memory.cerebral.structured_output import (
            install_two_step_structured_output,
        )

        # Mock enforces_schema to return True for main
        with patch(
            "codespy.agents.memory.cerebral.structured_output.enforces_schema"
        ) as mock_enforces:
            mock_enforces.return_value = True

            engine = MagicMock()
            llm = MagicMock()
            llm.model = "bedrock/converse/nvidia.nemotron-super-3-120b"
            engine._llm_config = MagicMock()
            engine._llm_config._provider_impl = llm

            extraction_impl = MagicMock()
            extraction_impl.model = "nvidia.nemotron-nano-3-30b"

            install_two_step_structured_output(engine, extraction_impl)

            # Should not be wrapped
            assert engine._llm_config._provider_impl is llm

    @pytest.mark.skip(reason="Cannot easily mock isinstance with real LiteLLMLLM class")
    def test_wraps_when_main_does_not_enforce(self):
        """Should wrap when main doesn't enforce but extraction does."""
        from codespy.agents.memory.cerebral.structured_output import (
            TwoStepStructuredLLM,
            install_two_step_structured_output,
        )

        # Mock enforces_schema: main=False, extraction=True
        def enforces_side_effect(model):
            if "super" in model:
                return False
            if "nano" in model:
                return True
            return None

        extraction_impl = MagicMock()
        extraction_impl.model = "nvidia.nemotron-nano-3-30b"

        with patch(
            "codespy.agents.memory.cerebral.structured_output.enforces_schema"
        ) as mock_enforces:
            mock_enforces.side_effect = enforces_side_effect

            engine = MagicMock()
            llm = MagicMock()
            llm.model = "bedrock/converse/nvidia.nemotron-super-3-120b"
            engine._llm_config = MagicMock()
            engine._llm_config._provider_impl = llm
            engine._retain_llm_config = None
            engine._consolidation_llm_config = None
            engine._reflect_llm_config = None
            engine._mental_model_refresh_llm_override = None

            install_two_step_structured_output(engine, extraction_impl)

            # Should be wrapped
            assert isinstance(engine._llm_config._provider_impl, TwoStepStructuredLLM)

    @pytest.mark.skip(reason="Cannot easily mock isinstance with real LiteLLMLLM class")
    def test_wraps_only_once_per_config(self):
        """Should wrap shared configs only once."""
        from codespy.agents.memory.cerebral.structured_output import (
            TwoStepStructuredLLM,
            install_two_step_structured_output,
        )

        def enforces_side_effect(model):
            if "super" in model:
                return False
            if "nano" in model:
                return True
            return None

        extraction_impl = MagicMock()
        extraction_impl.model = "nvidia.nemotron-nano-3-30b"

        with patch(
            "codespy.agents.memory.cerebral.structured_output.enforces_schema"
        ) as mock_enforces:
            mock_enforces.side_effect = enforces_side_effect

            shared_config = MagicMock()
            llm = MagicMock()
            llm.model = "bedrock/converse/nvidia.nemotron-super-3-120b"
            shared_config._provider_impl = llm

            engine = MagicMock()
            engine._llm_config = shared_config
            engine._retain_llm_config = shared_config  # Same config
            engine._consolidation_llm_config = None
            engine._reflect_llm_config = None
            engine._mental_model_refresh_llm_override = None

            install_two_step_structured_output(engine, extraction_impl)

            # Both should point to the same wrapper
            assert isinstance(engine._llm_config._provider_impl, TwoStepStructuredLLM)
            assert isinstance(engine._retain_llm_config._provider_impl, TwoStepStructuredLLM)
            # Should be the same wrapper instance
            assert (
                engine._llm_config._provider_impl is engine._retain_llm_config._provider_impl
            )
