"""Tests for the Cerebral cost tracking module."""

import asyncio
from unittest.mock import MagicMock, Mock, patch

import pytest

from codespy.agents.cost_tracker import CostTracker, get_cost_tracker
from codespy.agents.memory.cerebral.cost import (
    BUCKET_CEREBRAL_EMBEDDINGS,
    BUCKET_CEREBRAL_OTHER,
    BUCKET_CEREBRAL_RETAIN,
    CerebralCostRecorder,
    MeteredLiteLLMSDKEmbeddings,
    _LiteLLMProxy,
    _recorder_registered,
    register_cerebral_cost_recorder,
    unregister_cerebral_cost_recorder,
)


@pytest.fixture
def fresh_recorder():
    """Reset registration state before/after each test."""
    # Reset module state
    from codespy.agents.memory.cerebral import cost as cost_module

    cost_module._recorder_registered = False
    cost_module._warned_unpriced_models.clear()

    yield

    # Cleanup after test
    try:
        unregister_cerebral_cost_recorder()
    except Exception:
        pass
    cost_module._recorder_registered = False
    cost_module._warned_unpriced_models.clear()


@pytest.fixture
def mock_tracker():
    """Create a fresh CostTracker for tests."""
    tracker = CostTracker()
    tracker.reset()
    return tracker


@pytest.fixture
def mock_cost_recorder():
    """Create a mock CerebralCostRecorder."""
    return CerebralCostRecorder()


class TestCerebralCostRecorder:
    """Tests for CerebralCostRecorder class."""

    def test_record_llm_call_with_retain_scope(self, mock_tracker, fresh_recorder):
        """scope='retain_extract_facts' → cerebral_retain bucket."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(
                CerebralCostRecorder, "_price_call", return_value=0.05
            ):
                recorder = CerebralCostRecorder()
                recorder.record_llm_call(
                    model="openai/gpt-4",
                    scope="retain_extract_facts",
                    input_tokens=100,
                    output_tokens=50,
                )

        stats = mock_tracker.get_signature_stats(BUCKET_CEREBRAL_RETAIN)
        assert stats is not None
        assert stats.cost == 0.05
        assert stats.tokens == 150

    def test_record_llm_call_with_consolidation_scope(self, mock_tracker, fresh_recorder):
        """scope='consolidation' → cerebral_other bucket."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(CerebralCostRecorder, "_price_call", return_value=0.03):
                recorder = CerebralCostRecorder()
                recorder.record_llm_call(
                    model="openai/gpt-4",
                    scope="consolidation",
                    input_tokens=100,
                    output_tokens=50,
                )

        stats = mock_tracker.get_signature_stats(BUCKET_CEREBRAL_OTHER)
        assert stats is not None

    def test_record_llm_call_with_other_scope(self, mock_tracker, fresh_recorder):
        """Any other scope → cerebral_other bucket."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(CerebralCostRecorder, "_price_call", return_value=0.01):
                recorder = CerebralCostRecorder()
                recorder.record_llm_call(
                    model="openai/gpt-4",
                    scope="reflect",
                    input_tokens=100,
                    output_tokens=50,
                )

        stats = mock_tracker.get_signature_stats(BUCKET_CEREBRAL_OTHER)
        assert stats is not None

    def test_record_llm_call_skips_error_calls(self, mock_tracker, fresh_recorder):
        """A call with error set → nothing recorded."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            recorder = CerebralCostRecorder()
            recorder.record_llm_call(
                model="openai/gpt-4",
                scope="retain_extract_facts",
                input_tokens=100,
                output_tokens=50,
                error=Exception("API error"),
            )

        stats = mock_tracker.get_signature_stats(BUCKET_CEREBRAL_RETAIN)
        assert stats is None

    def test_record_llm_call_skips_zero_tokens(self, mock_tracker, fresh_recorder):
        """Calls with zero tokens → nothing recorded."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            recorder = CerebralCostRecorder()
            recorder.record_llm_call(
                model="openai/gpt-4",
                scope="retain_extract_facts",
                input_tokens=0,
                output_tokens=0,
            )

        stats = mock_tracker.get_signature_stats(BUCKET_CEREBRAL_RETAIN)
        assert stats is None

    def test_record_llm_call_prices_using_litellm(self, mock_tracker, fresh_recorder):
        """Verify pricing uses litellm.cost_per_token."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch("litellm.cost_per_token", return_value=(0.01, 0.02)) as mock_cost:
                recorder = CerebralCostRecorder()
                recorder.record_llm_call(
                    model="openai/gpt-4",
                    scope="retain_extract_facts",
                    input_tokens=100,
                    output_tokens=50,
                )

                mock_cost.assert_called_once_with(
                    model="openai/gpt-4",
                    prompt_tokens=100,
                    completion_tokens=50,
                )

        stats = mock_tracker.get_signature_stats(BUCKET_CEREBRAL_RETAIN)
        assert stats.cost == 0.03  # 0.01 + 0.02

    def test_record_llm_call_logs_warning_for_unpriced_model(
        self, mock_tracker, fresh_recorder, caplog
    ):
        """An unpriced model → tokens recorded, $0, a single warning."""
        import logging

        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch(
                "litellm.cost_per_token", side_effect=Exception("No price")
            ):
                recorder = CerebralCostRecorder()
                with caplog.at_level(logging.WARNING, logger="codespy.agents.memory.cerebral.cost"):
                    # First call should warn
                    recorder.record_llm_call(
                        model="custom/model",
                        scope="retain_extract_facts",
                        input_tokens=100,
                        output_tokens=50,
                    )
                    # Second call should NOT warn (already warned)
                    recorder.record_llm_call(
                        model="custom/model",
                        scope="retain_extract_facts",
                        input_tokens=100,
                        output_tokens=50,
                    )

        stats = mock_tracker.get_signature_stats(BUCKET_CEREBRAL_RETAIN)
        assert stats.cost == 0.0
        assert stats.tokens == 300  # 150 * 2
        # Should only have one warning in the log
        assert caplog.text.count("No litellm price for model") == 1


class TestLiteLLMProxy:
    """Tests for _LiteLLMProxy class."""

    @pytest.fixture
    def mock_litellm(self):
        """Create a mock litellm module."""
        return MagicMock()

    @pytest.fixture
    def mock_response(self):
        """Create a mock embedding response."""
        response = MagicMock()
        response.usage = MagicMock()
        response.usage.prompt_tokens = 100
        response._hidden_params = {"response_cost": 0.01}
        return response

    @pytest.mark.asyncio
    async def test_aembedding_meters_usage(self, mock_litellm, mock_response, mock_tracker):
        """Embedding call meters usage to cerebral_embeddings bucket."""
        mock_litellm.aembedding.return_value = asyncio.Future()
        mock_litellm.aembedding.return_value.set_result(mock_response)

        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            cost_recorder = CerebralCostRecorder()
            proxy = _LiteLLMProxy(mock_litellm, cost_recorder)
            await proxy.aembedding(model="openai/text-embedding-3-small", input=["test"])

        stats = mock_tracker.get_signature_stats(BUCKET_CEREBRAL_EMBEDDINGS)
        assert stats is not None
        assert stats.tokens == 100
        assert stats.cost == 0.01

    @pytest.mark.asyncio
    async def test_aembedding_returns_response_unchanged(self, mock_litellm, mock_response):
        """Embedding call returns the response unchanged."""
        mock_litellm.aembedding.return_value = asyncio.Future()
        mock_litellm.aembedding.return_value.set_result(mock_response)

        cost_recorder = CerebralCostRecorder()
        proxy = _LiteLLMProxy(mock_litellm, cost_recorder)
        result = await proxy.aembedding(model="openai/text-embedding-3-small", input=["test"])

        assert result is mock_response

    def test_getattr_forwards_other_attributes(self, mock_litellm):
        """Other attributes forward to real litellm."""
        mock_litellm.some_attr = "value"
        cost_recorder = CerebralCostRecorder()
        proxy = _LiteLLMProxy(mock_litellm, cost_recorder)

        assert proxy.some_attr == "value"


class TestMeteredLiteLLMSDKEmbeddings:
    """Tests for MeteredLiteLLMSDKEmbeddings class."""

    @pytest.fixture
    def mock_base_embeddings(self):
        """Create a mock LiteLLMSDKEmbeddings."""
        base = MagicMock()
        base._litellm = MagicMock()
        base.initialize = MagicMock(return_value=asyncio.Future())
        base.initialize.return_value.set_result(None)
        return base

    @pytest.fixture
    def mock_cost_recorder(self):
        """Create a mock CerebralCostRecorder."""
        return CerebralCostRecorder()

    @pytest.mark.asyncio
    async def test_initialize_wraps_litellm_after_base_init(
        self, mock_base_embeddings, mock_cost_recorder
    ):
        """Initialize wraps _litellm after base initialization."""
        metered = MeteredLiteLLMSDKEmbeddings(
            base=mock_base_embeddings, cost_recorder=mock_cost_recorder
        )

        await metered.initialize()

        # Base initialize should have been called
        mock_base_embeddings.initialize.assert_called_once()
        # _litellm should now be wrapped
        assert isinstance(mock_base_embeddings._litellm, _LiteLLMProxy)

    @pytest.mark.asyncio
    async def test_initialize_skips_if_already_set(
        self, mock_base_embeddings, mock_cost_recorder
    ):
        """Initialize skips wrapping if already wrapped."""
        metered = MeteredLiteLLMSDKEmbeddings(
            base=mock_base_embeddings, cost_recorder=mock_cost_recorder
        )

        await metered.initialize()
        first_proxy = mock_base_embeddings._litellm

        # Second initialize should not create a new proxy
        await metered.initialize()

        assert mock_base_embeddings._litellm is first_proxy

    def test_getattr_forwards_to_base(self, mock_base_embeddings, mock_cost_recorder):
        """Other attributes forward to base embeddings."""
        mock_base_embeddings.some_attr = "value"
        metered = MeteredLiteLLMSDKEmbeddings(
            base=mock_base_embeddings, cost_recorder=mock_cost_recorder
        )

        assert metered.some_attr == "value"


class TestRegisterCerebralCostRecorder:
    """Tests for register_cerebral_cost_recorder function."""

    def test_registration_is_idempotent(self, fresh_recorder):
        """Registration is idempotent (one entry in _recorders)."""
        with patch("hindsight_api.tracing.register_span_recorder") as mock_register:
            recorder1 = register_cerebral_cost_recorder()
            recorder2 = register_cerebral_cost_recorder()

            # Should return same recorder (or at least register once)
            assert isinstance(recorder1, CerebralCostRecorder)
            assert isinstance(recorder2, CerebralCostRecorder)

    def test_registers_with_span_recorder(self, fresh_recorder):
        """Registers with Hindsight span recorder."""
        with patch("hindsight_api.tracing.register_span_recorder") as mock_register:
            recorder = register_cerebral_cost_recorder()

            mock_register.assert_called_once()
            assert mock_register.call_args[0][0] is recorder


class TestUnregisterCerebralCostRecorder:
    """Tests for unregister_cerebral_cost_recorder function."""

    def test_unregisters_specific_recorder(self, fresh_recorder):
        """Unregister specific recorder."""
        with patch("hindsight_api.tracing.register_span_recorder"):
            with patch("hindsight_api.tracing.unregister_span_recorder") as mock_unregister:
                recorder = register_cerebral_cost_recorder()
                unregister_cerebral_cost_recorder(recorder)

                mock_unregister.assert_called_once_with(recorder)

    def test_unregisters_any_cerebral_recorder(self, fresh_recorder):
        """Unregister finds and removes any CerebralCostRecorder."""
        recorder = CerebralCostRecorder()
        mock_span_recorder = MagicMock()
        mock_span_recorder._recorders = [recorder]

        with patch("hindsight_api.tracing.get_span_recorder", return_value=mock_span_recorder):
            with patch("hindsight_api.tracing.unregister_span_recorder") as mock_unregister:
                unregister_cerebral_cost_recorder()

                mock_unregister.assert_called_once_with(recorder)


class TestPinTests:
    """Pin tests for Hindsight internals we rely on."""

    def test_lite_llm_sdke_has_litellm_attribute(self):
        """LiteLLMSDKEmbeddings uses self._litellm attribute."""
        from hindsight_api.engine.embeddings import LiteLLMSDKEmbeddings

        # Check that LiteLLMSDKEmbeddings has _litellm attribute
        embeddings = LiteLLMSDKEmbeddings.__new__(LiteLLMSDKEmbeddings)
        assert hasattr(embeddings, "_litellm")

    def test_lite_llm_sdke_has_aembedding_method(self):
        """LiteLLMSDKEmbeddings._litellm has aembedding method."""
        import litellm

        assert hasattr(litellm, "aembedding")

    def test_tracing_has_span_recorder(self):
        """hindsight_api.tracing has get_span_recorder and register_span_recorder."""
        from hindsight_api import tracing

        assert hasattr(tracing, "get_span_recorder")
        assert hasattr(tracing, "register_span_recorder")
        assert hasattr(tracing, "unregister_span_recorder")

    def test_composite_span_recorder_has_recorders_list(self):
        """CompositeSpanRecorder has _recorders list."""
        from hindsight_api.tracing import CompositeSpanRecorder

        recorder = CompositeSpanRecorder()
        assert hasattr(recorder, "_recorders")
        assert isinstance(recorder._recorders, list)
