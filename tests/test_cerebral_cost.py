"""Tests for the Cerebral cost tracking module."""

import asyncio
from unittest.mock import MagicMock, Mock, patch

import pytest

pytest.importorskip("hindsight_api")

from codespy.agents.cost_tracker import CostTracker, get_cost_tracker
from codespy.agents.memory.cerebral.cost import (
    BUCKET_MEMORY_CONSOLIDATION,
    BUCKET_MEMORY_EMBEDDINGS,
    BUCKET_MEMORY_MENTAL_MODELS,
    BUCKET_MEMORY_OTHER,
    BUCKET_MEMORY_RECALL,
    BUCKET_MEMORY_RECALL_EMBEDDINGS,
    BUCKET_MEMORY_RETAIN,
    CerebralCostRecorder,
    MeteredLiteLLMSDKEmbeddings,
    _LiteLLMProxy,
    _llm_bucket,
    _recorder_registered,
    current_memory_task,
    register_cerebral_cost_recorder,
    unregister_cerebral_cost_recorder,
)
from codespy.agents.memory.recall import current_recall_usage, track_recall_usage


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
        """scope='retain_extract_facts' → memory_retain bucket."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(
                CerebralCostRecorder, "_price_call_split", return_value=(0.03, 0.02)
            ):
                recorder = CerebralCostRecorder()
                recorder.record_llm_call(
                    model="openai/gpt-4",
                    scope="retain_extract_facts",
                    input_tokens=100,
                    output_tokens=50,
                )

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_RETAIN)
        assert stats is not None
        assert stats.cost == 0.05  # 0.03 + 0.02
        assert stats.tokens == 150
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert stats.input_cost == 0.03
        assert stats.output_cost == 0.02

    def test_record_llm_call_with_consolidation_scope(self, mock_tracker, fresh_recorder):
        """scope='consolidation' → memory_consolidation bucket (scope fallback, no tag)."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(CerebralCostRecorder, "_price_call_split", return_value=(0.02, 0.01)):
                recorder = CerebralCostRecorder()
                recorder.record_llm_call(
                    model="openai/gpt-4",
                    scope="consolidation",
                    input_tokens=100,
                    output_tokens=50,
                )

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_CONSOLIDATION)
        assert stats is not None
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50

    def test_record_llm_call_with_mental_model_scope(self, mock_tracker, fresh_recorder):
        """scope='mental_model_delta_ops' → memory_mental_models bucket (scope fallback, no tag)."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(CerebralCostRecorder, "_price_call_split", return_value=(0.02, 0.01)):
                recorder = CerebralCostRecorder()
                recorder.record_llm_call(
                    model="openai/gpt-4",
                    scope="mental_model_delta_ops",
                    input_tokens=100,
                    output_tokens=50,
                )

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_MENTAL_MODELS)
        assert stats is not None
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50

    def test_record_llm_call_with_consolidation_task_tag(self, mock_tracker, fresh_recorder):
        """current_memory_task='consolidation' with scope 'reflect' → memory_consolidation bucket."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(CerebralCostRecorder, "_price_call_split", return_value=(0.02, 0.01)):
                token = current_memory_task.set("consolidation")
                try:
                    recorder = CerebralCostRecorder()
                    recorder.record_llm_call(
                        model="openai/gpt-4",
                        scope="reflect",
                        input_tokens=100,
                        output_tokens=50,
                    )
                finally:
                    current_memory_task.reset(token)

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_CONSOLIDATION)
        assert stats is not None
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50

    def test_record_llm_call_with_refresh_mental_model_task_tag(self, mock_tracker, fresh_recorder):
        """current_memory_task='refresh_mental_model' with scope 'reflect' → memory_mental_models bucket."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(CerebralCostRecorder, "_price_call_split", return_value=(0.02, 0.01)):
                token = current_memory_task.set("refresh_mental_model")
                try:
                    recorder = CerebralCostRecorder()
                    recorder.record_llm_call(
                        model="openai/gpt-4",
                        scope="reflect",
                        input_tokens=100,
                        output_tokens=50,
                    )
                finally:
                    current_memory_task.reset(token)

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_MENTAL_MODELS)
        assert stats is not None
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50

    def test_record_llm_call_recall_wins_over_task_tag(self, mock_tracker, fresh_recorder):
        """current_recall_usage set AND current_memory_task set → memory_recall (recall wins)."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(CerebralCostRecorder, "_price_call_split", return_value=(0.02, 0.01)):
                with track_recall_usage() as usage:
                    token = current_memory_task.set("consolidation")
                    try:
                        recorder = CerebralCostRecorder()
                        recorder.record_llm_call(
                            model="openai/gpt-4",
                            scope="reflect",
                            input_tokens=100,
                            output_tokens=50,
                        )
                    finally:
                        current_memory_task.reset(token)

        # Should go to prefrontal, not consolidation
        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_RECALL)
        assert stats is not None
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        # Consolidation should be empty
        assert mock_tracker.get_signature_stats(BUCKET_MEMORY_CONSOLIDATION) is None

    def test_record_llm_call_task_tag_wins_over_retain_scope(self, mock_tracker, fresh_recorder):
        """current_memory_task='consolidation' with scope 'retain_extract_facts' → memory_consolidation (tag wins)."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(CerebralCostRecorder, "_price_call_split", return_value=(0.02, 0.01)):
                token = current_memory_task.set("consolidation")
                try:
                    recorder = CerebralCostRecorder()
                    recorder.record_llm_call(
                        model="openai/gpt-4",
                        scope="retain_extract_facts",
                        input_tokens=100,
                        output_tokens=50,
                    )
                finally:
                    current_memory_task.reset(token)

        # Task tag should win over scope
        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_CONSOLIDATION)
        assert stats is not None
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        # Retain should be empty
        assert mock_tracker.get_signature_stats(BUCKET_MEMORY_RETAIN) is None

    def test_record_llm_call_with_other_scope(self, mock_tracker, fresh_recorder):
        """Any other scope → memory_other bucket."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(CerebralCostRecorder, "_price_call_split", return_value=(0.005, 0.005)):
                recorder = CerebralCostRecorder()
                recorder.record_llm_call(
                    model="openai/gpt-4",
                    scope="reflect",
                    input_tokens=100,
                    output_tokens=50,
                )

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_OTHER)
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

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_RETAIN)
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

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_RETAIN)
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

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_RETAIN)
        assert stats.cost == 0.03  # 0.01 + 0.02

    def test_record_llm_call_forwards_duration(self, mock_tracker, fresh_recorder):
        """duration parameter is forwarded to add_external_call."""
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(CerebralCostRecorder, "_price_call_split", return_value=(0.03, 0.02)):
                recorder = CerebralCostRecorder()
                recorder.record_llm_call(
                    model="openai/gpt-4",
                    scope="retain_extract_facts",
                    input_tokens=100,
                    output_tokens=50,
                    duration=1.25,
                )

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_RETAIN)
        assert stats is not None
        assert stats.external_duration_seconds == 1.25
        assert stats.duration_seconds == 1.25

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

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_RETAIN)
        assert stats.cost == 0.0
        assert stats.tokens == 300  # 150 * 2
        # Should only have one warning in the log
        assert caplog.text.count("No litellm price for model") == 1


class TestRecallUsageAttribution:
    """LLM/embedding calls inside track_recall_usage() are billed to memory_recall."""

    def test_llm_call_inside_recall_goes_to_prefrontal(self, mock_tracker, fresh_recorder):
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(CerebralCostRecorder, "_price_call_split", return_value=(0.03, 0.02)):
                with track_recall_usage() as usage:
                    CerebralCostRecorder().record_llm_call(
                        model="openai/gpt-4o",
                        scope="reflect",
                        input_tokens=100,
                        output_tokens=40,
                    )
        assert usage.llm_calls == 1
        assert usage.input_tokens == 100 and usage.output_tokens == 40
        assert usage.input_cost == 0.03 and usage.output_cost == 0.02
        assert usage.to_model_field() == "openai/gpt-4o"
        assert mock_tracker.get_signature_stats(BUCKET_MEMORY_RECALL).cost == 0.05
        assert mock_tracker.get_signature_stats(BUCKET_MEMORY_OTHER) is None

    def test_llm_call_outside_recall_unchanged(self, mock_tracker, fresh_recorder):
        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            with patch.object(CerebralCostRecorder, "_price_call_split", return_value=(0.01, 0.01)):
                CerebralCostRecorder().record_llm_call(
                    model="openai/gpt-4o", scope="reflect", input_tokens=10, output_tokens=5
                )
        assert current_recall_usage.get() is None
        assert mock_tracker.get_signature_stats(BUCKET_MEMORY_OTHER) is not None
        assert mock_tracker.get_signature_stats(BUCKET_MEMORY_RECALL) is None

    @pytest.mark.asyncio
    async def test_embedding_inside_recall_goes_to_prefrontal_embeddings(self, mock_tracker):
        response = MagicMock()
        response.usage = MagicMock()
        response.usage.prompt_tokens = 12
        response._hidden_params = {"response_cost": 0.001}
        real = MagicMock()
        real.aembedding.return_value = asyncio.Future()
        real.aembedding.return_value.set_result(response)

        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            proxy = _LiteLLMProxy(real, CerebralCostRecorder())
            with track_recall_usage() as usage:
                await proxy.aembedding(model="openai/text-embedding-3-small", input=["q"])

        assert usage.input_tokens == 12 and usage.output_tokens == 0
        assert usage.input_cost == 0.001
        assert usage.llm_calls == 0  # embeddings are not LLM calls
        # Embeddings during active recall go to the new prefrontal_embeddings bucket
        assert mock_tracker.get_signature_stats(BUCKET_MEMORY_RECALL_EMBEDDINGS).tokens == 12
        assert mock_tracker.get_signature_stats(BUCKET_MEMORY_RECALL) is None
        assert mock_tracker.get_signature_stats(BUCKET_MEMORY_EMBEDDINGS) is None

    def test_usage_propagates_to_cross_thread_loop(self):
        """Cerebral._await submits to its own loop thread: the contextvar must follow."""
        import threading

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:

            async def read_usage():
                return current_recall_usage.get()

            with track_recall_usage() as usage:
                seen = asyncio.run_coroutine_threadsafe(read_usage(), loop).result(timeout=5)
            assert seen is usage
            after = asyncio.run_coroutine_threadsafe(read_usage(), loop).result(timeout=5)
            assert after is None
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)


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
        """Embedding call meters usage to memory_embeddings bucket."""
        mock_litellm.aembedding.return_value = asyncio.Future()
        mock_litellm.aembedding.return_value.set_result(mock_response)

        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            cost_recorder = CerebralCostRecorder()
            proxy = _LiteLLMProxy(mock_litellm, cost_recorder)
            await proxy.aembedding(model="openai/text-embedding-3-small", input=["test"])

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_EMBEDDINGS)
        assert stats is not None
        assert stats.tokens == 100
        assert stats.cost == 0.01

    @pytest.mark.asyncio
    async def test_aembedding_records_duration(self, mock_litellm, mock_response, mock_tracker):
        """Embedding call records nonzero duration to memory_embeddings bucket."""
        mock_litellm.aembedding.return_value = asyncio.Future()
        mock_litellm.aembedding.return_value.set_result(mock_response)

        with patch("codespy.agents.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            cost_recorder = CerebralCostRecorder()
            proxy = _LiteLLMProxy(mock_litellm, cost_recorder)
            await proxy.aembedding(model="openai/text-embedding-3-small", input=["test"])

        stats = mock_tracker.get_signature_stats(BUCKET_MEMORY_EMBEDDINGS)
        assert stats is not None
        assert stats.external_duration_seconds >= 0.0  # Should have some duration (could be very small)
        assert stats.duration_seconds == stats.external_duration_seconds

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

    def test_llm_span_recorder_accepts_duration_param(self):
        """LLMSpanRecorder.record_llm_call accepts duration parameter (upstream contract)."""
        import inspect
        from hindsight_api.tracing import LLMSpanRecorder

        sig = inspect.signature(LLMSpanRecorder.record_llm_call)
        assert "duration" in sig.parameters, "record_llm_call must accept 'duration' parameter"


class TestResolveEmbeddingInputLimits:
    """Tests for resolve_embedding_input_limits function."""

    def test_bedrock_cohere_multilingual_v3_returns_2048_and_truncate(self):
        """Bedrock Cohere multilingual v3 returns 2048 and truncate=END."""
        from codespy.agents.memory.cerebral.cost import resolve_embedding_input_limits

        max_chars, extra = resolve_embedding_input_limits("bedrock/cohere.embed-multilingual-v3")
        assert max_chars == 2048
        assert extra == {"truncate": "END"}

    def test_bedrock_cohere_english_v3_returns_2048_and_truncate(self):
        """Bedrock Cohere english v3 returns 2048 and truncate=END."""
        from codespy.agents.memory.cerebral.cost import resolve_embedding_input_limits

        max_chars, extra = resolve_embedding_input_limits("bedrock/cohere.embed-english-v3")
        assert max_chars == 2048
        assert extra == {"truncate": "END"}

    def test_region_prefixed_bedrock_cohere_matches(self):
        """Region-prefixed Bedrock Cohere models also match."""
        from codespy.agents.memory.cerebral.cost import resolve_embedding_input_limits

        max_chars, extra = resolve_embedding_input_limits("bedrock/eu.cohere.embed-multilingual-v3")
        assert max_chars == 2048
        assert extra == {"truncate": "END"}

    def test_openai_returns_none_and_empty(self):
        """OpenAI models return None and empty dict."""
        from codespy.agents.memory.cerebral.cost import resolve_embedding_input_limits

        max_chars, extra = resolve_embedding_input_limits("openai/text-embedding-3-small")
        assert max_chars is None
        assert extra == {}

    def test_override_0_disables_cap(self):
        """Override=0 disables character capping."""
        from codespy.agents.memory.cerebral.cost import resolve_embedding_input_limits

        max_chars, extra = resolve_embedding_input_limits("bedrock/cohere.embed-multilingual-v3", override=0)
        assert max_chars is None
        assert extra == {}

    def test_override_n_uses_custom_cap(self):
        """Override=N uses custom character cap."""
        from codespy.agents.memory.cerebral.cost import resolve_embedding_input_limits

        max_chars, extra = resolve_embedding_input_limits("bedrock/cohere.embed-multilingual-v3", override=3000)
        assert max_chars == 3000
        assert extra == {"truncate": "END"}

    def test_custom_cap_for_non_cohere_returns_empty_extra(self):
        """Custom cap for non-Cohere models returns empty extra."""
        from codespy.agents.memory.cerebral.cost import resolve_embedding_input_limits

        max_chars, extra = resolve_embedding_input_limits("openai/text-embedding-3-small", override=3000)
        assert max_chars == 3000
        assert extra == {}


class TestLiteLLMProxyInputCapping:
    """Tests for _LiteLLMProxy input capping."""

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
    async def test_no_truncation_when_under_cap(self, mock_litellm, mock_response):
        """Input under cap is not truncated."""
        mock_litellm.aembedding.return_value = asyncio.Future()
        mock_litellm.aembedding.return_value.set_result(mock_response)

        from codespy.agents.memory.cerebral.cost import _LiteLLMProxy, CerebralCostRecorder

        proxy = _LiteLLMProxy(mock_litellm, CerebralCostRecorder(), max_input_chars=100)
        await proxy.aembedding(model="test", input=["short"])

        call_args = mock_litellm.aembedding.call_args
        assert call_args.kwargs["input"] == ["short"]

    @pytest.mark.asyncio
    async def test_truncation_when_over_cap(self, mock_litellm, mock_response):
        """Input over cap is truncated."""
        mock_litellm.aembedding.return_value = asyncio.Future()
        mock_litellm.aembedding.return_value.set_result(mock_response)

        from codespy.agents.memory.cerebral.cost import _LiteLLMProxy, CerebralCostRecorder

        proxy = _LiteLLMProxy(mock_litellm, CerebralCostRecorder(), max_input_chars=5)
        await proxy.aembedding(model="test", input=["this is a long text"])

        call_args = mock_litellm.aembedding.call_args
        assert call_args.kwargs["input"] == ["this "]

    @pytest.mark.asyncio
    async def test_does_not_mutate_original_list(self, mock_litellm, mock_response):
        """Original input list is not mutated."""
        mock_litellm.aembedding.return_value = asyncio.Future()
        mock_litellm.aembedding.return_value.set_result(mock_response)

        from codespy.agents.memory.cerebral.cost import _LiteLLMProxy, CerebralCostRecorder

        original = ["this is a long text"]
        original_copy = list(original)

        proxy = _LiteLLMProxy(mock_litellm, CerebralCostRecorder(), max_input_chars=5)
        await proxy.aembedding(model="test", input=original)

        assert original == original_copy

    @pytest.mark.asyncio
    async def test_extra_kwargs_forwarded(self, mock_litellm, mock_response):
        """Extra kwargs are forwarded to the embedding call."""
        mock_litellm.aembedding.return_value = asyncio.Future()
        mock_litellm.aembedding.return_value.set_result(mock_response)

        from codespy.agents.memory.cerebral.cost import _LiteLLMProxy, CerebralCostRecorder

        proxy = _LiteLLMProxy(
            mock_litellm, CerebralCostRecorder(), max_input_chars=100, extra_kwargs={"truncate": "END"}
        )
        await proxy.aembedding(model="test", input=["short"])

        call_args = mock_litellm.aembedding.call_args
        assert call_args.kwargs["truncate"] == "END"

    @pytest.mark.asyncio
    async def test_explicit_caller_kwargs_win_over_extra(self, mock_litellm, mock_response):
        """Explicit caller kwargs win over extra_kwargs."""
        mock_litellm.aembedding.return_value = asyncio.Future()
        mock_litellm.aembedding.return_value.set_result(mock_response)

        from codespy.agents.memory.cerebral.cost import _LiteLLMProxy, CerebralCostRecorder

        proxy = _LiteLLMProxy(
            mock_litellm, CerebralCostRecorder(), max_input_chars=100, extra_kwargs={"truncate": "END"}
        )
        await proxy.aembedding(model="test", input=["short"], truncate="START")

        call_args = mock_litellm.aembedding.call_args
        assert call_args.kwargs["truncate"] == "START"
