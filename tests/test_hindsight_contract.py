"""Contract tests for the Hindsight internals codespy still relies on.

Real ``hindsight_api``, no database. Each assertion message names the codespy
call site that breaks when Hindsight changes the internal.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("hindsight_api")

from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM  # noqa: E402

from codespy.agents.memory.cerebral.cerebral import (  # noqa: E402
    _BRIEFING_TRIGGER_STATIC,
    Cerebral,
)


@pytest.fixture(scope="module")
def engine():
    """A MemoryEngine built without initialize(): __init__ opens no connection."""
    from hindsight_api import MemoryEngine
    from hindsight_api.engine.cross_encoder import RRFPassthroughCrossEncoder
    from hindsight_api.engine.embeddings import LiteLLMSDKEmbeddings

    return MemoryEngine(
        db_url="postgresql://user:pw@127.0.0.1:1/none",
        memory_llm_provider="litellm",
        memory_llm_model="openai/gpt-4o-mini",
        memory_llm_api_key="x",
        embeddings=LiteLLMSDKEmbeddings(model="openai/text-embedding-3-small", api_key=None),
        cross_encoder=RRFPassthroughCrossEncoder(),
        skip_llm_verification=True,
    )


def _strategy_payload(cap: int) -> dict:
    c = Cerebral.__new__(Cerebral)
    c._max_observations_per_scope = cap
    return c._consolidation_strategies()[0]


class TestLLMProvider:
    def test_provider_impl_is_litellm_and_reassignable(self):
        from hindsight_api.engine.llm_wrapper import LLMProvider

        llm = LLMProvider(
            provider="litellm",
            model="openai/gpt-4o-mini",
            api_key="x",
            base_url="",
            reasoning_effort=None,
        )
        assert isinstance(llm._provider_impl, LiteLLMLLM), (
            "Cerebral.__init__ (extraction model) reads LLMProvider._provider_impl"
        )
        sentinel = object()
        llm._provider_impl = sentinel
        assert llm._provider_impl is sentinel, (
            "structured_output.install_two_step_structured_output reassigns _provider_impl"
        )


class TestMemoryEngineInternals:
    # Checked on a real instance: MemoryEngine.__init__ needs no DB or network.
    @pytest.mark.parametrize(
        "attr",
        [
            "_llm_config",
            "_retain_llm_config",
            "_consolidation_llm_config",
            "_reflect_llm_config",
            "_mental_model_refresh_llm_override",
        ],
    )
    def test_llm_config_attrs(self, engine, attr):
        assert hasattr(engine, attr), (
            f"structured_output.install_two_step_structured_output reads engine.{attr}"
        )

    def test_backend(self, engine):
        assert getattr(engine, "_backend", None) is not None, (
            "cerebral._SchemaSyncTaskBackend._mark_*_if_pending and "
            "orphans.repair_orphaned_operations use engine._backend"
        )

    def test_execute_task_pops_schema(self):
        from hindsight_api import MemoryEngine

        assert 'task_dict.pop("_schema"' in inspect.getsource(MemoryEngine._execute_task), (
            "cerebral._SchemaSyncTaskBackend.submit_task injects _schema for _execute_task"
        )

    def test_list_mental_models_signature(self):
        from hindsight_api import MemoryEngine

        params = inspect.signature(MemoryEngine.list_mental_models).parameters
        for name in ("detail", "limit", "offset", "request_context"):
            assert name in params, f"Cerebral._clear_legacy_placeholders passes {name}="


class TestImports:
    def test_imports(self):
        from hindsight_api.config import clear_config_cache, get_config
        from hindsight_api.engine.db_utils import acquire_with_retry
        from hindsight_api.engine.multi_llm import MultiLLMProvider
        from hindsight_api.tracing import register_span_recorder, unregister_span_recorder
        from hindsight_api.worker.exceptions import DeferOperation, format_task_error

        assert callable(acquire_with_retry), "cerebral._SchemaSyncTaskBackend, orphans.py"
        assert issubclass(DeferOperation, Exception), "cerebral._SchemaSyncTaskBackend"
        assert callable(format_task_error), "cerebral._SchemaSyncTaskBackend._mark_failed"
        assert callable(register_span_recorder), "cost.register_cerebral_cost_recorder"
        assert callable(unregister_span_recorder), "cost.unregister_cerebral_cost_recorder"
        assert MultiLLMProvider is not None, "structured_output.install_two_step_structured_output"
        assert callable(clear_config_cache), "config_memory._hindsight_config / reset_cerebral"
        assert callable(get_config), "config_memory._hindsight_config"


class TestPayloadValidation:
    @pytest.mark.parametrize("cap", [100, -1])
    def test_consolidation_strategy_payload(self, cap):
        from hindsight_api.engine.response_models import StrictConsolidationStrategySpec

        spec = StrictConsolidationStrategySpec.model_validate(_strategy_payload(cap))
        assert spec.max_observations_per_scope == cap, (
            "Cerebral._consolidation_strategies payload rejected by Hindsight"
        )

    def test_briefing_trigger_validates(self):
        from hindsight_api.api.http import MentalModelTrigger

        trigger = _BRIEFING_TRIGGER_STATIC | {
            "refresh_after_consolidation": True,
            "min_refresh_interval_seconds": 0,
        }
        model = MentalModelTrigger.model_validate(trigger)
        # create_mental_model stores the dict without validation, so a typo would
        # be stored silently: every key codespy sends must be a known field.
        unknown = set(trigger) - set(MentalModelTrigger.model_fields)
        assert not unknown, f"cerebral._BRIEFING_TRIGGER_STATIC has unknown keys: {unknown}"
        assert model.budget is not None and model.budget.value == "low"
        assert model.keep_trace is True
