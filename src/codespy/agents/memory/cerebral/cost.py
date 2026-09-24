"""Cost tracking for Cerebral (Hindsight) LLM calls and embeddings.

Uses the Hindsight span recorder to intercept LLM calls made by MemoryEngine
and price them using litellm.cost_per_token. Embeddings are metered by
wrapping the LiteLLM SDK with a proxy that captures usage.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hindsight_api.engine.embeddings import LiteLLMSDKEmbeddings

logger = logging.getLogger(__name__)

# Module-level lock for thread-safe registration
_recorder_lock = threading.Lock()
_recorder_registered = False

# Buckets for cost attribution
BUCKET_CEREBRAL_RETAIN = "cerebral_retain"
BUCKET_CEREBRAL_OTHER = "cerebral_other"
BUCKET_CEREBRAL_EMBEDDINGS = "cerebral_embeddings"

# Track models we've warned about missing prices (one warning per model)
_warned_unpriced_models: set[str] = set()


class CerebralCostRecorder:
    """Records Hindsight LLM calls into the CostTracker.

    Registered as a span recorder with Hindsight's CompositeSpanRecorder.
    Receives every LLM call made by MemoryEngine (retain, consolidation, etc.)
    and accumulates costs via CostTracker.add_external_call.
    """

    def record_llm_call(  # type: ignore[return]
        self,
        *,
        model: str = "",
        scope: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: Exception | None = None,
        **_: Any,
    ) -> None:
        """Record a completed LLM call.

        Args:
            model: Full litellm model string (e.g., "bedrock/converse/moonshotai.kimi-k2.5")
            scope: Hindsight scope (e.g., "retain_extract_facts", "consolidation")
            input_tokens: Input/prompt token count
            output_tokens: Output/completion token count
            error: Exception if the call failed
            **_: Ignored extra kwargs for forward compatibility
        """
        try:
            # Skip failed calls
            if error is not None:
                return

            # Skip calls with no tokens
            if input_tokens == 0 and output_tokens == 0:
                return

            # Determine bucket from scope
            if scope and scope.startswith("retain"):
                bucket = BUCKET_CEREBRAL_RETAIN
            else:
                bucket = BUCKET_CEREBRAL_OTHER

            # Price the call using litellm
            prompt_cost, completion_cost = self._price_call_split(model, input_tokens, output_tokens)
            cost = prompt_cost + completion_cost
            tokens = input_tokens + output_tokens

            # Import here to avoid circular imports at module load
            from codespy.agents.cost_tracker import get_cost_tracker

            tracker = get_cost_tracker()
            tracker.add_external_call(
                bucket,
                cost=cost,
                tokens=tokens,
                calls=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=prompt_cost,
                output_cost=completion_cost,
            )

        except Exception as e:
            # Metering must never break the actual operation
            logger.debug("CerebralCostRecorder failed to record call: %s", e)

    def _price_call(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Price a call using litellm.cost_per_token.

        Returns 0.0 if the model has no price registered. Logs a warning
        once per unpriced model.

        Deprecated: Use _price_call_split for input/output cost breakdown.
        """
        prompt_cost, completion_cost = self._price_call_split(model, input_tokens, output_tokens)
        return prompt_cost + completion_cost

    def _price_call_split(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> tuple[float, float]:
        """Price a call using litellm.cost_per_token.

        Returns (prompt_cost, completion_cost). Returns (0.0, 0.0) if the model
        has no price registered. Logs a warning once per unpriced model.

        Returns:
            Tuple of (input/prompt cost, output/completion cost)
        """
        try:
            import litellm

            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            return float(prompt_cost), float(completion_cost)
        except Exception:
            # Model may not have a price
            if model not in _warned_unpriced_models:
                _warned_unpriced_models.add(model)
                logger.warning(
                    "No litellm price for model %s; cost will be $0. "
                    "Tokens are still recorded.",
                    model,
                )
            return 0.0, 0.0


class _LiteLLMProxy:
    """Proxy for the litellm module that meters embedding calls.

    Intercepts ``aembedding`` to capture usage and record costs before
    returning the response. All other attributes forward to the real litellm.
    """

    def __init__(self, real_litellm: Any, cost_recorder: "CerebralCostRecorder") -> None:
        self._real = real_litellm
        self._recorder = cost_recorder

    async def aembedding(self, **kwargs: Any) -> Any:
        """Call litellm.aembedding and meter the usage."""
        response = await self._real.aembedding(**kwargs)
        self._meter_embedding(response, kwargs.get("model", ""))
        return response

    def __getattr__(self, name: str) -> Any:
        """Forward all other attributes to the real litellm."""
        return getattr(self._real, name)

    def _meter_embedding(self, response: Any, model: str) -> None:
        """Extract usage from embedding response and record cost."""
        try:
            # Try _hidden_params["response_cost"] first (litellm populates this)
            cost: float | None = None
            if hasattr(response, "_hidden_params") and isinstance(response._hidden_params, dict):
                cost = response._hidden_params.get("response_cost")

            # Fallback to litellm.completion_cost if available
            if cost is None:
                try:
                    import litellm

                    cost = litellm.completion_cost(response)
                except Exception:
                    cost = 0.0

            # Token extraction: usage.prompt_tokens or usage.total_tokens
            tokens = 0
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                if hasattr(usage, "prompt_tokens"):
                    tokens = usage.prompt_tokens
                elif hasattr(usage, "total_tokens"):
                    tokens = usage.total_tokens
                elif isinstance(usage, dict):
                    tokens = usage.get("prompt_tokens", 0) or usage.get("total_tokens", 0)

            if tokens == 0:
                # Can't meter without tokens
                return

            # Import here to avoid circular imports
            from codespy.agents.cost_tracker import get_cost_tracker

            tracker = get_cost_tracker()
            # Embeddings have no output tokens - all tokens are input
            tracker.add_external_call(
                BUCKET_CEREBRAL_EMBEDDINGS,
                cost=cost or 0.0,
                tokens=tokens,
                calls=1,
                input_tokens=tokens,
                output_tokens=0,
                input_cost=cost or 0.0,
                output_cost=0.0,
            )

        except Exception as e:
            logger.debug("Failed to meter embedding call: %s", e)


class MeteredLiteLLMSDKEmbeddings:
    """Wrapper for LiteLLMSDKEmbeddings that meters costs.

    Proxies the underlying ``_litellm`` module after initialization
to capture embedding usage. The dimension detection probe during
    ``initialize()`` is intentionally unmetered.
    """

    def __init__(self, *, base: LiteLLMSDKEmbeddings, cost_recorder: CerebralCostRecorder) -> None:
        self._base = base
        self._recorder = cost_recorder
        self._proxy_set = False

    async def initialize(self) -> None:
        """Initialize the base embeddings, then wrap _litellm."""
        # Run base initialization (this does the dimension probe)
        await self._base.initialize()

        # Wrap _litellm after initialization to meter actual retain/recall
        # calls, not the probe. Must check if already wrapped because
        # initialize() may be called multiple times.
        if not self._proxy_set and hasattr(self._base, "_litellm") and self._base._litellm is not None:
            self._base._litellm = _LiteLLMProxy(self._base._litellm, self._recorder)
            self._proxy_set = True

    def __getattr__(self, name: str) -> Any:
        """Forward all other attributes to the base embeddings."""
        return getattr(self._base, name)


def register_cerebral_cost_recorder() -> CerebralCostRecorder:
    """Register the Cerebral cost recorder with Hindsight's span recorder.

    Idempotent: returns existing recorder if already registered.
    Call unregister_cerebral_cost_recorder() to remove.

    Returns:
        The registered CerebralCostRecorder instance.
    """
    global _recorder_registered

    with _recorder_lock:
        if _recorder_registered:
            # Find and return existing recorder
            try:
                from hindsight_api.tracing import get_span_recorder

                recorder = get_span_recorder()
                for r in getattr(recorder, "_recorders", []):
                    if isinstance(r, CerebralCostRecorder):
                        return r
            except Exception:
                pass
            # If we can't find it, create a new one anyway

        recorder = CerebralCostRecorder()
        try:
            from hindsight_api.tracing import register_span_recorder

            register_span_recorder(recorder)
            _recorder_registered = True
            logger.debug("CerebralCostRecorder registered")
        except Exception as e:
            logger.warning("Failed to register CerebralCostRecorder: %s", e)

        return recorder


def unregister_cerebral_cost_recorder(recorder: CerebralCostRecorder | None = None) -> None:
    """Unregister the Cerebral cost recorder.

    Args:
        recorder: The recorder to unregister. If None, attempts to find
            and unregister any CerebralCostRecorder in the composite.
    """
    global _recorder_registered

    with _recorder_lock:
        try:
            from hindsight_api.tracing import get_span_recorder, unregister_span_recorder

            if recorder is not None:
                unregister_span_recorder(recorder)
            else:
                # Find and unregister any CerebralCostRecorder
                span_recorder = get_span_recorder()
                for r in getattr(span_recorder, "_recorders", []):
                    if isinstance(r, CerebralCostRecorder):
                        unregister_span_recorder(r)
                        break
            _recorder_registered = False
            logger.debug("CerebralCostRecorder unregistered")
        except Exception as e:
            logger.debug("Failed to unregister CerebralCostRecorder: %s", e)
