"""Cost tracking for Cerebral (Hindsight) LLM calls and embeddings.

Uses the Hindsight span recorder to intercept LLM calls made by MemoryEngine
and price them using litellm.cost_per_token. Embeddings are metered by
wrapping the LiteLLM SDK with a proxy that captures usage.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from codespy.agents.memory.recall import current_recall_usage
from codespy.config_memory import (
    MEMORY_EMBEDDINGS,
    MEMORY_OTHER,
    MEMORY_PREFRONTAL,
    MEMORY_RETAIN,
)

if TYPE_CHECKING:
    from hindsight_api.engine.embeddings import LiteLLMSDKEmbeddings

logger = logging.getLogger(__name__)

# Track if we've warned about truncating inputs (once per process)
_warned_truncation: bool = False

# Module-level lock for thread-safe registration
_recorder_lock = threading.Lock()
_recorder_registered = False

# Buckets for cost attribution (using memory_* naming to match config/env names)
BUCKET_MEMORY_RETAIN = MEMORY_RETAIN
BUCKET_MEMORY_OTHER = MEMORY_OTHER
BUCKET_MEMORY_EMBEDDINGS = MEMORY_EMBEDDINGS
# Prefrontal reads (recall/reflect inside a track_recall_usage() block)
BUCKET_MEMORY_PREFRONTAL = MEMORY_PREFRONTAL
# Prefrontal embeddings (separate bucket for embeddings during Prefrontal recall)
BUCKET_MEMORY_PREFRONTAL_EMBEDDINGS = f"{MEMORY_PREFRONTAL}_embeddings"

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
        duration: float = 0.0,
        **_: Any,
    ) -> None:
        """Record a completed LLM call.

        Args:
            model: Full litellm model string (e.g., "bedrock/converse/moonshotai.kimi-k2.5")
            scope: Hindsight scope (e.g., "retain_extract_facts", "consolidation")
            input_tokens: Input/prompt token count
            output_tokens: Output/completion token count
            error: Exception if the call failed
            duration: Duration of the call in seconds
            **_: Ignored extra kwargs for forward compatibility

        Note:
            Provider-internal retries are not counted — only successful calls
            and errored calls that reached the provider (with billed usage).
        """
        try:
            # Skip calls with no tokens (includes failed calls that never reached provider)
            if input_tokens == 0 and output_tokens == 0:
                return

            # Price the call using litellm
            prompt_cost, completion_cost = self._price_call_split(model, input_tokens, output_tokens)
            cost = prompt_cost + completion_cost
            tokens = input_tokens + output_tokens

            # Determine bucket: an active Prefrontal recall wins over the scope,
            # because Hindsight scopes cannot tell a Prefrontal reflect apart
            # from a mental-model refresh.
            usage = current_recall_usage.get()
            if usage is not None:
                usage.add_llm(model, input_tokens, output_tokens, prompt_cost, completion_cost)
                bucket = BUCKET_MEMORY_PREFRONTAL
            elif scope and scope.startswith("retain"):
                bucket = BUCKET_MEMORY_RETAIN
            else:
                bucket = BUCKET_MEMORY_OTHER

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
                duration=duration,
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

    def __init__(
        self,
        real_litellm: Any,
        cost_recorder: "CerebralCostRecorder",
        max_input_chars: int | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._real = real_litellm
        self._recorder = cost_recorder
        self._max_input_chars = max_input_chars
        self._extra_kwargs = extra_kwargs or {}

    async def aembedding(self, **kwargs: Any) -> Any:
        """Call litellm.aembedding and meter the usage."""
        # Apply character cap to input if configured
        if self._max_input_chars is not None:
            kwargs = self._apply_input_cap(kwargs)

        # Merge extra_kwargs (explicit caller values win)
        for key, value in self._extra_kwargs.items():
            kwargs.setdefault(key, value)

        start = time.perf_counter()
        response = await self._real.aembedding(**kwargs)
        elapsed = time.perf_counter() - start
        self._meter_embedding(response, kwargs.get("model", ""), elapsed)
        return response

    def _apply_input_cap(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Apply character cap to input strings.

        Creates a new input list with each string longer than the cap cut to
        text[:cap]. Non-str items are left as-is. Logs a warning on first
        truncation, then debug.
        """
        global _warned_truncation

        input_data = kwargs.get("input")
        if not isinstance(input_data, list):
            return kwargs

        new_input: list[Any] = []
        max_seen = 0
        truncated_count = 0

        for item in input_data:
            if isinstance(item, str) and len(item) > self._max_input_chars:
                new_input.append(item[: self._max_input_chars])
                max_seen = max(max_seen, len(item))
                truncated_count += 1
            else:
                new_input.append(item)
                if isinstance(item, str):
                    max_seen = max(max_seen, len(item))

        if truncated_count > 0:
            level = logging.WARNING if not _warned_truncation else logging.DEBUG
            _warned_truncation = True
            logger.log(
                level,
                "Truncated %d embedding input(s) to %d chars (largest was %d)",
                truncated_count,
                self._max_input_chars,
                max_seen,
            )

        # Return new kwargs with capped input (don't mutate caller's dict)
        return {**kwargs, "input": new_input}

    def __getattr__(self, name: str) -> Any:
        """Forward all other attributes to the real litellm."""
        return getattr(self._real, name)

    def _meter_embedding(self, response: Any, model: str, duration: float = 0.0) -> None:
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

            usage = current_recall_usage.get()
            if usage is not None:
                usage.add_embedding(tokens, cost or 0.0)
                bucket = BUCKET_MEMORY_PREFRONTAL_EMBEDDINGS
            else:
                bucket = BUCKET_MEMORY_EMBEDDINGS

            tracker = get_cost_tracker()
            # Embeddings have no output tokens - all tokens are input
            tracker.add_external_call(
                bucket,
                cost=cost or 0.0,
                tokens=tokens,
                calls=1,
                input_tokens=tokens,
                output_tokens=0,
                input_cost=cost or 0.0,
                output_cost=0.0,
                duration=duration,
            )

        except Exception as e:
            logger.debug("Failed to meter embedding call: %s", e)


class MeteredLiteLLMSDKEmbeddings:
    """Wrapper for LiteLLMSDKEmbeddings that meters costs.

    Proxies the underlying ``_litellm`` module after initialization
to capture embedding usage. The dimension detection probe during
    ``initialize()`` is intentionally unmetered.
    """

    def __init__(
        self,
        *,
        base: LiteLLMSDKEmbeddings,
        cost_recorder: CerebralCostRecorder,
        max_input_chars: int | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._base = base
        self._recorder = cost_recorder
        self._proxy_set = False
        self._max_input_chars = max_input_chars
        self._extra_kwargs = extra_kwargs or {}

    async def initialize(self) -> None:
        """Initialize the base embeddings, then wrap _litellm."""
        # Run base initialization (this does the dimension probe)
        await self._base.initialize()

        # Wrap _litellm after initialization to meter actual retain/recall
        # calls, not the probe. Must check if already wrapped because
        # initialize() may be called multiple times.
        if not self._proxy_set and hasattr(self._base, "_litellm") and self._base._litellm is not None:
            self._base._litellm = _LiteLLMProxy(
                self._base._litellm,
                self._recorder,
                max_input_chars=self._max_input_chars,
                extra_kwargs=self._extra_kwargs,
            )
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


def resolve_embedding_input_limits(
    model: str, override: int | None = None
) -> tuple[int | None, dict[str, Any]]:
    """Resolve the maximum input characters and extra kwargs for embeddings.

    Bedrock Cohere embed v3 has a 2048 character limit that Hindsight does not
    handle. This function returns the cap and any extra kwargs (e.g., truncate)
    needed to stay within the limit.

    Args:
        model: The litellm model string (e.g., "bedrock/cohere.embed-multilingual-v3").
        override: User override for the character cap:
            - None (default): Auto-detect based on model (2048 for Bedrock Cohere v3).
            - 0: Disable character capping (use Hindsight's token-based limit).
            - N > 0: Use this specific character cap.

    Returns:
        Tuple of (max_input_chars, extra_kwargs):
        - max_input_chars: The character cap (None means no cap), or the override value.
        - extra_kwargs: Dict of extra kwargs to pass to the embedding call
          (e.g., {"truncate": "END"} for Cohere models).
    """
    # Handle override
    if override == 0:
        return None, {}
    if override is not None and override > 0:
        # For Bedrock Cohere models, still add truncate even with custom cap
        if re.search(r"bedrock/.*cohere\.embed-(english|multilingual)-v3", model):
            return override, {"truncate": "END"}
        return override, {}

    # Auto-detect based on model
    # Match Bedrock Cohere v3 models (including region-prefixed like eu.cohere...)
    if re.search(r"bedrock/.*cohere\.embed-(english|multilingual)-v3", model):
        return 2048, {"truncate": "END"}

    # Default: no character cap (Hindsight's token-based limit applies)
    return None, {}
