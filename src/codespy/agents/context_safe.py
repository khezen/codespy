"""ContextSafe wrapper for context window overflow resilience.

Provides transparent fallback to RLM (Recursive Language Model) when inputs
exceed the model's context window. Used to wrap all DSPy signature modules.
"""

import enum
import logging
import re

import dspy  # type: ignore[import-untyped]
import litellm  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Regex for detecting context window overflow in error messages
_RE_CONTEXT_LENGTH = re.compile(r"maximum context length is \d+ tokens", re.IGNORECASE)

# Minimum input tokens before proactive threshold applies.
# Below this, context rot is not a concern regardless of ratio.
_MIN_RLM_THRESHOLD_TOKENS = 8192

# Replaces dspy.RLM's stock `reasoning` desc ("Think step-by-step: what do you
# know? What remains? Plan your next action."), which on its own triggers Bedrock
# content_filter on Claude Opus 5.5.
# NOTE: This override retires itself once dspy ships PR #10643 (expected 3.4.2 or 3.5),
# which changes the field to `dspy.Reasoning` with a safer fallback plan. Once the
# minimum dspy includes that fix, delete this override and its tests.
RLM_REASONING_DESC = "Short plan for the next step."

# Stock dspy RLM reasoning description that triggers Bedrock content_filter.
# See dspy PR #10643 for the upstream fix.
_STOCK_RLM_REASONING_DESC = "Think step-by-step: what do you know? What remains? Plan your next action."


def _plain(value):  # type: ignore[no-untyped-def]
    """Replace Enum members with their values, recursively.

    dspy's PythonInterpreter injects inputs as `name = repr(value)`; an Enum repr
    (`<IssueCategory.BUG: 'bug'>`) is not valid Python, so every REPL execution
    fails with an empty SyntaxError.
    """
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {_plain(k): _plain(v) for k, v in value.items()}
    return value


def is_content_filter_refusal(exc: BaseException) -> bool:
    """Whether ``exc`` (or its cause chain) is a provider ``content_filter`` refusal."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if "content_filter" in str(cur):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


REFUSAL_EXTRACT_REASONING = "Extract after content_filter refusal"


class CodespyRLM(dspy.RLM):
    """dspy.RLM adjusted for codespy.

    - Neutral action-signature `reasoning` description (Bedrock content_filter).
    - Enum inputs passed as plain values (REPL variable injection).
    - A content_filter refusal on an action step ends the loop through dspy's
      extract step on the history so far, instead of discarding every verified
      finding. Bedrock's classifier flips on byte-level changes of a long REPL
      history (one ``Output (`` -> ``Output: (`` edit clears it), so no wording
      avoids it reliably. Not a retry: the refused step is not re-sent.
    """

    def forward(self, **kwargs):  # type: ignore[no-untyped-def]
        return super().forward(**{k: _plain(v) for k, v in kwargs.items()})

    async def aforward(self, **kwargs):  # type: ignore[no-untyped-def]
        return await super().aforward(**{k: _plain(v) for k, v in kwargs.items()})

    def _refusal_warning(self, iteration: int, history) -> None:  # type: ignore[no-untyped-def]
        logger.warning(
            "RLM action step %d/%d refused (content_filter); "
            "extracting outputs from %d REPL steps",
            iteration + 1, self.max_iters, len(history),
        )

    @staticmethod
    def _mark_refusal(pred: dspy.Prediction) -> dspy.Prediction:
        pred.final_reasoning = REFUSAL_EXTRACT_REASONING
        return pred

    def _execute_iteration(  # type: ignore[no-untyped-def]
        self, repl, variables, history, iteration, input_args, output_field_names, *args, **kw
    ):
        try:
            return super()._execute_iteration(
                repl, variables, history, iteration, input_args, output_field_names, *args, **kw
            )
        except Exception as exc:
            if not is_content_filter_refusal(exc) or not len(history):
                raise
            self._refusal_warning(iteration, history)
            pred = self._extract_fallback(variables, history, output_field_names)
            return self._mark_refusal(pred)

    async def _aexecute_iteration(  # type: ignore[no-untyped-def]
        self, repl, variables, history, iteration, input_args, output_field_names, *args, **kw
    ):
        try:
            return await super()._aexecute_iteration(
                repl, variables, history, iteration, input_args, output_field_names, *args, **kw
            )
        except Exception as exc:
            if not is_content_filter_refusal(exc) or not len(history):
                raise
            self._refusal_warning(iteration, history)
            pred = await self._aextract_fallback(variables, history, output_field_names)
            return self._mark_refusal(pred)

    def _build_signatures(self):  # type: ignore[no-untyped-def]
        action_sig, extract_sig = super()._build_signatures()
        if "reasoning" in action_sig.output_fields:
            # Only override the stock description that triggers Bedrock content_filter.
            # Once dspy PR #10643 ships (native reasoning with fallback), this check
            # becomes a no-op and the override can be removed.
            current_desc = action_sig.output_fields["reasoning"].json_schema_extra.get("desc", "")
            if _STOCK_RLM_REASONING_DESC in current_desc:
                action_sig = action_sig.with_updated_fields("reasoning", desc=RLM_REASONING_DESC)
        return action_sig, extract_sig


def estimate_context_overflow(model: str, max_tokens: int, input_text: str) -> bool:
    """Estimate whether input + max_tokens would exceed the model's context window.

    Returns True if overflow is likely. Returns False if no overflow expected
    or if the model is not in litellm's DB (can't estimate).
    """
    safety_margin = 4096
    try:
        info = litellm.get_model_info(model)
        max_input = info.get("max_input_tokens") or 0
        if not max_input:
            return False
        estimated_input = litellm.token_counter(model=model, text=input_text)
        return (estimated_input + max_tokens + safety_margin) > max_input
    except Exception:
        return False


def is_context_overflow_error(exc: Exception) -> bool:
    """Detect context window overflow from any source.

    Handles two cases:
    - Known models: dspy raises dspy.ContextWindowExceededError
    - Unknown models (nvidia/zai Bedrock): litellm raises BadRequestError,
      dspy wraps as LMInvalidRequestError — detected via error message regex.
    """
    if isinstance(exc, dspy.ContextWindowExceededError):
        return True
    return bool(_RE_CONTEXT_LENGTH.search(str(exc)))


class ContextSafe(dspy.Module):
    """Wraps a dspy.Module and falls back to RLM on context window overflow.

    Two-layer defense:
    - Pre-flight: estimates input tokens via litellm; if overflow is predicted
      for a known model, skips the inner module and uses RLM directly.
    - Try/catch: if the inner module raises a context overflow error (unknown
      models where pre-flight can't estimate), catches it and retries with RLM.

    Transparent to signature injection — delegates .signature to inner module.
    """

    def __init__(
        self,
        module: dspy.Module,
        signature,
        tools: list | None = None,
        name: str = "",
        max_iters: int | None = None,
        max_llm_calls: int | None = None,
        rlm_threshold: float = 1.0,
    ):
        super().__init__()
        self.module = module
        self._orig_signature = signature
        self._tools = tools
        self._name = name or signature.__name__
        self._max_iters = max_iters
        self._max_llm_calls = max_llm_calls
        self._rlm_threshold = rlm_threshold

    @property
    def signature(self):
        return getattr(self.module, "signature", None)

    @signature.setter
    def signature(self, value):
        self.module.signature = value

    def forward(self, **kwargs) -> dspy.Prediction:
        should_fallback, reason = self._should_use_rlm(kwargs)
        if should_fallback:
            logger.warning(
                "ContextSafe[%s]: %s for model=%s; falling back to RLM",
                self._name,
                reason,
                getattr(dspy.settings.lm, "model", "unknown"),
            )
            return self._create_rlm_fallback()(**kwargs)

        try:
            return self.module(**kwargs)
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise
            logger.warning(
                "ContextSafe[%s]: context window overflow on model=%s; "
                "falling back to RLM (error: %s)",
                self._name,
                getattr(dspy.settings.lm, "model", "unknown"),
                str(exc)[:200],
            )
            return self._create_rlm_fallback()(**kwargs)

    async def aforward(self, **kwargs) -> dspy.Prediction:
        """Async path — used by code_review, scope, supply_chain, doc via agent.acall()."""
        should_fallback, reason = self._should_use_rlm(kwargs)
        if should_fallback:
            logger.warning(
                "ContextSafe[%s]: %s for model=%s; falling back to RLM",
                self._name,
                reason,
                getattr(dspy.settings.lm, "model", "unknown"),
            )
            return await self._create_rlm_fallback().acall(**kwargs)

        try:
            return await self.module.acall(**kwargs)
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise
            logger.warning(
                "ContextSafe[%s]: context window overflow on model=%s; "
                "falling back to RLM (error: %s)",
                self._name,
                getattr(dspy.settings.lm, "model", "unknown"),
                str(exc)[:200],
            )
            return await self._create_rlm_fallback().acall(**kwargs)

    def _should_use_rlm(self, kwargs: dict) -> tuple[bool, str]:
        """Check if RLM should be used (proactive threshold or hard overflow).

        Returns (should_fallback, reason) for logging.
        """
        try:
            lm = dspy.settings.lm
            if lm is None:
                return False, ""
            model = lm.model
            max_tokens = lm.kwargs.get("max_tokens") or 0

            info = litellm.get_model_info(model)
            max_input = info.get("max_input_tokens") or 0
            if not max_input:
                return False, ""

            input_text = "\n".join(str(v) for v in kwargs.values())
            estimated_input = litellm.token_counter(model=model, text=input_text)

            # Layer 1: Proactive threshold (context rot prevention)
            # Only applies when input exceeds the minimum floor (8192 tokens) —
            # below that, context rot is not a concern.
            if self._rlm_threshold < 1.0 and estimated_input >= _MIN_RLM_THRESHOLD_TOKENS:
                threshold_tokens = int(max_input * self._rlm_threshold)
                if estimated_input > threshold_tokens:
                    return True, (
                        f"context rot threshold exceeded "
                        f"({estimated_input} > {threshold_tokens} = "
                        f"{self._rlm_threshold:.0%} of {max_input})"
                    )

            # Layer 2: Hard overflow check (existing safety net)
            safety_margin = 4096
            if max_tokens and (estimated_input + max_tokens + safety_margin) > max_input:
                return True, (
                    f"context overflow predicted "
                    f"({estimated_input} + {max_tokens} + {safety_margin} > {max_input})"
                )

            return False, ""
        except Exception:
            return False, ""

    def _get_current_signature(self):
        """Get current signature."""
        sig = getattr(self.module, "signature", None)
        if sig is not None:
            return sig
        preds = list(self.module.named_predictors())
        if preds:
            return preds[0][1].signature
        return self._orig_signature

    def _create_rlm_fallback(self) -> dspy.RLM:
        """Create RLM with current signature and tools."""
        return CodespyRLM(
            self._get_current_signature(),
            tools=self._tools,
            max_iters=self._max_iters,
            max_llm_calls=self._max_llm_calls,
            verbose=logger.isEnabledFor(logging.DEBUG),
        )
