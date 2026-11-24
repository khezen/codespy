"""ContextSafe wrapper for context window overflow resilience.

Provides transparent fallback to RLM (Recursive Language Model) when inputs
exceed the model's context window. Used to wrap all DSPy signature modules.
"""

import enum
import logging
import re
import threading
from collections.abc import Callable

import dspy  # type: ignore[import-untyped]
import litellm  # type: ignore[import-untyped]
from dspy.primitives.code_interpreter import FinalOutput  # type: ignore[import-untyped]
from dspy.primitives.repl_types import REPLHistory  # type: ignore[import-untyped]

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

# Prefix of the REPL output that sends an early SUBMIT back to the model.
COVERAGE_REJECTION_PREFIX = "[Coverage]"

# Called with the REPL history on a valid SUBMIT; returns a message to send the
# SUBMIT back (loop continues), or None to accept it.
SubmitCheck = Callable[[REPLHistory], "str | None"]

# dspy.RLM's default ``max_output_chars`` (kept when codespy passes None).
DSPY_DEFAULT_MAX_OUTPUT_CHARS = 10_000


class _CountingBudget:
    """Wraps dspy's per-forward sub-LLM budget and counts accepted charges.

    ``reserve(n)`` delegates first: a charge dspy refuses (limit exceeded)
    raises before it is counted. dspy does not refund failed LM calls, and
    neither does this count.
    """

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.calls = 0
        self._lock = threading.Lock()

    def reserve(self, n: int = 1) -> None:
        self.inner.reserve(n)  # type: ignore[attr-defined]
        with self._lock:
            self.calls += n


class CodespyRLM(dspy.RLM):
    """dspy.RLM adjusted for codespy.

    - Neutral action-signature `reasoning` description (Bedrock content_filter).
    - Enum inputs passed as plain values (REPL variable injection).
    - A content_filter refusal on an action step ends the loop through dspy's
      extract step on the history so far, instead of discarding every verified
      finding. Bedrock's classifier flips on byte-level changes of a long REPL
      history (one ``Output (`` -> ``Output: (`` edit clears it), so no wording
      avoids it reliably. Not a retry: the refused step is not re-sent.
    - Configurable ``max_output_chars`` (cap of one REPL step output shown to the
      model; ``DSPY_DEFAULT_MAX_OUTPUT_CHARS`` is kept when ``None``).
    - Optional SUBMIT gate (``submit_check``): while it reports a message (e.g.
      unseen changed files) and budget remains, a valid SUBMIT is sent back to
      the model with that message and the loop continues. Once the budget is
      spent (last iteration, or ``max_llm_calls`` exhausted) SUBMIT is accepted.
      No rejection cap: the loop is bounded by ``max_iters``.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        *args,
        max_output_chars: int | None = None,
        submit_check: SubmitCheck | None = None,
        **kwargs,
    ):
        if max_output_chars is not None:
            kwargs["max_output_chars"] = max_output_chars
        super().__init__(*args, **kwargs)
        self._submit_check = submit_check
        self.coverage_rejections = 0
        self._iteration = 0
        self._budget_proxy: _CountingBudget | None = None

    def _reset_gate_state(self) -> None:
        self.coverage_rejections = 0
        self._iteration = 0
        self._budget_proxy = None

    def forward(self, **kwargs):  # type: ignore[no-untyped-def]
        self._reset_gate_state()
        return super().forward(**{k: _plain(v) for k, v in kwargs.items()})

    async def aforward(self, **kwargs):  # type: ignore[no-untyped-def]
        self._reset_gate_state()
        return await super().aforward(**{k: _plain(v) for k, v in kwargs.items()})

    # --- SUBMIT gate -------------------------------------------------------

    # dspy 3.4.1 hands the per-forward sub-LLM budget to these two methods, and
    # every charge goes through its `reserve(n)` (llm_query/llm_query_batched and
    # sandbox sub-agents). Passing the same counting proxy to both keeps an exact
    # count without reading the budget's fields (pinned by tests/test_rlm_coverage.py).

    def _counting(self, budget: object | None) -> "_CountingBudget | None":
        if budget is None:
            return None
        proxy = self._budget_proxy
        if proxy is None or proxy.inner is not budget:
            proxy = self._budget_proxy = _CountingBudget(budget)
        return proxy

    def _make_llm_tools(self, budget=None, max_workers: int = 8):  # type: ignore[no-untyped-def]
        return super()._make_llm_tools(self._counting(budget), max_workers)

    def _setup_facade(self, repl, budget, interpreter_factory):  # type: ignore[no-untyped-def]
        return super()._setup_facade(repl, self._counting(budget), interpreter_factory)

    def _llm_budget_spent(self) -> bool:
        proxy = self._budget_proxy
        limit = self.max_llm_calls
        return proxy is not None and limit is not None and proxy.calls >= limit

    def _budget_spent(self) -> bool:
        return self._iteration + 1 >= self.max_iters or self._llm_budget_spent()

    def _run_submit_check(self, history) -> str | None:  # type: ignore[no-untyped-def]
        if self._submit_check is None:
            return None
        try:
            message = self._submit_check(history)
        except Exception:
            logger.warning("RLM submit_check failed; accepting SUBMIT", exc_info=True)
            return None
        if not message:
            return None
        if not message.startswith(COVERAGE_REJECTION_PREFIX):
            message = f"{COVERAGE_REJECTION_PREFIX} {message}"
        return message

    def _process_execution_result(  # type: ignore[no-untyped-def]
        self, pred, code, result, history, output_field_names
    ):
        if self._submit_check is not None and isinstance(result, FinalOutput):
            _, error = self._process_final_output(result, output_field_names)
            if error is None:
                message = self._run_submit_check(history)
                if message:
                    if self._budget_spent():
                        logger.warning(
                            "RLM SUBMIT accepted at step %d/%d with budget spent; %s",
                            self._iteration + 1, self.max_iters,
                            message[len(COVERAGE_REJECTION_PREFIX):].strip(),
                        )
                    else:
                        self.coverage_rejections += 1
                        logger.info(
                            "RLM SUBMIT rejected at step %d/%d (rejection %d): %s",
                            self._iteration + 1, self.max_iters, self.coverage_rejections,
                            message[len(COVERAGE_REJECTION_PREFIX):].strip(),
                        )
                        return history.append(reasoning=pred.reasoning, code=code, output=message)
        return super()._process_execution_result(pred, code, result, history, output_field_names)

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
        self._iteration = iteration
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
        self._iteration = iteration
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
        max_output_chars: int | None = None,
        submit_check: SubmitCheck | None = None,
    ):
        super().__init__()
        self.module = module
        self._orig_signature = signature
        self._tools = tools
        self._name = name or signature.__name__
        self._max_iters = max_iters
        self._max_llm_calls = max_llm_calls
        self._rlm_threshold = rlm_threshold
        self._max_output_chars = max_output_chars
        self._submit_check = submit_check

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
            max_output_chars=self._max_output_chars,
            submit_check=self._submit_check,
            verbose=logger.isEnabledFor(logging.DEBUG),
        )
