"""Thread-safe cost tracking for LLM calls with per-signature attribution.

Uses DSPy's internal LM history mechanism for reliable per-signature attribution,
even during parallel execution with dspy.Parallel.
"""

import logging
import sys
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import dspy  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class SignatureStats:
    """Statistics for a single signature's LLM usage."""

    name: str
    cost: float = 0.0
    tokens: int = 0
    call_count: int = 0
    start_time: float | None = None
    end_time: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    external_duration_seconds: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds, or 0 if not completed.

        For signatures tracked via SignatureContext, this is wall-clock time
        from start to end. For external calls (e.g., cerebral), this is the
        sum of per-call work durations (not wall-clock), consistent with how
        cost/tokens are summed.
        """
        wall = 0.0
        if self.start_time is not None:
            end = self.end_time if self.end_time is not None else time.time()
            wall = end - self.start_time
        return wall + self.external_duration_seconds

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "cost": self.cost,
            "tokens": self.tokens,
            "call_count": self.call_count,
            "duration_seconds": self.duration_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "external_duration_seconds": self.external_duration_seconds,
        }


class CostTracker:
    """Track LLM costs across multiple calls with per-signature attribution.

    Uses DSPy's LM history for per-signature tracking, which works reliably
    even during parallel execution.
    """

    def __init__(self) -> None:
        """Initialize the cost tracker."""
        self._lock = threading.Lock()
        self._signature_stats: dict[str, SignatureStats] = {}

    def reset(self) -> None:
        """Reset all tracking."""
        with self._lock:
            self._signature_stats.clear()

    def start_signature(self, signature_name: str) -> None:
        """Mark the start of a signature's execution.

        Args:
            signature_name: Name of the signature starting execution
        """
        with self._lock:
            if signature_name not in self._signature_stats:
                self._signature_stats[signature_name] = SignatureStats(name=signature_name)
            self._signature_stats[signature_name].start_time = time.time()
            self._signature_stats[signature_name].end_time = None

    def end_signature(
        self,
        signature_name: str,
        cost: float,
        tokens: int,
        call_count: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
    ) -> None:
        """Mark the end of a signature's execution with its costs.

        Args:
            signature_name: Name of the signature ending execution
            cost: Total cost for this signature's LLM calls
            tokens: Total tokens used by this signature
            call_count: Number of LLM calls made by this signature
            input_tokens: Input/prompt tokens used
            output_tokens: Output/completion tokens used
            input_cost: Cost for input tokens
            output_cost: Cost for output tokens
        """
        with self._lock:
            if signature_name not in self._signature_stats:
                self._signature_stats[signature_name] = SignatureStats(name=signature_name)
            stats = self._signature_stats[signature_name]
            stats.end_time = time.time()
            stats.cost += cost
            stats.tokens += tokens
            stats.call_count += call_count
            stats.input_tokens += input_tokens
            stats.output_tokens += output_tokens
            stats.input_cost += input_cost
            stats.output_cost += output_cost

    def add_external_call(
        self,
        name: str,
        cost: float,
        tokens: int,
        calls: int = 1,
        input_tokens: int = 0,
        output_tokens: int = 0,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        duration: float = 0.0,
    ) -> None:
        """Accumulate costs from external LLM calls (not tracked by DSPy).

        Creates a SignatureStats entry if missing, without touching
        start_time/end_time. Duration is summed per-call work time for the
        bucket (not wall-clock), consistent with how cost/tokens are summed.

        Args:
            name: Bucket name for the external call (e.g., "cerebral_retain")
            cost: Cost in USD for the call(s)
            tokens: Total tokens used
            calls: Number of calls (default 1)
            input_tokens: Input/prompt tokens used
            output_tokens: Output/completion tokens used
            input_cost: Cost for input tokens
            output_cost: Cost for output tokens
            duration: Duration in seconds for this call (default 0.0)
        """
        with self._lock:
            if name not in self._signature_stats:
                self._signature_stats[name] = SignatureStats(name=name)
            stats = self._signature_stats[name]
            stats.cost += cost
            stats.tokens += tokens
            stats.call_count += calls
            stats.input_tokens += input_tokens
            stats.output_tokens += output_tokens
            stats.input_cost += input_cost
            stats.output_cost += output_cost
            stats.external_duration_seconds += duration

    @property
    def total_cost(self) -> float:
        """Get total cost in USD across all signatures."""
        with self._lock:
            return sum(s.cost for s in self._signature_stats.values())

    @property
    def total_tokens(self) -> int:
        """Get total tokens used across all signatures."""
        with self._lock:
            return sum(s.tokens for s in self._signature_stats.values())

    @property
    def call_count(self) -> int:
        """Get total number of LLM calls across all signatures."""
        with self._lock:
            return sum(s.call_count for s in self._signature_stats.values())

    def get_signature_stats(self, signature_name: str) -> SignatureStats | None:
        """Get stats for a specific signature.

        Args:
            signature_name: Name of the signature

        Returns:
            SignatureStats or None if signature not found
        """
        with self._lock:
            return self._signature_stats.get(signature_name)

    def get_all_signature_stats(self) -> dict[str, SignatureStats]:
        """Get stats for all signatures.

        Returns:
            Dictionary of signature name to SignatureStats
        """
        with self._lock:
            # Return a copy to avoid concurrent modification issues
            return {
                k: SignatureStats(
                    name=v.name,
                    cost=v.cost,
                    tokens=v.tokens,
                    call_count=v.call_count,
                    start_time=v.start_time,
                    end_time=v.end_time,
                    input_tokens=v.input_tokens,
                    output_tokens=v.output_tokens,
                    input_cost=v.input_cost,
                    output_cost=v.output_cost,
                    external_duration_seconds=v.external_duration_seconds,
                )
                for k, v in self._signature_stats.items()
            }


def _get_history_entries() -> list[dict]:
    """Get current LM history entries from DSPy.

    Returns:
        List of history entries, or empty list if LM not configured
    """
    try:
        lm = dspy.settings.lm
        if lm is not None and hasattr(lm, "history"):
            return lm.history
    except Exception:
        pass
    return []


def _get_history_uuids() -> set[str]:
    """Get UUIDs of current history entries.

    Returns:
        Set of UUIDs from current history
    """
    entries = _get_history_entries()
    return {entry.get("uuid", "") for entry in entries if entry.get("uuid")}


def _as_number(value: object) -> float:
    """Coerce a history field to a number, yielding 0.0 for anything unusable.

    History entries are raw provider/LiteLLM payloads, so ``cost`` and the
    ``usage`` counters are only *conventionally* numeric. Coercing here keeps a
    provider-shape surprise from turning accounting into an exception.

    Args:
        value: A value read from an LM history entry.

    Returns:
        The value as a float, or 0.0 if it is missing or not numeric.
    """
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:  # Some providers report numbers as strings.
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _calculate_costs_from_entries(
    entries: list[dict], exclude_uuids: set[str]
) -> tuple[float, int, int, int, int, float, float]:
    """Calculate costs from history entries, excluding specific UUIDs.

    Every field is read defensively: cost accounting is observability, so a
    malformed entry degrades that entry to zero rather than failing the review
    that produced it.

    Args:
        entries: List of history entries
        exclude_uuids: Set of UUIDs to exclude from calculation

    Returns:
        Tuple of (total_cost, total_tokens, call_count, input_tokens, output_tokens,
                  input_cost, output_cost)
    """
    total_cost = 0.0
    total_tokens = 0
    call_count = 0
    input_tokens = 0
    output_tokens = 0
    input_cost = 0.0
    output_cost = 0.0

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        entry_uuid = entry.get("uuid", "")
        if entry_uuid and entry_uuid not in exclude_uuids:
            entry_cost = _as_number(entry.get("cost"))
            total_cost += entry_cost

            # Get tokens from usage
            usage = entry.get("usage")
            entry_input_tokens = 0
            entry_output_tokens = 0
            if isinstance(usage, dict):
                entry_input_tokens = int(_as_number(usage.get("prompt_tokens")))
                entry_output_tokens = int(_as_number(usage.get("completion_tokens")))
                input_tokens += entry_input_tokens
                output_tokens += entry_output_tokens
                total_tokens += entry_input_tokens + entry_output_tokens

            # Calculate split costs using litellm if model is available
            entry_input_cost, entry_output_cost = _calculate_split_cost(
                entry, entry_input_tokens, entry_output_tokens, entry_cost
            )
            input_cost += entry_input_cost
            output_cost += entry_output_cost

            call_count += 1

    return total_cost, total_tokens, call_count, input_tokens, output_tokens, input_cost, output_cost


def _calculate_split_cost(
    entry: dict, prompt_tokens: int, completion_tokens: int, fallback_cost: float
) -> tuple[float, float]:
    """Calculate split input/output cost for a history entry.

    Uses litellm.cost_per_token if model is available, otherwise falls back
    to proportional split of the fallback_cost.

    Args:
        entry: History entry dict
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        fallback_cost: Fallback total cost for proportional split

    Returns:
        Tuple of (input_cost, output_cost)
    """
    model = entry.get("model")
    if not model:
        # Fall back to proportional split by token ratio
        return _proportional_split(prompt_tokens, completion_tokens, fallback_cost)

    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return float(prompt_cost), float(completion_cost)
    except Exception:
        # Model may not have pricing or be available; use proportional split
        return _proportional_split(prompt_tokens, completion_tokens, fallback_cost)


def _proportional_split(input_tokens: int, output_tokens: int, total_cost: float) -> tuple[float, float]:
    """Split cost proportionally by token count.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_cost: Total cost to split

    Returns:
        Tuple of (input_cost, output_cost)
    """
    total = input_tokens + output_tokens
    if total == 0:
        return 0.0, 0.0
    input_ratio = input_tokens / total
    return total_cost * input_ratio, total_cost * (1 - input_ratio)


class SignatureContext:
    """Context manager scoping one named unit of LLM work.

    Two responsibilities, both keyed off the same name:

    1. **LM selection** — applies the model, temperature, and reasoning effort
       configured for this name (``signatures.<name>`` or ``memory.<name>``),
       falling back to the top-level defaults.
    2. **Cost attribution** — uses DSPy's LM history to attribute costs
       reliably, even during parallel execution with dspy.Parallel, by
       recording history UUIDs on entry and summing only the new entries.

    Usage:
        with SignatureContext("code_review", cost_tracker):
            # Runs on code_review's configured model, costs attributed to it
            result = await agent.acall(...)
    """

    def __init__(self, signature_name: str, tracker: "CostTracker") -> None:
        """Initialize the signature context.

        Args:
            signature_name: Name of the signature
            tracker: CostTracker instance
        """
        self.signature_name = signature_name
        self.tracker = tracker
        self._before_uuids: set[str] = set()
        # Annotated so the None default doesn't narrow the attribute to
        # ``None``, which would hide the enter/exit calls from type checking.
        self._lm_context: AbstractContextManager[Any] | None = None

    def __enter__(self) -> "SignatureContext":
        """Enter the context, applying the LM and capturing history state.

        If the bookkeeping that follows the LM swap fails, the swap is rolled
        back before propagating: Python does not call ``__exit__`` when
        ``__enter__`` raises, so without this the overridden LM would stay
        installed for the rest of the thread.
        """
        # Imported here to avoid a circular import at module load time
        # (dspy_config imports codespy.config, which must not import agents).
        from codespy.agents.dspy_config import lm_context

        # Apply this name's LM first, so the history we snapshot below belongs
        # to the LM that will actually serve the enclosed calls.
        self._lm_context = lm_context(self.signature_name)
        self._lm_context.__enter__()
        try:
            self._before_uuids = _get_history_uuids()
            self.tracker.start_signature(self.signature_name)
        except BaseException:
            self._lm_context.__exit__(*sys.exc_info())
            self._lm_context = None
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context, calculating costs from new history entries.

        Cost calculation failures are logged at WARNING level but never
        propagated — bookkeeping must not mask application errors.

        The LM context is released in a ``finally``: a leaked
        ``dspy.context`` does not raise, it silently leaves the overridden LM
        installed for the remainder of the thread, so every later predictor
        would run on the wrong model and be attributed to the wrong signature.
        Guaranteeing the exit keeps an accounting failure loud and local
        instead of quiet and global.
        """
        try:
            # Read history before leaving the LM context, so dspy.settings.lm
            # still points at the LM whose history we need.
            entries = _get_history_entries()
            cost, tokens, call_count, input_tokens, output_tokens, input_cost, output_cost = _calculate_costs_from_entries(
                entries, self._before_uuids
            )
            self.tracker.end_signature(
                self.signature_name,
                cost=cost,
                tokens=tokens,
                call_count=call_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
            )
        except Exception as e:
            logger.warning("Cost calculation failed for %s: %s", self.signature_name, e)
        finally:
            if self._lm_context is not None:
                self._lm_context.__exit__(exc_type, exc_val, exc_tb)
                self._lm_context = None

    async def __aenter__(self) -> "SignatureContext":
        """Async enter the context."""
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async exit the context."""
        self.__exit__(exc_type, exc_val, exc_tb)


# Global cost tracker instance
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    return _cost_tracker
