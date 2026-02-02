"""Thread-safe cost tracking for LLM calls."""

import threading


class CostTracker:
    """Track LLM costs across multiple calls."""

    def __init__(self) -> None:
        """Initialize the cost tracker."""
        self._lock = threading.Lock()
        self._total_cost = 0.0
        self._total_tokens = 0
        self._call_count = 0
        self._last_call_cost = 0.0
        self._last_call_tokens = 0

    def reset(self) -> None:
        """Reset all tracking."""
        with self._lock:
            self._total_cost = 0.0
            self._total_tokens = 0
            self._call_count = 0
            self._last_call_cost = 0.0
            self._last_call_tokens = 0

    def add_call(self, cost: float, tokens: int) -> None:
        """Record a call's cost and tokens."""
        with self._lock:
            self._total_cost += cost
            self._total_tokens += tokens
            self._call_count += 1
            self._last_call_cost = cost
            self._last_call_tokens = tokens

    @property
    def total_cost(self) -> float:
        """Get total cost in USD."""
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self._total_tokens

    @property
    def call_count(self) -> int:
        """Get number of calls made."""
        return self._call_count

    @property
    def last_call_cost(self) -> float:
        """Get cost of last call."""
        return self._last_call_cost

    @property
    def last_call_tokens(self) -> int:
        """Get tokens of last call."""
        return self._last_call_tokens


# Global cost tracker instance
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    return _cost_tracker