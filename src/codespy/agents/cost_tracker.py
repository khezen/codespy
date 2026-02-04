"""Thread-safe cost tracking for LLM calls with per-module attribution.

Uses DSPy's internal LM history mechanism for reliable per-module attribution,
even during parallel execution with dspy.Parallel.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import dspy  # type: ignore[import-untyped]


@dataclass
class ModuleStats:
    """Statistics for a single module's LLM usage."""

    name: str
    cost: float = 0.0
    tokens: int = 0
    call_count: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds, or 0 if not completed."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "cost": self.cost,
            "tokens": self.tokens,
            "call_count": self.call_count,
            "duration_seconds": self.duration_seconds,
        }


class CostTracker:
    """Track LLM costs across multiple calls with per-module attribution.
    
    Uses DSPy's LM history for per-module tracking, which works reliably
    even during parallel execution.
    """

    def __init__(self) -> None:
        """Initialize the cost tracker."""
        self._lock = threading.Lock()
        self._module_stats: dict[str, ModuleStats] = {}

    def reset(self) -> None:
        """Reset all tracking."""
        with self._lock:
            self._module_stats.clear()

    def start_module(self, module_name: str) -> None:
        """Mark the start of a module's execution.
        
        Args:
            module_name: Name of the module starting execution
        """
        with self._lock:
            if module_name not in self._module_stats:
                self._module_stats[module_name] = ModuleStats(name=module_name)
            self._module_stats[module_name].start_time = time.time()
            self._module_stats[module_name].end_time = None

    def end_module(self, module_name: str, cost: float, tokens: int, call_count: int) -> None:
        """Mark the end of a module's execution with its costs.
        
        Args:
            module_name: Name of the module ending execution
            cost: Total cost for this module's LLM calls
            tokens: Total tokens used by this module
            call_count: Number of LLM calls made by this module
        """
        with self._lock:
            if module_name not in self._module_stats:
                self._module_stats[module_name] = ModuleStats(name=module_name)
            stats = self._module_stats[module_name]
            stats.end_time = time.time()
            stats.cost += cost
            stats.tokens += tokens
            stats.call_count += call_count

    @property
    def total_cost(self) -> float:
        """Get total cost in USD across all modules."""
        with self._lock:
            return sum(s.cost for s in self._module_stats.values())

    @property
    def total_tokens(self) -> int:
        """Get total tokens used across all modules."""
        with self._lock:
            return sum(s.tokens for s in self._module_stats.values())

    @property
    def call_count(self) -> int:
        """Get total number of LLM calls across all modules."""
        with self._lock:
            return sum(s.call_count for s in self._module_stats.values())

    def get_module_stats(self, module_name: str) -> Optional[ModuleStats]:
        """Get stats for a specific module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            ModuleStats or None if module not found
        """
        with self._lock:
            return self._module_stats.get(module_name)

    def get_all_module_stats(self) -> dict[str, ModuleStats]:
        """Get stats for all modules.
        
        Returns:
            Dictionary of module name to ModuleStats
        """
        with self._lock:
            # Return a copy to avoid concurrent modification issues
            return {k: ModuleStats(
                name=v.name,
                cost=v.cost,
                tokens=v.tokens,
                call_count=v.call_count,
                start_time=v.start_time,
                end_time=v.end_time,
            ) for k, v in self._module_stats.items()}


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


def _calculate_costs_from_entries(entries: list[dict], exclude_uuids: set[str]) -> tuple[float, int, int]:
    """Calculate costs from history entries, excluding specific UUIDs.
    
    Args:
        entries: List of history entries
        exclude_uuids: Set of UUIDs to exclude from calculation
        
    Returns:
        Tuple of (total_cost, total_tokens, call_count)
    """
    total_cost = 0.0
    total_tokens = 0
    call_count = 0
    
    for entry in entries:
        entry_uuid = entry.get("uuid", "")
        if entry_uuid and entry_uuid not in exclude_uuids:
            # Get cost
            cost = entry.get("cost")
            if cost is not None:
                total_cost += cost
            
            # Get tokens from usage
            usage = entry.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                completion_tokens = usage.get("completion_tokens", 0) or 0
                total_tokens += prompt_tokens + completion_tokens
            
            call_count += 1
    
    return total_cost, total_tokens, call_count


class ModuleContext:
    """Context manager for tracking module execution.
    
    Uses DSPy's LM history mechanism to track costs reliably, even during
    parallel execution with dspy.Parallel. Works by:
    1. Recording history UUIDs before module execution
    2. After execution, finding new entries (by UUID)
    3. Summing costs/tokens from new entries
    
    Usage:
        with ModuleContext("bug_detector", cost_tracker):
            # All LLM calls here will be attributed to bug_detector
            result = await agent.acall(...)
    """
    
    def __init__(self, module_name: str, tracker: "CostTracker") -> None:
        """Initialize the module context.
        
        Args:
            module_name: Name of the module
            tracker: CostTracker instance
        """
        self.module_name = module_name
        self.tracker = tracker
        self._before_uuids: set[str] = set()

    def __enter__(self) -> "ModuleContext":
        """Enter the context, capturing current history state."""
        # Capture UUIDs of entries that exist before module execution
        self._before_uuids = _get_history_uuids()
        self.tracker.start_module(self.module_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context, calculating costs from new history entries."""
        # Get all current entries and calculate costs from new ones
        entries = _get_history_entries()
        cost, tokens, call_count = _calculate_costs_from_entries(entries, self._before_uuids)
        
        self.tracker.end_module(self.module_name, cost, tokens, call_count)

    async def __aenter__(self) -> "ModuleContext":
        """Async enter the context."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async exit the context."""
        self.__exit__(exc_type, exc_val, exc_tb)


# Global cost tracker instance
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    return _cost_tracker