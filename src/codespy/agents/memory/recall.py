"""Recall monitoring: record what Prefrontal gave to the agent.

This module has no dependencies on hippocampus, prefrontal or cerebral,
so all of them can import it without circular import risks.
"""

from __future__ import annotations

import contextvars
import threading
import uuid
from collections.abc import Iterator, Set
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RecallRecord(BaseModel):
    """One Prefrontal recall (pre-call load or recall_memory tool call).

    Attributes:
        id: UUID identifying this recall record.
        run_id: Run identifier this recall belongs to.
        task: Consumer task name (e.g. "code_review", "review" for run-level load).
        kind: "load" for pre-call aload, "tool" for recall_memory.
        timestamp: UTC time the recall started.
        query: The query used (context facet for load, truncated query for tool).
        reach: Effective reach (local/org/bank; clamped for tool calls).
        reflects: The reflects config at call time.
        status: Outcome classification.
        text: Exact string returned to the agent ("" on load error).
        model: LLM model(s) that answered, comma-joined; "" = no LLM used.
        llm_calls: Number of LLM calls (reflect iterations + rewrite).
        input_tokens: LLM prompt tokens + embedding tokens.
        output_tokens: LLM completion tokens.
        input_cost: Cost in USD for input tokens.
        output_cost: Cost in USD for output tokens.
        latency_ms: Wall time from start to result, in milliseconds.
        details: JSON-serializable extra data per kind.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    run_id: str = Field(default="")
    task: str = Field(default="")
    kind: Literal["load", "tool"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    query: str = Field(default="")
    reach: str = Field(default="local")
    reflects: int = Field(default=0)
    status: Literal["ok", "empty", "limit", "error"] = Field(default="ok")
    text: str = Field(default="")
    model: str = Field(default="")
    llm_calls: int = Field(default=0)
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    input_cost: float = Field(default=0.0)
    output_cost: float = Field(default=0.0)
    latency_ms: int = Field(default=0)
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RecallUsage:
    """Thread-safe accumulator for one recall's usage.

    Used via `current_recall_usage` contextvar so that concurrent recalls
    (e.g., per-scope in asyncio.gather) don't mix their metrics.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    llm_calls: int = 0
    _models: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_llm(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        input_cost: float,
        output_cost: float,
    ) -> None:
        """Record one LLM call."""
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.input_cost += input_cost
            self.output_cost += output_cost
            self.llm_calls += 1
            if model:
                self._models.add(model)

    def add_embedding(self, tokens: int, cost: float) -> None:
        """Record one embedding call (counts as input only)."""
        with self._lock:
            self.input_tokens += tokens
            self.input_cost += cost

    @property
    def models(self) -> Set[str]:
        """Return the set of models used."""
        with self._lock:
            return set(self._models)

    def to_model_field(self) -> str:
        """Comma-joined models, empty if none."""
        with self._lock:
            return ",".join(sorted(self._models))


# Context variable to attribute LLM/embedding calls to the current recall.
current_recall_usage: contextvars.ContextVar[RecallUsage | None] = contextvars.ContextVar(
    "current_recall_usage", default=None
)


@contextmanager
def track_recall_usage() -> Iterator[RecallUsage]:
    """Set current_recall_usage for the duration of the context manager.

    Usage:
        with track_recall_usage() as usage:
            ...  # LLM calls here will add to usage
        # usage now holds the accumulated metrics
    """
    usage = RecallUsage()
    token = current_recall_usage.set(usage)
    try:
        yield usage
    finally:
        current_recall_usage.reset(token)
