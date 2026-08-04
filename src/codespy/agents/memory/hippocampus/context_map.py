from __future__ import annotations

import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ItemTag(str, Enum):
    """How a context-map item performed in the trajectory just observed.

    - helpful: directly aided orientation or answering; keep.
    - harmful: misled the agent or contradicted observations; remove.
    - neutral: present but unused this round; keep with no boost.
    - stale:   no longer reflects the external context; remove.
    """

    HELPFUL = "helpful"
    HARMFUL = "harmful"
    NEUTRAL = "neutral"
    STALE = "stale"


class OpType(str, Enum):
    """Cartographer edit operations against the context map."""

    ADD = "ADD"
    DELETE = "DELETE"
    REPLACE = "REPLACE"


SectionName = Literal[
    "context_roadmap",
    "context_understanding",
    "domain_constants",
    "parsing_schema",
    "reusable_results",
]

# Abbreviated prefixes
_SECTION_PREFIX: dict[str, str] = {
    "context_roadmap": "cr",
    "context_understanding": "cu",
    "domain_constants": "dc",
    "parsing_schema": "ps",
    "reusable_results": "rr",
}


class Item(BaseModel):
    id: str
    content: str


class CacheCandidate(BaseModel):
    section: SectionName
    value: str = Field(
        description="Compact candidate cache item, within the max_context_item_tokens budget."
    )
    transferability: str = Field(description="Kinds of future questions this would help.")
    rationale: str = Field(description="Why this is shared understanding, not a one-off fact.")


class Operation(BaseModel):
    type: OpType
    section: SectionName | None = Field(default=None, description="Required for ADD.")
    item_id: str | None = Field(default=None, description="Required for DELETE / REPLACE.")
    content: str | None = Field(default=None, description="Required for ADD / REPLACE.")


class ContextMap(BaseModel):
    context_roadmap: list[Item] = Field(
        default_factory=list,
        description="Index of what the context contains and where to find it",
    )
    context_understanding: list[Item] = Field(
        default_factory=list,
        description="High-level understanding of the context",
    )
    domain_constants: list[Item] = Field(
        default_factory=list,
        description=(
            "Exact parameters, formulas, thresholds, reference values, "
            "enum sets, and output field requirements"
        ),
    )
    parsing_schema: list[Item] = Field(
        default_factory=list,
        description=(
            "How to parse and navigate the context's format: "
            "delimiters, boundary patterns, field structure"
        ),
    )
    reusable_results: list[Item] = Field(
        default_factory=list,
        description=(
            "Agent-derived aggregated outputs (counts, distributions, classifications) "
            "that multiple questions would need"
        ),
    )

    @classmethod
    def section_names(cls) -> list[str]:
        return list(cls.model_fields)

    def section(self, name: str) -> list[Item]:
        return getattr(self, name)

    def all_items(self) -> list[Item]:
        return [it for s in self.section_names() for it in self.section(s)]

    def ids(self) -> set[str]:
        return {it.id for it in self.all_items()}

    def render(self) -> str:
        lines: list[str] = []
        for sec in self.section_names():
            info = type(self).model_fields[sec]
            items = self.section(sec)
            lines.append(f"## {sec.upper().replace('_', ' ')}")
            if items:
                lines.extend(f"[{it.id}] {it.content}" for it in items)
            else:
                lines.append(f"({info.description})")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def apply(self, ops: list[Operation]) -> tuple[ContextMap, list[str]]:
        """Return (new map, ids of newly-added items)."""
        cm = self.model_copy(deep=True)
        new_ids: list[str] = []
        for op in ops:
            if op.type == OpType.DELETE and op.item_id:
                for sec in cm.section_names():
                    lst = cm.section(sec)
                    lst[:] = [it for it in lst if it.id != op.item_id]
            elif op.type == OpType.REPLACE and op.item_id and op.content:
                for sec in cm.section_names():
                    lst = cm.section(sec)
                    for i, it in enumerate(lst):
                        if it.id == op.item_id:
                            lst[i] = Item(id=it.id, content=op.content)
            elif op.type == OpType.ADD and op.section and op.content:
                prefix = _SECTION_PREFIX.get(op.section, op.section[:2])
                new_id = f"{prefix}-{uuid.uuid4().hex}"
                cm.section(op.section).append(Item(id=new_id, content=op.content))
                new_ids.append(new_id)
        return cm, new_ids

    def without(self, ids: set[str]) -> ContextMap:
        cm = self.model_copy(deep=True)
        for sec in cm.section_names():
            lst = cm.section(sec)
            lst[:] = [it for it in lst if it.id not in ids]
        return cm

    def to_json(self) -> str:
        """Serialize the map to a JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> ContextMap:
        """Deserialize a map from a JSON string produced by ``to_json()``."""
        return cls.model_validate_json(text)
