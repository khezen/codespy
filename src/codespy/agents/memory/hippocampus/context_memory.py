from __future__ import annotations

import logging
import os
import uuid
from enum import StrEnum
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

logger = logging.getLogger(__name__)


class ItemTag(StrEnum):
    """How a context-memory item performed in the trajectory just observed.

    - helpful: directly aided orientation or answering; keep.
    - harmful: misled the agent or contradicted observations; remove.
    - neutral: present but unused this round; keep with no boost.
    - stale:   no longer reflects the external context; remove.
    """

    HELPFUL = "helpful"
    HARMFUL = "harmful"
    NEUTRAL = "neutral"
    STALE = "stale"

    @classmethod
    def _missing_(cls, value: object) -> ItemTag | None:
        if isinstance(value, str):
            lower = value.lower()
            for member in cls:
                if member.value == lower:
                    return member
        return None


class OpType(StrEnum):
    """Cartographer edit operations against the context memory."""

    ADD = "ADD"
    DELETE = "DELETE"
    REPLACE = "REPLACE"

    @classmethod
    def _missing_(cls, value: object) -> OpType | None:
        if isinstance(value, str):
            upper = value.upper()
            for member in cls:
                if member.value == upper:
                    return member
        return None


SectionName = Literal[
    "context_roadmap",
    "context_understanding",
    "domain_constants",
    "experiences",
    "parsing_schema",
    "reusable_results",
]

# Abbreviated prefixes
_SECTION_PREFIX: dict[str, str] = {
    "context_roadmap": "cr",
    "context_understanding": "cu",
    "domain_constants": "dc",
    "experiences": "ex",
    "parsing_schema": "ps",
    "reusable_results": "rr",
}


class Topic(BaseModel):
    """A topic representing a scope in the repository.

    Topics are used to group context items by their relevant scope.
    """

    id: str = Field(description="Topic identifier (e.g., 'owner/repo/package-name')")
    description: str = Field(description="Description of this topic's role")
    dependencies: list[str] = Field(
        default_factory=list, description="Topic IDs of this topic's dependencies"
    )


class Item(BaseModel):
    """A single item in the context memory."""

    id: str = Field(description="Unique item identifier")
    content: str = Field(description="Item content")
    topic_ids: list[str] = Field(
        default_factory=list, description="IDs of topics this item belongs to"
    )

    def bind_topics(self, topic_ids: list[str]) -> None:
        """Bind this item to the given topic_ids (only if currently unbound)."""
        if not self.topic_ids:
            self.topic_ids = list(topic_ids)


class CacheCandidate(BaseModel):
    """A candidate item to be added to the context memory."""

    section: SectionName = Field(
        default="context_understanding",
        description=(
            "Target section: context_understanding, domain_constants, "
            "context_roadmap, reusable_results, parsing_schema, or experiences"
        ),
    )
    value: str = Field(
        description="Compact candidate cache item, within the max_context_item_tokens budget."
    )
    transferability: str = Field(
        default="",
        description="Kinds of future questions this would help.",
    )
    rationale: str = Field(
        default="",
        description="Why this is shared understanding, not a one-off fact.",
    )


class Operation(BaseModel):
    """A single edit operation against the context memory."""

    type: OpType = Field(
        description="Operation type: ADD, DELETE, or REPLACE",
        validation_alias=AliasChoices("type", "op"),
    )
    section: SectionName | None = Field(default=None, description="Required for ADD.")
    item_id: str | None = Field(default=None, description="Required for DELETE / REPLACE.")
    content: str | None = Field(default=None, description="Required for ADD / REPLACE.")


class Mutation(BaseModel):
    """A recorded Cartographer mutation applied to the context memory.

    Tracks the sequence of ADD/DELETE/REPLACE operations with pre-mutation
    state for debugging and audit purposes.
    """

    step: int = Field(description="Which _distill() pass produced this mutation (0-indexed)")
    type: OpType = Field(description="Type of mutation: ADD, DELETE, or REPLACE")
    item_id: str = Field(description="Generated ID (ADD) or existing ID (DELETE/REPLACE)")
    section: SectionName = Field(description="Section the item belongs to")
    content: str | None = Field(
        default=None, description="New content (ADD/REPLACE); None for DELETE"
    )
    previous_content: str | None = Field(
        default=None, description="Old content (DELETE/REPLACE); None for ADD"
    )
    topic_ids: list[str] = Field(
        default_factory=list, description="Topic IDs associated with this mutation"
    )


class ContextMemory(BaseModel):
    """Context memory with topics and sectioned items.

    Topic-aware structure where each Item links to one or more topics via topic_ids.
    """

    topics: list[Topic] = Field(default_factory=list, description="Topics representing repo scopes")
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
    experiences: list[Item] = Field(
        default_factory=list,
        description=(
            "Tool execution experiences: what tool was used, for what purpose, "
            "and what the result was. Helps avoid redundant tool calls in future runs."
        ),
    )

    @classmethod
    def section_names(cls) -> list[str]:
        """Return list of section field names (excluding 'topics')."""
        return [f for f in cls.model_fields if f != "topics"]

    def section(self, name: str) -> list[Item]:
        """Get items from a section by name."""
        return getattr(self, name)

    def bind_topics(self, topics: list[Topic], default_topic_ids: list[str]) -> None:
        """Set topics and bind all untagged items to default_topic_ids.

        Used by the scope resolver after topics are computed post-hoc
        (chicken-and-egg: Hippocampus runs before topics are known).

        Args:
            topics: Full list of Topic objects to set on this memory.
            default_topic_ids: topic_ids to assign to any item with empty topic_ids.
        """
        self.topics = topics
        for sec in self.section_names():
            for item in self.section(sec):
                item.bind_topics(default_topic_ids)

    def all_items(self) -> list[Item]:
        """Return all items across all sections."""
        return [it for s in self.section_names() for it in self.section(s)]

    def find_item(self, item_id: str) -> tuple[SectionName, Item] | None:
        """Look up an item by ID across all sections.

        Returns:
            Tuple of (section_name, item) if found, None otherwise.
        """
        for sec in self.section_names():
            for it in self.section(sec):
                if it.id == item_id:
                    return sec, it  # type: ignore[return-value]
        return None

    def ids(self) -> set[str]:
        """Return set of all item IDs."""
        return {it.id for it in self.all_items()}

    def render(self) -> str:
        """Render the context memory as topic-grouped text for LLM consumption.

        Returns:
            Topic-grouped text with items organized under their respective topics.
            Returns empty string if ContextMemory is completely empty (no topics with items).
        """
        # Build topic ID -> topic map
        topic_map: dict[str, Topic] = {t.id: t for t in self.topics}

        # Categorize items
        shared_items: list[tuple[SectionName, Item]] = []
        topic_items: dict[str, list[tuple[SectionName, Item]]] = {}

        for sec_name in self.section_names():
            for item in self.section(sec_name):
                if not item.topic_ids or len(item.topic_ids) != 1:
                    # Empty or multiple topics -> SHARED
                    shared_items.append((sec_name, item))
                else:
                    topic_id = item.topic_ids[0]
                    if topic_id in topic_map:
                        if topic_id not in topic_items:
                            topic_items[topic_id] = []
                        topic_items[topic_id].append((sec_name, item))
                    else:
                        # Unknown topic_id -> SHARED
                        shared_items.append((sec_name, item))

        # Check if completely empty
        if not shared_items and not topic_items:
            return ""

        lines: list[str] = []

        # Render SHARED group first
        if shared_items:
            lines.append("## SHARED")
            lines.extend(self._render_items_by_section(shared_items))
            lines.append("")

        # Render topic groups in order they appear in topics list
        for topic in self.topics:
            if topic.id in topic_items:
                items = topic_items[topic.id]
                lines.append(f"## TOPIC: {topic.id} ({topic.description})")
                lines.extend(self._render_items_by_section(items))
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _render_items_by_section(self, items: list[tuple[SectionName, Item]]) -> list[str]:
        """Render items grouped by section.

        Args:
            items: List of (section_name, item) tuples

        Returns:
            List of formatted lines
        """
        # Group by section
        by_section: dict[str, list[Item]] = {}
        for sec_name, item in items:
            if sec_name not in by_section:
                by_section[sec_name] = []
            by_section[sec_name].append(item)

        lines: list[str] = []
        # Render in section order (as defined in section_names)
        for sec_name in self.section_names():
            if sec_name in by_section:
                sec_items = by_section[sec_name]
                if sec_items:
                    sec_display = sec_name.upper().replace("_", " ")
                    lines.append(f"### {sec_display}")
                    for item in sec_items:
                        lines.append(f"[{item.id}] {item.content}")
        return lines

    def apply(
        self, ops: list[Operation], topic_ids: list[str] | None = None
    ) -> tuple[ContextMemory, list[str]]:
        """Apply operations to create a new ContextMemory.

        Args:
            ops: List of operations to apply (ADD, DELETE, REPLACE)
            topic_ids: Optional list of topic IDs to assign to new items

        Returns:
            Tuple of (new ContextMemory, list of IDs of newly-added items)
        """
        cm = self.model_copy(deep=True)
        new_ids: list[str] = []

        for op in ops:
            if op.type == OpType.DELETE and op.item_id:
                for sec in cm.section_names():
                    lst = cm.section(sec)
                    lst[:] = [it for it in lst if it.id != op.item_id]

            elif op.type == OpType.REPLACE and op.item_id and op.content:
                replaced = False
                for sec in cm.section_names():
                    lst = cm.section(sec)
                    for i, it in enumerate(lst):
                        if it.id == op.item_id:
                            # Preserve existing topic_ids on REPLACE
                            lst[i] = Item(id=it.id, content=op.content, topic_ids=it.topic_ids)
                            replaced = True
                            break
                    if replaced:
                        break
                if not replaced:
                    logger.warning(
                        "REPLACE target %r not found in context memory; skipping",
                        op.item_id,
                    )

            elif op.type == OpType.ADD and op.section and op.content:
                prefix = _SECTION_PREFIX.get(op.section, op.section[:2])
                new_id = f"{prefix}-{uuid.uuid4().hex}"
                new_item = Item(id=new_id, content=op.content, topic_ids=topic_ids or [])
                cm.section(op.section).append(new_item)
                new_ids.append(new_id)

        return cm, new_ids

    def without(self, ids: set[str]) -> ContextMemory:
        """Return a new ContextMemory without the specified items."""
        cm = self.model_copy(deep=True)
        for sec in cm.section_names():
            lst = cm.section(sec)
            lst[:] = [it for it in lst if it.id not in ids]
        return cm

    def to_json(self) -> str:
        """Serialize the memory to a JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> ContextMemory:
        """Deserialize a memory from a JSON string."""
        return cls.model_validate_json(text)

    @classmethod
    def merge(cls, *memories: ContextMemory) -> ContextMemory:
        """Merge multiple context memories into a single memory.

        Later memories win on ID collision (items with duplicate IDs are
        replaced by those from later memories in the argument list).

        Topics are deduplicated by ID, with later non-empty descriptions winning.

        Args:
            *memories: One or more ContextMemory instances to merge.

        Returns:
            A new ContextMemory containing merged topics and items.
        """
        merged = cls()

        # Merge topics (deduplicate by id, later non-empty description wins)
        topic_map: dict[str, Topic] = {}
        for mem in memories:
            for topic in mem.topics:
                if topic.id not in topic_map:
                    topic_map[topic.id] = topic
                elif topic.description and not topic_map[topic.id].description:
                    # Later non-empty description wins
                    topic_map[topic.id] = topic
        merged.topics = list(topic_map.values())

        # Merge items
        for mem in memories:
            for sec in cls.section_names():
                merged_section = merged.section(sec)
                existing_ids = {item.id for item in merged_section}
                for item in mem.section(sec):
                    if item.id in existing_ids:
                        # Replace existing item (later wins)
                        merged_section[:] = [
                            it if it.id != item.id else item.model_copy(deep=True)
                            for it in merged_section
                        ]
                    else:
                        merged_section.append(item.model_copy(deep=True))
                        existing_ids.add(item.id)

        return merged


def make_topic_id(repo_full_name: str, subroot: str, package_name: str | None = None) -> str:
    """Build topic ID from repo identity and scope info.

    Priority:
    1. package_name provided and contains owner/repo -> use from owner/repo onwards
    2. package_name provided -> {repo_full_name}/{package_name}
    3. Fallback -> {repo_full_name}/{subroot} (or just repo_full_name for root)

    Args:
        repo_full_name: Repository full name (owner/repo)
        subroot: Path relative to repo root
        package_name: Optional package name from manifest

    Returns:
        Topic ID string
    """
    if package_name:
        if repo_full_name in package_name:
            idx = package_name.index(repo_full_name)
            return package_name[idx:]
        return f"{repo_full_name}/{package_name}"
    if subroot in (".", ""):
        return repo_full_name
    return f"{repo_full_name}/{subroot.strip('/')}"


def compute_common_ancestor_topic_id(repo_full_name: str, subroots: list[str]) -> str | None:
    """Return topic ID for deepest common ancestor when >1 scope exists.

    Args:
        repo_full_name: Repository full name (owner/repo)
        subroots: List of subroot paths

    Returns:
        Topic ID for common ancestor, or None if <=1 scope
    """
    if len(subroots) <= 1:
        return None
    paths = [s.rstrip("/") for s in subroots if s not in (".", "")]
    if len(paths) < 2:
        return make_topic_id(repo_full_name, ".")
    common = os.path.commonpath(paths)
    if not common or common == ".":
        return make_topic_id(repo_full_name, ".")
    return make_topic_id(repo_full_name, common)
