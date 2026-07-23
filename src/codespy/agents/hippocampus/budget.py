from __future__ import annotations

import dspy
import tiktoken

from codespy.agents.hippocampus.context_map import ContextMap

_ENCODING = tiktoken.get_encoding("o200k_base")
_MAX_INPUT_FIELD_TOKENS = 256

def count_tokens(s: str) -> int:
    return len(_ENCODING.encode(s))

def format_inputs(kwargs: dict) -> str:
    """Serialize call inputs (excluding context_map) for the distiller.

    Each field value is truncated to _MAX_INPUT_FIELD_TOKENS so that large
    RLM inputs (documents, file dumps) don't blow up the Distiller context.
    """
    parts: list[str] = []
    for k, v in kwargs.items():
        if k == "context_map":
            continue
        text = str(v)
        if count_tokens(text) > _MAX_INPUT_FIELD_TOKENS:
            lines, kept, tokens = text.splitlines(keepends=True), [], 0
            for line in lines:
                lt = count_tokens(line)
                if tokens + lt > _MAX_INPUT_FIELD_TOKENS:
                    kept.append("... (truncated)")
                    break
                kept.append(line)
                tokens += lt
            text = "".join(kept)
        parts.append(f"{k}: {text}")
    return "\n".join(parts)


# Eviction priority
_SECTION_EVICT_PRIORITY: dict[str, int] = {
    "parsing_schema": 0,        # evict first — cheap to rediscover
    "reusable_results": 1,      # agent-derived; can be recomputed
    "domain_constants": 2,      # exact values worth protecting
    "context_roadmap": 3,       # protected — structural index
    "context_understanding": 4,  # most protected — core orientation
}


def evict(cmap: ContextMap, scores: dict[str, int], budget: int) -> ContextMap:
    if count_tokens(cmap.render()) <= budget:
        return cmap
    item_section: dict[str, str] = {
        it.id: sec for sec in cmap.section_names() for it in cmap.section(sec)
    }
    flat = cmap.all_items()
    order = {it.id: i for i, it in enumerate(flat)}
    victims = sorted(
        flat,
        key=lambda it: (
            _SECTION_EVICT_PRIORITY.get(item_section[it.id], 99),
            scores.get(it.id, 0),
            order[it.id],
        ),
    )
    removed: set[str] = set()
    for v in victims:
        removed.add(v.id)
        trial = cmap.without(removed)
        if count_tokens(trial.render()) <= budget:
            return trial
    return cmap.without(removed)

def format_trajectory(pred: dspy.Prediction) -> str:
    traj = getattr(pred, "trajectory", None)
    if isinstance(traj, list):
        parts = []
        for i, entry in enumerate(traj):
            parts.append(f"--- Step {i + 1} ---")
            if entry.get("reasoning"):
                parts.append(f"Reasoning: {entry['reasoning']}")
            parts.append(f"Code:\n{entry['code']}")
            parts.append(f"Output:\n{entry['output']}")
        return "\n".join(parts)
    if isinstance(traj, dict):
        return "\n".join(f"{k}: {v}" for k, v in traj.items())
    try:
        return "\n".join(f"{k}: {v}" for k, v in pred.toDict().items())
    except Exception:
        return str(pred)


def truncate_trajectory(text: str, max_tokens: int) -> str:
    """Keep as many leading steps as fit within max_tokens."""
    if count_tokens(text) <= max_tokens:
        return text
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    tokens = 0
    for line in lines:
        line_tokens = count_tokens(line)
        if tokens + line_tokens > max_tokens:
            kept.append("... (truncated)\n")
            break
        kept.append(line)
        tokens += line_tokens
    return "".join(kept)
