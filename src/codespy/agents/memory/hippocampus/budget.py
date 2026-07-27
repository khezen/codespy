from __future__ import annotations

import dspy
import tiktoken

from codespy.agents.memory.hippocampus.context_map import ContextMap

_ENCODING = tiktoken.get_encoding("o200k_base")

# Eviction priority — lower number = evict first
_SECTION_EVICT_PRIORITY: dict[str, int] = {
    "parsing_schema": 0,        # evict first — cheap to rediscover
    "reusable_results": 1,      # agent-derived; can be recomputed
    "domain_constants": 2,      # exact values worth protecting
    "context_roadmap": 3,       # protected — structural index
    "context_understanding": 4,  # most protected — core orientation
}


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(s: str) -> int:
    return len(_ENCODING.encode(s))


# ---------------------------------------------------------------------------
# Input serialisation (for the question field sent to the Distiller)
# ---------------------------------------------------------------------------

def format_inputs(kwargs: dict, max_tokens: int | None = None) -> str:
    """Serialize call inputs (excluding context_map) for the Distiller question.

    All fields are included in full. If max_tokens is set, the joined result is
    head+tail bounded via _head_tail_text so both the instruction and any
    trailing intent survive. See Hippocampus.max_input_tokens for guidance on
    when and how to set a limit.
    """
    parts: list[str] = []
    for k, v in kwargs.items():
        if k == "context_map":
            continue
        parts.append(f"{k}: {v}")
    text = "\n".join(parts)
    if max_tokens is None:
        return text
    return _head_tail_text(text, max_tokens)


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Trajectory formatting with optional step-aware head+tail bounding
# ---------------------------------------------------------------------------

def _format_step(i: int, entry: dict) -> str:
    parts = [f"--- Step {i + 1} ---"]
    if entry.get("reasoning"):
        parts.append(f"Reasoning: {entry['reasoning']}")
    parts.append(f"Code:\n{entry['code']}")
    parts.append(f"Output:\n{entry['output']}")
    return "\n".join(parts)


def _head_tail_text(text: str, max_tokens: int, head_ratio: float = 0.6) -> str:
    """Keep the first head_ratio and last (1-head_ratio) of the token budget,
    dropping the middle with an omission marker.

    Operates at line granularity; returns text unchanged if it fits.
    """
    if count_tokens(text) <= max_tokens:
        return text

    head_budget = int(max_tokens * head_ratio)
    tail_budget = max_tokens - head_budget

    lines = text.splitlines(keepends=True)

    # Collect head lines
    head_lines: list[str] = []
    head_tokens = 0
    for line in lines:
        lt = count_tokens(line)
        if head_tokens + lt > head_budget:
            break
        head_lines.append(line)
        head_tokens += lt

    # Collect tail lines (from the end)
    tail_lines: list[str] = []
    tail_tokens = 0
    for line in reversed(lines):
        lt = count_tokens(line)
        if tail_tokens + lt > tail_budget:
            break
        tail_lines.append(line)
        tail_tokens += lt
    tail_lines.reverse()

    total = count_tokens(text)
    omitted = total - head_tokens - tail_tokens
    marker = f"... ({omitted} tokens omitted) ...\n"

    return "".join(head_lines) + marker + "".join(tail_lines)


def format_trajectory(pred: dspy.Prediction, max_tokens: int | None = None) -> str:
    """Serialize a dspy trajectory to text, with optional head+tail bounding.

    Args:
        pred: The dspy Prediction returned by the wrapped agent.
        max_tokens: If None (default), the full trajectory is returned so the
            Distiller can do all compression. If set, step-aware head+tail
            bounding is applied: whole steps are kept from both the front and
            back of the trajectory (60 % head / 40 % tail), and the middle is
            replaced by an omission marker. For dict / fallback trajectories,
            the same head+tail logic is applied at line granularity. A single
            oversized step's Output block is itself head+tail bounded before
            the per-step budget accounting.

    Trajectory shapes handled:
        list  — ReAct / CodeAct: list of dicts with 'code', 'output',
                optional 'reasoning'. Step-aware bounding.
        dict  — flat key/value dump. Line-granularity bounding.
        other — str(pred) or pred.toDict() fallback. Line-granularity bounding.
    """
    traj = getattr(pred, "trajectory", None)

    # ----- list path (ReAct / CodeAct) -----
    if isinstance(traj, list):
        step_texts = [_format_step(i, entry) for i, entry in enumerate(traj)]

        if max_tokens is None:
            return "\n\n".join(step_texts)

        # Check if everything fits as-is
        full = "\n\n".join(step_texts)
        if count_tokens(full) <= max_tokens:
            return full

        head_budget = int(max_tokens * 0.6)
        tail_budget = max_tokens - head_budget

        # Cap individual oversized step outputs before budgeting
        capped: list[str] = []
        for s in step_texts:
            if count_tokens(s) > max_tokens:
                s = _head_tail_text(s, max_tokens)
            capped.append(s)

        # Greedily keep head steps
        head_steps: list[str] = []
        head_tokens = 0
        for s in capped:
            t = count_tokens(s)
            if head_tokens + t > head_budget:
                break
            head_steps.append(s)
            head_tokens += t

        # Greedily keep tail steps (from the end)
        tail_steps: list[str] = []
        tail_tokens = 0
        for s in reversed(capped):
            t = count_tokens(s)
            if tail_tokens + t > tail_budget:
                break
            tail_steps.append(s)
            tail_tokens += t
        tail_steps.reverse()

        # Determine omitted range
        n_head = len(head_steps)
        n_tail = len(tail_steps)
        n_total = len(capped)
        n_omitted = n_total - n_head - n_tail

        parts = list(head_steps)
        if n_omitted > 0:
            first_omitted = n_head + 1
            last_omitted = n_total - n_tail
            parts.append(
                f"--- Steps {first_omitted}–{last_omitted} omitted ({n_omitted} steps) ---"
            )
        parts.extend(tail_steps)
        return "\n\n".join(parts)

    # ----- dict path -----
    if isinstance(traj, dict):
        text = "\n".join(f"{k}: {v}" for k, v in traj.items())
        if max_tokens is None:
            return text
        return _head_tail_text(text, max_tokens)

    # ----- fallback -----
    try:
        text = "\n".join(f"{k}: {v}" for k, v in pred.toDict().items())
    except Exception:
        text = str(pred)
    if max_tokens is None:
        return text
    return _head_tail_text(text, max_tokens)
