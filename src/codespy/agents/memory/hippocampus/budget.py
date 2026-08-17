from __future__ import annotations

from dataclasses import dataclass

import dspy
import tiktoken

from codespy.agents.memory.hippocampus.context_memory import ContextMemory

_ENCODING = tiktoken.get_encoding("o200k_base")


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryBudget:
    """The four token budgets bounding Hippocampus memory.

    These four always travel together: a signature's memory is configured as a
    set, not field by field. All are measured with the ``o200k_base`` encoding.

    Frozen because they are read-only configuration — a budget can be resolved
    once (see ``Settings.get_memory_budget``) and shared across instances.

    Attributes:
        max_context_memory_tokens: Hard ceiling on the rendered ContextMemory,
            enforced by the Evictor after every reflection. This is the
            *persisted* artifact and it is prepended to every predictor of the
            wrapped agent, so it is re-sent on every agent iteration
            (~``max_iters`` times per run) plus once per reflection call — the
            most cost-sensitive of the four. Divided by ``max_context_item_tokens`` it
            gives the memory's approximate item capacity (8192 / 410 ~= 19 items).
        max_context_item_tokens: Budget for a *single* context memory item, passed to the
            Distiller and the Cartographer as a prompt input so they keep each
            item compact rather than spending the whole memory budget on one
            verbose entry. Unlike the other three this is a **soft** budget:
            expressed to the LLM, not enforced in code, since truncating an
            item could corrupt an exact constant it holds. Lower it to fit
            more, terser items in the same memory budget; raise it for richer
            items.
        max_trajectory_tokens: Budget for trajectories fed to the Distiller.
            None = full trajectory, Distiller does all compression — only
            viable for agents with short, predictable trajectories. Tool-using
            agents can produce 100k+ token trajectories, and TwoStepAdapter
            sends the value twice, so prefer ~5–10 % of the Distiller model's
            context window (default 8192, i.e. ~5 % of 128k). Applied per call
            (stage 1) and again over the combined episode in end_episode()
            (stage 2). Step-aware head+tail bounding (60 % head / 40 % tail)
            preserves both setup and conclusions.
        max_question_tokens: Budget for the serialized inputs used as the
            reflection "question", on the fallback path when
            ``Hippocampus.question`` is None. None = unbounded, which is
            rarely safe: *every* input field is serialized, so an agent taking
            a large field (a document dump, or a diff of every changed file)
            sends all of it to both the Distiller and the Cartographer.
            Ignored when ``question`` is set — prefer that when a single
            field cleanly captures intent.
    """

    max_context_memory_tokens: int = 8192
    max_context_item_tokens: int = 410
    max_trajectory_tokens: int | None = 8192
    max_question_tokens: int | None = 2048


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
    """Serialize call inputs (excluding context_memory) for the Distiller question.

    All fields are included in full. If max_tokens is set, the joined result is
    head+tail bounded via _head_tail_text so both the instruction and any
    trailing intent survive. See MemoryBudget.max_question_tokens for guidance
    on when and how to set a limit.
    """
    parts: list[str] = []
    for k, v in kwargs.items():
        if k == "context_memory":
            continue
        parts.append(f"{k}: {v}")
    text = "\n".join(parts)
    if max_tokens is None:
        return text
    return _head_tail_text(text, max_tokens)


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

def evict(context_memory: ContextMemory, scores: dict[str, int], budget: int) -> ContextMemory:
    if count_tokens(context_memory.render()) <= budget:
        return context_memory
    item_section: dict[str, str] = {
        it.id: sec for sec in context_memory.section_names() for it in context_memory.section(sec)
    }
    flat = context_memory.all_items()
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
        trial = context_memory.without(removed)
        if count_tokens(trial.render()) <= budget:
            return trial
    return context_memory.without(removed)


# ---------------------------------------------------------------------------
# Trajectory formatting with optional step-aware head+tail bounding
# ---------------------------------------------------------------------------

def _format_step(i: int, entry: dict) -> str:
    parts = [f"--- Step {i + 1} ---"]
    if entry.get("reasoning"):
        parts.append(f"Reasoning: {entry['reasoning']}")
    code = entry.get("code", "")
    if code:
        parts.append(f"Code:\n{code}")
    output = entry.get("output", "")
    if output:
        parts.append(f"Output:\n{output}")
    return "\n".join(parts)


# Tokens set aside for the "... (N tokens omitted) ..." marker so the returned
# text honours max_tokens including the marker itself.
_MARKER_RESERVE = 16


def _slice_tokens(text: str, n: int, from_end: bool = False) -> str:
    """Return the first (or last) ``n`` tokens of ``text`` as a string.

    Used to split a line that is itself larger than the remaining budget, so
    line-granularity bounding never has to discard a line wholesale.
    """
    if n <= 0:
        return ""
    tokens = _ENCODING.encode(text)
    kept = tokens[-n:] if from_end else tokens[:n]
    return _ENCODING.decode(kept)


def _head_tail_text(text: str, max_tokens: int, head_ratio: float = 0.6) -> str:
    """Keep the first head_ratio and last (1-head_ratio) of the token budget,
    dropping the middle with an omission marker.

    Prefers line granularity, but falls back to token granularity for a line
    that alone exceeds the remaining budget — without that fallback, a single
    long line (e.g. the one-line repr of a pydantic input field) would blow the
    whole budget and return almost nothing. Returns text unchanged if it fits.

    The omission marker is charged against ``max_tokens`` (see _MARKER_RESERVE),
    so the result never exceeds the budget.
    """
    if count_tokens(text) <= max_tokens:
        return text

    content_budget = max(max_tokens - _MARKER_RESERVE, 0)
    head_budget = int(content_budget * head_ratio)
    tail_budget = content_budget - head_budget

    lines = text.splitlines(keepends=True)

    # Collect head lines, token-slicing the line that straddles the budget.
    head_lines: list[str] = []
    head_tokens = 0
    for line in lines:
        lt = count_tokens(line)
        if head_tokens + lt > head_budget:
            head_lines.append(_slice_tokens(line, head_budget - head_tokens))
            head_tokens = head_budget
            break
        head_lines.append(line)
        head_tokens += lt

    # Collect tail lines (from the end), same straddle handling.
    tail_lines: list[str] = []
    tail_tokens = 0
    for line in reversed(lines):
        lt = count_tokens(line)
        if tail_tokens + lt > tail_budget:
            tail_lines.append(
                _slice_tokens(line, tail_budget - tail_tokens, from_end=True)
            )
            tail_tokens = tail_budget
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

        content_budget = max(max_tokens - _MARKER_RESERVE, 0)
        head_budget = int(content_budget * 0.6)
        tail_budget = content_budget - head_budget

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

        # Greedily keep tail steps (from the end), never reusing a head step
        tail_steps: list[str] = []
        tail_tokens = 0
        for s in reversed(capped[len(head_steps):]):
            t = count_tokens(s)
            if tail_tokens + t > tail_budget:
                break
            tail_steps.append(s)
            tail_tokens += t
        tail_steps.reverse()

        # If no whole step fits in either half (every step is larger than its
        # budget), fall back to bounding single steps so the budget is actually
        # used instead of returning just the omission marker.
        if not head_steps and capped:
            head_steps = [_head_tail_text(capped[0], head_budget)]
        if not tail_steps and len(capped) > len(head_steps):
            tail_steps = [_head_tail_text(capped[-1], tail_budget)]

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
