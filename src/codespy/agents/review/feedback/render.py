"""Render PR conversation context blocks for LLM inputs and memory.

Every block is wrapped in a ``text`` fence (human text is untrusted) and passes
through :func:`assert_no_self_content` so codespy's own output never comes
back in as human input.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from codespy.agents.review.feedback.extract import feedback_reply_ids
from codespy.agents.review.models import IssueFeedback
from codespy.tools.git.conversation import contains_self_content, strip_self_content
from codespy.tools.git.models import PRCommit, PullRequest

logger = logging.getLogger(__name__)

REACTION_EMOJI = {
    "+1": "👍",
    "-1": "👎",
    "laugh": "😄",
    "hooray": "🎉",
    "confused": "😕",
    "heart": "❤️",
    "rocket": "🚀",
    "eyes": "👀",
}


def format_reactions(reactions: dict[str, int] | None) -> str:
    """``(👍2 👎1)`` or empty string."""
    if not reactions:
        return ""
    parts = [f"{REACTION_EMOJI.get(k, ':' + k + ':')}{n}" for k, n in reactions.items() if n > 0]
    return f" ({' '.join(parts)})" if parts else ""


def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _fence(text: str) -> str:
    """Wrap in a text fence; neutralize inner fences so content cannot break out."""
    return "```text\n" + text.replace("```", "ʼʼʼ") + "\n```"


def assert_no_self_content(text: str, block: str = "") -> bool:
    """True if ``text`` is free of codespy markers/headers; logs a warning otherwise."""
    if contains_self_content(text):
        logger.warning("Dropped item leaking codespy content from %s block", block or "context")
        return False
    return True


def _pack_recent(items: Sequence[str], max_chars: int) -> list[str]:
    """Keep the most recent items (end of list) fitting ``max_chars``; oldest dropped first."""
    if max_chars <= 0:
        return []
    kept: list[str] = []
    total = 0
    for item in reversed(items):
        size = len(item) + 1
        if total + size > max_chars:
            break
        kept.append(item)
        total += size
    kept.reverse()
    return kept


def render_commits(
    commits: Iterable[PRCommit], max_comment_chars: int = 1000, max_block_chars: int = 12000
) -> str:
    """Commit messages, oldest first (most recent kept under the cap)."""
    items: list[str] = []
    for c in commits or []:
        msg = _clip(c.message, max_comment_chars)
        if not msg:
            continue
        first, _, rest = msg.partition("\n")
        line = f"- {c.sha[:7]} {first.strip()}"
        if rest.strip():
            line += "\n" + "\n".join("    " + ln for ln in rest.strip().splitlines())
        items.append(line)
    items = _pack_recent(items, max_block_chars)
    return _fence("\n".join(items)) if items else ""


def render_discussion(
    pr: PullRequest, max_comment_chars: int = 1000, max_block_chars: int = 12000
) -> str:
    """Human PR comments (codespy and bot comments excluded) with reaction counts.

    Replies to codespy findings are excluded too: they are rendered as review
    feedback instead.
    """
    skip = feedback_reply_ids(pr)
    items: list[str] = []
    for c in pr.comments or []:
        if c.origin != "human" or c.id in skip:
            continue
        text = _clip(strip_self_content(c.body), max_comment_chars)
        if not text:
            continue
        where = c.kind
        if c.path:
            where += f" {c.path}" + (f":{c.line}" if c.line else "")
        item = f"- [{where}] @{c.author}{format_reactions(c.reactions)}: {text}"
        if assert_no_self_content(item, "pr_discussion"):
            items.append(item)
    header = []
    if pr.reactions:
        header.append(f"PR description reactions:{format_reactions(pr.reactions)}")
    items = _pack_recent(items, max_block_chars - sum(len(h) + 1 for h in header))
    if not items and not header:
        return ""
    return _fence("\n".join(header + items))


def render_feedback(
    items: Iterable[IssueFeedback], max_comment_chars: int = 1000, max_block_chars: int = 12000
) -> str:
    """Feedback on codespy's earlier findings, grouped by file.

    Only the issue identity (title/category/severity/location) is rendered from
    codespy's side, never its description or suggestion.
    """
    by_file: dict[str, list[str]] = {}
    for f in items or []:
        loc = f" line {f.line}" if f.line else ""
        kind = "/".join(x for x in (f.category, f.severity) if x)
        entry = f"- [{f.verdict}] {kind} \"{f.title}\"{loc}{format_reactions(f.reactions)}"
        for r in f.replies:
            entry += f"\n    reply {_clip(r, max_comment_chars)}"
        if assert_no_self_content(entry, "review_feedback"):
            by_file.setdefault(f.filename, []).append(entry)
    if not by_file:
        return ""
    lines: list[str] = []
    for filename, entries in by_file.items():
        lines.append(f"{filename}:")
        lines.extend(entries)
    text = "\n".join(lines)
    if len(text) > max_block_chars:
        text = _clip(text, max_block_chars)
    return _fence(text) if text else ""
