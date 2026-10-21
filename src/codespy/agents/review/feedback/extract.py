"""Extract human feedback on codespy's earlier review comments of a PR."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from codespy.agents.review.models import Issue, IssueFeedback
from codespy.tools.git.conversation import (
    contains_self_content,
    fingerprint,
    parse_inline_header,
    parse_issue_marker,
    strip_self_content,
)
from codespy.tools.git.models import PRComment, PullRequest

logger = logging.getLogger(__name__)

POSITIVE_REACTIONS = frozenset({"+1", "heart", "hooray", "rocket"})
NEGATIVE_REACTIONS = frozenset({"-1", "confused"})


def verdict(reactions: dict[str, int], has_replies: bool, resolved: bool | None) -> str | None:
    """Deterministic verdict from human signals, or None when there is no signal."""
    pos = sum(n for k, n in reactions.items() if k in POSITIVE_REACTIONS)
    neg = sum(n for k, n in reactions.items() if k in NEGATIVE_REACTIONS)
    if neg > pos:
        return "rejected"
    if pos > neg:
        return "accepted"
    if has_replies:
        return "discussed"
    if resolved:
        return "resolved"
    return None


def feedback_reply_ids(pr: PullRequest) -> set:
    """IDs of non-codespy comments that reply to a codespy comment (they are feedback)."""
    own = [c for c in pr.comments if c.origin == "codespy"]
    own_ids = {c.id for c in own}
    own_threads = {c.thread_id for c in own if c.thread_id and c.kind == "inline"}
    return {
        c.id
        for c in pr.comments
        if c.origin != "codespy"
        and (
            (c.in_reply_to_id is not None and c.in_reply_to_id in own_ids)
            or (c.thread_id is not None and c.thread_id in own_threads)
        )
    }


def _human_replies(root: PRComment, comments: Sequence[PRComment]) -> list[str]:
    replies: list[str] = []
    for c in comments:
        if c.origin != "human" or c.id == root.id:
            continue
        is_reply = (c.in_reply_to_id is not None and c.in_reply_to_id == root.id) or (
            root.thread_id is not None and c.thread_id == root.thread_id
        )
        if not is_reply:
            continue
        text = strip_self_content(c.body)
        if not text or contains_self_content(text):
            continue
        replies.append(f"@{c.author}: {text}" if c.author else text)
    return replies


def extract_feedback(pr: PullRequest) -> list[IssueFeedback]:
    """Build one IssueFeedback per codespy finding that received human signals.

    Only codespy INLINE comments carry per-issue feedback (review bodies have no
    per-issue reactions). A comment is trusted as a codespy finding only when it
    carries a codespy marker/header AND the platform says it was posted by the
    token's own account (``authored_by_self``): markers alone can be forged to
    fabricate "rejected" findings. The same finding posted by several runs is
    merged by fingerprint. Items without any human signal are dropped.
    """
    comments = list(pr.comments or [])
    merged: dict[str, dict] = {}
    for root in comments:
        if root.origin != "codespy" or root.kind != "inline" or not root.path:
            continue
        if root.in_reply_to_id is not None:
            continue
        if not root.authored_by_self:
            logger.warning(
                "Ignoring unverified codespy-looking comment %s by @%s on %s "
                "(author is neither the token's account nor in review.pr_context.trusted_authors)",
                root.id, root.author, root.path,
            )
            continue
        marker = parse_issue_marker(root.body) or {}
        header = parse_inline_header(root.body)
        if not marker and not header:
            continue
        sev, title, cat = header or ("", "", "")
        cat = marker.get("cat") or cat
        sev = marker.get("sev") or sev
        fp = marker.get("fp") or fingerprint(root.path, cat, title)
        entry = merged.setdefault(
            fp,
            {
                "fingerprint": fp, "filename": root.path, "line": root.line,
                "category": cat, "severity": sev, "title": title,
                "reactions": {}, "replies": [], "resolved": False,
            },
        )
        if title and not entry["title"]:
            entry["title"] = title
        for k, n in (root.reactions or {}).items():
            entry["reactions"][k] = entry["reactions"].get(k, 0) + n
        for r in _human_replies(root, comments):
            if r not in entry["replies"]:
                entry["replies"].append(r)
        entry["resolved"] = entry["resolved"] or bool(root.resolved)

    out: list[IssueFeedback] = []
    for e in merged.values():
        v = verdict(e["reactions"], bool(e["replies"]), e["resolved"])
        if v is None:
            continue
        out.append(
            IssueFeedback(
                fingerprint=e["fingerprint"], filename=e["filename"], line=e["line"],
                category=e["category"], severity=e["severity"], title=e["title"],
                verdict=v, reactions=e["reactions"], replies=e["replies"],
            )
        )
    if out:
        logger.info(
            "Review feedback: %d findings with human signals (%d rejected)",
            len(out), sum(1 for f in out if f.verdict == "rejected"),
        )
    return out


def _norm_subroot(subroot: str | None) -> str:
    s = (subroot or "").strip().strip("/")
    return "" if s in ("", ".") else s


def for_scope(items: Iterable[IssueFeedback], subroot: str | None) -> list[IssueFeedback]:
    """Feedback items whose file lies under ``subroot`` (all items for the repo root)."""
    root = _norm_subroot(subroot)
    if not root:
        return list(items)
    prefix = root + "/"
    return [f for f in items if f.filename == root or f.filename.startswith(prefix)]


def group_by_scope(
    items: Iterable[IssueFeedback], subroots: Sequence[str]
) -> dict[str, list[IssueFeedback]]:
    """Assign each item to the deepest matching scope subroot (each item once)."""
    norm = sorted({_norm_subroot(s): s for s in subroots}.items(), key=lambda kv: -len(kv[0]))
    groups: dict[str, list[IssueFeedback]] = {}
    for f in items:
        for root, original in norm:
            if not root or f.filename == root or f.filename.startswith(root + "/"):
                groups.setdefault(original, []).append(f)
                break
    return groups


def issue_fingerprint(issue: Issue) -> str:
    """Fingerprint of a review Issue (repo-relative filename expected)."""
    return fingerprint(issue.filename, issue.category.value, issue.title)


def drop_rejected(issues: list[Issue], feedback: Iterable[IssueFeedback]) -> list[Issue]:
    """Drop issues whose fingerprint matches a finding humans rejected on this PR."""
    rejected = {f.fingerprint for f in feedback if f.verdict == "rejected"}
    if not rejected:
        return issues
    kept: list[Issue] = []
    for issue in issues:
        if issue_fingerprint(issue) in rejected:
            logger.info("Dropping finding rejected by humans: %s (%s)", issue.title, issue.location)
            continue
        kept.append(issue)
    return kept
