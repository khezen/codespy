"""PR conversation primitives: codespy markers, fingerprints, author origin.

Pure helpers (no platform calls) shared by the Git clients, the reporter, and
the review feedback package. Kept under ``tools/git`` so the clients can use
them without importing the review agents.

Self-exclusion: every comment is classified once at fetch time by
:func:`classify_author`. codespy's own comments must never reach the LLM
context or memory as if they were human input.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

# --- Markers ------------------------------------------------------------------

MARKER_VERSION = "1"
ANY_MARKER_RE = re.compile(r"<!--\s*codespy:[^>]*-->")
REVIEW_MARKER_RE = re.compile(r"<!--\s*codespy:review\b([^>]*)-->")
ISSUE_MARKER_RE = re.compile(r"<!--\s*codespy:issue\b([^>]*)-->")
_KV_RE = re.compile(r"(\w+)=(\S+)")

# Legacy (pre-marker) codespy signatures.
LEGACY_HEADER_RE = re.compile(r"^\s*#\s*🔍\s*Code\[Spy\]")
LEGACY_INLINE_RE = re.compile(r"^\s*(?:🔴|🟠|🟡|🔵|⚪) \*\*\[")
# Same patterns anywhere at a line start (used by the leak guard).
_LEGACY_ANYLINE_RE = re.compile(
    r"^\s*(?:#\s*🔍\s*Code\[Spy\]|(?:🔴|🟠|🟡|🔵|⚪) \*\*\[)", re.MULTILINE
)

# Inline comment header written by the reporter: "🔴 **[High] Title**"
_INLINE_TITLE_RE = re.compile(r"\*\*\[(?P<sev>[^\]]+)\]\s*(?P<title>.+?)\*\*")
_INLINE_CATEGORY_RE = re.compile(r"\*\*Category:\*\*\s*(?P<cat>[\w-]+)")


def review_marker(run_id: str) -> str:
    """Hidden marker for a codespy review body."""
    return f"<!-- codespy:review v={MARKER_VERSION} run_id={run_id} -->"


def issue_marker(run_id: str, fp: str, category: str, severity: str) -> str:
    """Hidden marker for a codespy issue (inline comment or body issue)."""
    return (
        f"<!-- codespy:issue v={MARKER_VERSION} run_id={run_id} fp={fp} "
        f"cat={category} sev={severity} -->"
    )


def parse_issue_marker(body: str) -> dict[str, str] | None:
    """Parse the first issue marker of ``body`` into a key/value dict."""
    m = ISSUE_MARKER_RE.search(body or "")
    if not m:
        return None
    return dict(_KV_RE.findall(m.group(1)))


def parse_inline_header(body: str) -> tuple[str, str, str] | None:
    """Extract (severity, title, category) from a codespy inline comment body."""
    m = _INLINE_TITLE_RE.search(body or "")
    if not m:
        return None
    cat = _INLINE_CATEGORY_RE.search(body or "")
    category = cat.group("cat") if cat else ""
    return m.group("sev").strip().lower(), m.group("title").strip(), category


# --- Fingerprint --------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _WS_RE.sub(" ", _PUNCT_RE.sub("", (title or "").lower())).strip()


def fingerprint(filename: str, category: str, title: str) -> str:
    """Stable 12-hex fingerprint of an issue identity (repo-relative filename)."""
    key = f"{filename}|{str(category).lower()}|{normalize_title(title)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


# --- Origin classification ----------------------------------------------------


def has_self_signature(body: str) -> bool:
    """True if ``body`` carries a codespy marker or starts with a legacy codespy header."""
    body = body or ""
    return bool(
        ANY_MARKER_RE.search(body) or LEGACY_HEADER_RE.match(body) or LEGACY_INLINE_RE.match(body)
    )


def is_bot_login(login: str | None, bot_flag: bool = False) -> bool:
    """True for bot accounts (``*[bot]`` login or platform bot flag)."""
    return bool(bot_flag) or bool(login and login.lower().endswith("[bot]"))


def classify_author(
    body: str,
    author: str | None,
    *,
    self_login: str | None = None,
    in_codespy_review: bool = False,
    in_codespy_thread: bool = False,
    bot: bool = False,
) -> str:
    """Classify a comment as ``"codespy"``, ``"bot"`` or ``"human"``.

    A comment is codespy if any of:
      1. its body contains a ``<!-- codespy:`` marker;
      2. its body starts with the legacy header / legacy inline pattern;
      3. (GitHub) it is an inline comment of a codespy review (``in_codespy_review``);
      4. its author is ``self_login`` AND rule 1/2 holds (covered by 1/2: identity
         alone is never enough since humans may run codespy with their own token);
      5. (GitLab) it is in a discussion started by codespy (``in_codespy_thread``)
         AND its author is ``self_login``.
    """
    if has_self_signature(body):
        return "codespy"
    if in_codespy_review:
        return "codespy"
    if in_codespy_thread and self_login and author == self_login:
        return "codespy"
    if is_bot_login(author, bot):
        return "bot"
    return "human"


# --- Human text sanitation ----------------------------------------------------


def strip_self_content(text: str) -> str:
    """Remove quoted lines (``>``) and codespy markers from human text."""
    if not text:
        return ""
    kept = [ln for ln in text.splitlines() if not ln.lstrip().startswith(">")]
    return ANY_MARKER_RE.sub("", "\n".join(kept)).strip()


def contains_self_content(text: str) -> bool:
    """True if ``text`` contains a codespy marker or legacy codespy header anywhere."""
    return bool(text and (ANY_MARKER_RE.search(text) or _LEGACY_ANYLINE_RE.search(text)))


# --- Reactions ----------------------------------------------------------------

CANONICAL_REACTIONS = ("+1", "-1", "laugh", "hooray", "confused", "heart", "rocket", "eyes")

_GITLAB_EMOJI_MAP = {
    "thumbsup": "+1",
    "+1": "+1",
    "thumbsdown": "-1",
    "-1": "-1",
    "laughing": "laugh",
    "smile": "laugh",
    "joy": "laugh",
    "tada": "hooray",
    "confused": "confused",
    "heart": "heart",
    "rocket": "rocket",
    "eyes": "eyes",
}


def canonical_reaction(name: str) -> str:
    """Map a platform reaction/award-emoji name to the canonical GitHub-style set."""
    n = (name or "").strip().lower()
    return _GITLAB_EMOJI_MAP.get(n, n)


def rollup_reactions(raw: dict | None) -> dict[str, int]:
    """Canonical counts from a GitHub ``reactions`` rollup dict (non-zero only)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key in CANONICAL_REACTIONS:
        try:
            n = int(raw.get(key) or 0)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            out[key] = n
    return out


def count_reactions(items: Iterable[tuple[str, str | None, bool]]) -> dict[str, int]:
    """Count ``(name, user_login, is_bot)`` reactions, excluding bot accounts.

    The codespy token's own user is NOT excluded: codespy never reacts, and a
    human running codespy with their own token reacts as that same user.
    """
    out: dict[str, int] = {}
    for name, login, bot in items:
        if is_bot_login(login, bot):
            continue
        key = canonical_reaction(name)
        if key:
            out[key] = out.get(key, 0) + 1
    return out


# --- Capping ------------------------------------------------------------------


def cap_comments(comments: list, max_comments: int) -> list:
    """Bound non-codespy comments to ``max_comments``; codespy comments are always kept.

    Human/bot replies to codespy comments (feedback) are kept first, then the
    most recent other comments. Result is returned in chronological order.
    """
    from datetime import UTC, datetime

    def _ts(c) -> datetime:
        ts = getattr(c, "created_at", None)
        if ts is None:
            return datetime.min.replace(tzinfo=UTC)
        return ts if ts.tzinfo else ts.replace(tzinfo=UTC)

    own = [c for c in comments if c.origin == "codespy"]
    own_ids = {c.id for c in own}
    own_threads = {c.thread_id for c in own if c.thread_id}
    others = [c for c in comments if c.origin != "codespy"]

    def _is_feedback(c) -> bool:
        return (c.in_reply_to_id is not None and c.in_reply_to_id in own_ids) or (
            c.thread_id is not None and c.thread_id in own_threads
        )

    if len(others) > max(0, max_comments):
        ranked = sorted(others, key=lambda c: (not _is_feedback(c), -_ts(c).timestamp()))
        others = ranked[: max(0, max_comments)]
    return sorted(own + others, key=_ts)
