"""Tag scopes and reach filters shared by Cerebral (write side) and Prefrontal (read side).

This module is deliberately lightweight: it never imports ``hindsight_api`` at
module level. Hindsight builds and caches its config at import time, and
``get_cerebral`` must set ``HINDSIGHT_API_*`` env vars before that happens, so
the Hindsight tag-group classes are imported lazily inside the functions that
build them (only ever called once a Cerebral exists).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

# Reach levels, from narrowest to widest.
REACH_LOCAL = "local"
REACH_ORG = "org"
REACH_BANK = "bank"
REACHES: tuple[str, ...] = (REACH_LOCAL, REACH_ORG, REACH_BANK)

# Tag prefixes written by Cerebral.retain_episode.
TAG_ORG = "org:"
TAG_REPO = "repo:"
TAG_PROJECT_SCOPE = "project_scope:"
TAG_TASK = "task:"


def org_of(repo_full_name: str) -> str:
    """Return the owner (first path segment) of ``owner/repo``."""
    return (repo_full_name or "").strip("/").split("/", 1)[0]


def repo_tags(repo_full_name: str) -> list[str]:
    """Tags identifying a repository: ``[org:<owner>, repo:<owner/repo>]``."""
    return [f"{TAG_ORG}{org_of(repo_full_name)}", f"{TAG_REPO}{repo_full_name}"]


def scope_tags(repo_full_name: str, scope_topic_id: str) -> list[str]:
    """Tags identifying one project scope: ``[org:, repo:, project_scope:]``."""
    return [*repo_tags(repo_full_name), f"{TAG_PROJECT_SCOPE}{scope_topic_id}"]


def mental_model_id(tags: Iterable[str]) -> str:
    """Deterministic mental-model id for a tag scope.

    ``mm-`` plus the first 32 hex chars of ``sha1("|".join(sorted(tags)))``.
    """
    digest = hashlib.sha1("|".join(sorted(tags)).encode("utf-8")).hexdigest()
    return f"mm-{digest[:32]}"


def mental_model_scopes(
    repo_full_name: str,
    scope_topic_ids: Sequence[str] | None,
    include_repo: bool,
) -> list[list[str]]:
    """Tag scopes whose briefings an agent reads: its scopes first, then the repo."""
    scopes = [scope_tags(repo_full_name, sid) for sid in (scope_topic_ids or [])]
    if include_repo:
        scopes.append(repo_tags(repo_full_name))
    return scopes


def mental_model_ids(
    repo_full_name: str,
    scope_topic_ids: Sequence[str] | None,
    include_repo: bool,
) -> list[str]:
    """Mental-model ids an agent reads, in briefing order (scopes, then repo)."""
    return [
        mental_model_id(t)
        for t in mental_model_scopes(repo_full_name, scope_topic_ids, include_repo)
    ]


def clamp_reach(requested: str | None, configured: str) -> str:
    """Clamp a requested reach to the configured one. Unknown values mean ``local``."""
    req = (requested or REACH_LOCAL).strip().lower()
    if req not in REACHES:
        req = REACH_LOCAL
    cfg = configured if configured in REACHES else REACH_LOCAL
    return REACHES[min(REACHES.index(req), REACHES.index(cfg))]


def local_tags(
    repo_full_name: str,
    scope_topic_ids: Sequence[str] | None,
    include_repo: bool,
) -> list[str]:
    """Tags of the local pool (OR-ed): ``project_scope:`` ids, plus ``repo:`` if requested."""
    tags = [f"{TAG_PROJECT_SCOPE}{sid}" for sid in (scope_topic_ids or [])]
    if include_repo:
        tags.insert(0, f"{TAG_REPO}{repo_full_name}")
    return tags


def build_local_groups(
    repo_full_name: str,
    scope_topic_ids: Sequence[str] | None,
    include_repo: bool,
) -> list[Any]:
    """Hindsight tag groups for the local pool.

    One ``any_strict`` leaf: a fact matches when it carries any of the local
    tags (and untagged facts never match). Returns ``[]`` when there is no
    local tag at all — callers must then skip the local pool rather than
    search the whole bank.
    """
    tags = local_tags(repo_full_name, scope_topic_ids, include_repo)
    if not tags:
        return []
    from hindsight_api.engine.search.tags import TagGroupLeaf

    return [TagGroupLeaf(tags=tags, match="any_strict")]


def build_remote_groups(repo_full_name: str, reach: str) -> list[Any] | None:
    """Hindsight tag groups for the remote pool, or ``None`` for ``local`` reach.

    - ``org``: ``org:<owner>`` AND NOT ``repo:<own>``
    - ``bank``: NOT ``repo:<own>``
    """
    if reach not in (REACH_ORG, REACH_BANK):
        return None
    from hindsight_api.engine.search.tags import TagGroupLeaf, TagGroupNot

    not_own = TagGroupNot(
        filter=TagGroupLeaf(tags=[f"{TAG_REPO}{repo_full_name}"], match="any_strict")
    )
    if reach == REACH_ORG:
        return [
            TagGroupLeaf(tags=[f"{TAG_ORG}{org_of(repo_full_name)}"], match="any_strict"),
            not_own,
        ]
    return [not_own]


def build_reflect_groups(
    repo_full_name: str,
    scope_topic_ids: Sequence[str] | None,
    include_repo: bool,
    reach: str,
) -> list[Any] | None:
    """Tag groups for a tool-side reflect over ``reach``.

    - ``local``: the local pool
    - ``org``: the local pool OR ``org:<owner>``
    - ``bank``: no filter (the whole bank)
    """
    if reach == REACH_BANK:
        return None
    tags = local_tags(repo_full_name, scope_topic_ids, include_repo)
    if reach == REACH_ORG:
        tags = [*tags, f"{TAG_ORG}{org_of(repo_full_name)}"]
    if not tags:
        return []
    from hindsight_api.engine.search.tags import TagGroupLeaf

    return [TagGroupLeaf(tags=tags, match="any_strict")]


def tag_value(tags: Iterable[str] | None, prefix: str) -> str | None:
    """Return the value of the first tag with ``prefix`` (e.g. ``task:``), or None."""
    for tag in tags or ():
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


def is_own_repo_fact(tags: Iterable[str] | None, repo_full_name: str) -> bool:
    """True when a fact belongs to ``repo_full_name``.

    Covers old data that has no ``repo:`` tag but a ``project_scope:`` id under
    the repo, which ``NOT repo:<own>`` cannot exclude.
    """
    own_scope = f"{TAG_PROJECT_SCOPE}{repo_full_name}"
    for tag in tags or ():
        if tag == f"{TAG_REPO}{repo_full_name}":
            return True
        if tag == own_scope or tag.startswith(own_scope + "/"):
            return True
    return False
