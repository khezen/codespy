"""Paths the implementation agent must never write (shared by tools and collection)."""

# Repo-root-relative prefixes refused for writes (segment-aware)
PROTECTED_PREFIXES = (".github/workflows", ".codespy")


def normalize_rel_path(path: str) -> str:
    """Strip whitespace, leading ``./`` and slashes."""
    p = (path or "").strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


def is_protected_path(path: str) -> bool:
    """True for ``.git`` anywhere, ``.github/workflows/`` and ``.codespy/`` at the root."""
    p = normalize_rel_path(path)
    if not p:
        return True
    if ".git" in p.split("/"):
        return True
    return any(p == pre or p.startswith(pre + "/") for pre in PROTECTED_PREFIXES)
