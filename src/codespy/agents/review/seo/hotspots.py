"""SEO hotspot pre-scan: deterministic regex sweep for render-gating shapes.

Flags code shapes that usually keep facts or links out of the initial
server HTML (content behind open-state props, one-of-N dataset selectors,
links mounted on hover, viewport-gated rendering, ``ssr: false``). No LLM.
The SEO agent verifies or dismisses every hotspot; the scan never produces
issues by itself, and the agent runs its own sweep on top of it.
"""

from __future__ import annotations

import logging
import os
import posixpath
import re
import stat
import subprocess
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import dspy  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 512 * 1024
MAX_SCOPE_FILES = 2000
SCAN_EXTENSIONS = (".tsx", ".jsx", ".ts", ".js")
_RESOLVE_EXTENSIONS = (".tsx", ".ts", ".jsx", ".js")

_EXCLUDED_DIRS = re.compile(
    r"(^|/)(__tests__|__mocks__|mocks|stories|storybook|fixtures|tests?|e2e|"
    r"node_modules|\.next|dist|build|coverage)/"
)
_EXCLUDED_NAMES = re.compile(r"\.(test|spec|stories|story)\.[jt]sx?$")


@dataclass(frozen=True)
class Rule:
    """One hotspot pattern."""

    id: str
    regex: re.Pattern[str]
    requires: re.Pattern[str] | None
    hint: str


RULES: tuple[Rule, ...] = (
    Rule(
        "disclosure-prop",
        re.compile(r"(?<![\w-])(isOpen|open|show|isExpanded|expanded)=\{"),
        re.compile(r"\buseState\b"),
        "content passed to an open/closed component (modal, dialog, drawer, popover, "
        "accordion); closed content is often unmounted and absent from server HTML",
    ),
    Rule(
        "selected-dataset",
        re.compile(
            r"\.find\(\s*\(?\w+\)?\s*=>\s*\w+\.(selected|active|isDefault)\s*\)"
            r"|useState<[^>]*>\(\s*[\w.]+\[0\]\s*\)"
        ),
        None,
        "only the selected item of a collection renders; the other datasets are "
        "missing and the shown one may not name its subject",
    ),
    Rule(
        "mount-on-interaction",
        re.compile(
            r"\{\s*!?(isActive|isOpen|isHovered|isExpanded|subnav|hoveredLink|show\w*)\s*&&"
        ),
        None,
        "links or text mounted only after hover/focus/open; crawlers never trigger it",
    ),
    Rule(
        "client-viewport-gate",
        re.compile(r"\{\s*!?(isMobile|isDesktop|isTablet|isMobileNavigation)\w*\s*&&"),
        None,
        "rendering branches on a client viewport hook; check what the server default renders",
    ),
    Rule(
        "ssr-false",
        re.compile(r"\bssr\s*:\s*false\b"),
        None,
        "component excluded from server rendering",
    ),
)

_RULE_HINTS = {r.id: r.hint for r in RULES}

_COMPONENT_DEFAULT = re.compile(r"export\s+default\s+(?:function\s+|class\s+)?([A-Z]\w*)")
_COMPONENT_DECL = re.compile(
    r"^(?:export\s+)?(?:const|function|class)\s+([A-Z]\w*)\s*[=:(<{]", re.MULTILINE
)
_IMPORT = re.compile(
    r"import\s+(?:type\s+)?(?:(\w+)\s*,?\s*)?(?:\{([^}]*)\})?\s*from\s*[\"']([^\"']+)[\"']"
)
_JSX_TAG = re.compile(r"<([A-Z]\w*)")
_DISCLOSURE_TAG = re.compile(
    r"Modal|Dialog|Drawer|Popover|Popup|Accordion|Collapse|Collapsible|Sheet|Tooltip|Disclosure"
)
_CONTENT_PROP = re.compile(
    r"\b(\w*(?:[cC]ontent|[tT]ext|children|[dD]escription|[bB]ody|[aA]nswer)\w*)\??\s*:\s*"
    r"(?:CdaStructuredTextValue|React\.ReactNode|ReactNode|string|StructuredText\w*)\b"
)


@dataclass
class Hotspot:
    """One rule match in one file (first line kept, extra matches counted)."""

    rule: str
    filename: str
    line: int
    snippet: str
    component: str
    extra: int = 0
    tag: str | None = None
    content_props: list[str] = field(default_factory=list)
    external_source: str | None = None
    used_by: list[str] = field(default_factory=list)
    context_only: bool = False  # PR mode: one-hop neighbour, not changed by the PR


def is_scannable(rel: str) -> bool:
    """Source file the rules apply to (no tests, stories, mocks, build output)."""
    return (
        rel.endswith(SCAN_EXTENSIONS)
        and not _EXCLUDED_DIRS.search(rel)
        and not _EXCLUDED_NAMES.search(rel)
    )


class _Reader:
    """Safe, cached reads of scope files (no symlinks, inside root, size-capped)."""

    def __init__(self, scope_root: Path) -> None:
        self.root = scope_root.resolve()
        self._cache: dict[str, str | None] = {}

    def read(self, rel: str) -> str | None:
        if rel in self._cache:
            return self._cache[rel]
        self._cache[rel] = text = self._read(rel)
        return text

    def _read(self, rel: str) -> str | None:
        full = self.root / rel
        try:
            st = full.lstat()
            if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_FILE_BYTES:
                return None
            if not full.resolve().is_relative_to(self.root):
                return None
            fd = os.open(full, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as fh:
                raw = fh.read()
            return raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None


def _imports(text: str) -> dict[str, str]:
    """Local name -> import source."""
    out: dict[str, str] = {}
    for default, named, source in _IMPORT.findall(text):
        if default:
            out[default] = source
        for part in named.split(","):
            part = part.strip().removeprefix("type ").strip()
            if not part:
                continue
            local = part.split(" as ")[-1].strip()
            if local:
                out[local] = source
    return out


def _is_local_source(source: str) -> bool:
    return source.startswith((".", "@/", "~/"))


def resolve_import(importer: str, source: str, files: set[str]) -> str | None:
    """Resolve a relative or ``@/``/``~/`` alias import to a scope file, if any."""
    if source.startswith("."):
        bases = [posixpath.normpath(posixpath.join(posixpath.dirname(importer), source))]
    elif source.startswith(("@/", "~/")):
        rest = source[2:]
        bases = [f"src/{rest}", rest]
    else:
        return None
    for base in bases:
        candidates = [base, *(base + e for e in _RESOLVE_EXTENSIONS)]
        candidates += [f"{base}/index{e}" for e in _RESOLVE_EXTENSIONS]
        for c in candidates:
            if c in files:
                return c
    return None


def _component_name(rel: str, text: str) -> str:
    m = _COMPONENT_DEFAULT.search(text) or _COMPONENT_DECL.search(text)
    return m.group(1) if m else PurePosixPath(rel).stem.split(".")[0]


def _tag_before(text: str, pos: int) -> str | None:
    tags = _JSX_TAG.findall(text, 0, pos)
    return tags[-1] if tags else None


def _external_source(
    rel: str, text: str, tag: str | None, files: set[str], reader: _Reader
) -> str | None:
    """External package behind ``tag``: direct import, or one local wrapper hop."""
    if not tag:
        return None
    source = _imports(text).get(tag)
    if source is None:
        return None
    if not _is_local_source(source):
        return source
    wrapper = resolve_import(rel, source, files)
    wtext = reader.read(wrapper) if wrapper else None
    if not wtext:
        return None
    wimports = _imports(wtext)
    for wtag in _JSX_TAG.findall(wtext):
        src = wimports.get(wtag)
        if src and not _is_local_source(src) and _DISCLOSURE_TAG.search(wtag):
            return src
    return None


def _content_props(text: str) -> list[str]:
    seen: list[str] = []
    for name in _CONTENT_PROP.findall(text):
        if name not in seen:
            seen.append(name)
    return seen[:5]


def _used_by(component: str, rel: str, files: Iterable[str], reader: _Reader) -> list[str]:
    pattern = re.compile(
        rf"<{re.escape(component)}\b|import\s[^;]*?\b{re.escape(component)}\b[^;]*?from"
    )
    out = []
    for other in files:
        if other == rel:
            continue
        text = reader.read(other)
        if text and component in text and pattern.search(text):
            out.append(other)
    return sorted(out)


def scan(
    scope_root: Path,
    files: Sequence[str],
    all_files: Sequence[str] | None = None,
    changed: Iterable[str] | None = None,
) -> list[Hotspot]:
    """Scan ``files`` (scope-relative) for hotspots.

    Args:
        scope_root: Scope root directory on disk.
        files: Scope-relative files to scan.
        all_files: Scope-relative files used for consumer lookup and import
            resolution (defaults to ``files``).
        changed: Files changed by the PR. Hotspots in other files are marked
            ``context_only``. None means every file counts as changed (path mode).
    """
    reader = _Reader(scope_root)
    universe = [f for f in (all_files if all_files is not None else files) if is_scannable(f)]
    universe_set = set(universe)
    changed_set = set(changed) if changed is not None else None
    hotspots: list[Hotspot] = []

    for rel in files:
        if not is_scannable(rel):
            continue
        text = reader.read(rel)
        if not text:
            continue
        component: str | None = None
        for rule in RULES:
            if rule.requires is not None and not rule.requires.search(text):
                continue
            matches = list(rule.regex.finditer(text))
            if not matches:
                continue
            first = matches[0]
            line_no = text.count("\n", 0, first.start()) + 1
            line_text = text.splitlines()[line_no - 1].strip()
            component = component or _component_name(rel, text)
            hs = Hotspot(
                rule=rule.id,
                filename=rel,
                line=line_no,
                snippet=line_text[:120],
                component=component,
                extra=len(matches) - 1,
                content_props=_content_props(text),
                context_only=changed_set is not None and rel not in changed_set,
            )
            if rule.id == "disclosure-prop":
                hs.tag = _tag_before(text, first.start())
                hs.external_source = _external_source(rel, text, hs.tag, universe_set, reader)
            hs.used_by = _used_by(component, rel, universe, reader)
            hotspots.append(hs)

    hotspots.sort(key=lambda h: (-len(h.used_by), h.filename, h.rule))
    primary = [h for h in hotspots if not h.context_only]
    context = [h for h in hotspots if h.context_only]
    return _interleave_rules(primary) + _interleave_rules(context)


def _interleave_rules(hotspots: list[Hotspot]) -> list[Hotspot]:
    """Round-robin across rules (each keeps its own order) so a capped render
    shows every rule, not just the noisiest one."""
    by_rule: dict[str, list[Hotspot]] = {}
    for h in hotspots:
        by_rule.setdefault(h.rule, []).append(h)
    queues = [by_rule[r.id] for r in RULES if r.id in by_rule]
    depth = max((len(q) for q in queues), default=0)
    return [q[i] for i in range(depth) for q in queues if i < len(q)]


def neighbours(changed: Sequence[str], all_files: Sequence[str], scope_root: Path) -> list[str]:
    """One-hop import neighbours of ``changed`` within ``all_files`` (PR mode)."""
    reader = _Reader(scope_root)
    files = [f for f in all_files if is_scannable(f)]
    file_set = set(files)
    changed_set = set(changed)
    out: set[str] = set()
    for rel in changed:
        text = reader.read(rel) if is_scannable(rel) else None
        for source in _imports(text or "").values():
            target = resolve_import(rel, source, file_set)
            if target and target not in changed_set:
                out.add(target)
    for rel in files:
        if rel in changed_set or rel in out:
            continue
        text = reader.read(rel)
        if not text:
            continue
        for source in _imports(text).values():
            if resolve_import(rel, source, file_set) in changed_set:
                out.add(rel)
                break
    return sorted(out)


def list_scope_files(scope_root: Path, excluded_dirs: Sequence[str]) -> list[str]:
    """Tracked files under ``scope_root`` (``git ls-files``, rglob fallback), capped."""
    try:
        res = subprocess.run(
            ["git", "-C", str(scope_root), "ls-files", "-z", "--", "."],
            capture_output=True,
            timeout=60,
            check=True,
        )
        files = [f for f in res.stdout.decode("utf-8", "replace").split("\0") if f]
    except (subprocess.SubprocessError, OSError):
        files = []
        try:
            for p in scope_root.rglob("*"):
                if p.is_file() and not p.is_symlink():
                    files.append(p.relative_to(scope_root).as_posix())
        except OSError:
            pass
    excluded = set(excluded_dirs)
    files = [f for f in files if not excluded & set(f.split("/")[:-1]) and is_scannable(f)]
    if len(files) > MAX_SCOPE_FILES:
        logger.warning(
            "SEO hotspots: %d files under %s, only the first %d are used for lookups",
            len(files), scope_root, MAX_SCOPE_FILES,
        )
        files = sorted(files)[:MAX_SCOPE_FILES]
    return files


def rule_counts(hotspots: Sequence[Hotspot]) -> dict[str, int]:
    return dict(Counter(h.rule for h in hotspots))


def render(hotspots: Sequence[Hotspot], max_items: int = 40, max_chars: int = 4000) -> str:
    """Render hotspots as a compact text block ("" when there are none)."""
    if not hotspots:
        return ""
    head = ["HOTSPOTS (static pre-scan; verify each, dismiss false positives)"]
    rules_present = list(dict.fromkeys(h.rule for h in hotspots))
    head += [f"- {r}: {_RULE_HINTS[r]}" for r in rules_present if r in _RULE_HINTS]
    lines = list(head)
    size = sum(len(x) + 1 for x in lines)
    shown = 0
    for h in hotspots[:max_items]:
        ctx = " [context]" if h.context_only else ""
        entry = [f"[{h.rule}]{ctx} {h.filename}:{h.line} (+{h.extra} more)"]
        detail = [h.snippet]
        if h.content_props:
            detail.append(f"content props: {', '.join(h.content_props)}")
        if h.tag:
            via = f" from {h.external_source}" if h.external_source else ""
            detail.append(f"renders {h.tag}{via}")
        entry.append("  " + "; ".join(detail))
        if h.used_by:
            more = f" (+{len(h.used_by) - 3})" if len(h.used_by) > 3 else ""
            entry.append(f"  used by: {', '.join(h.used_by[:3])}{more}")
        entry_size = sum(len(x) + 1 for x in entry)
        if size + entry_size > max_chars:
            break
        lines += entry
        size += entry_size
        shown += 1
    if shown == 0:
        return ""
    if shown < len(hotspots):
        lines.append(f"(+{len(hotspots) - shown} more hotspots not shown)")
    return "\n".join(lines)


HOTSPOTS_DESC = (
    "Static regex pre-scan of render-gating shapes (scope-relative file:line). "
    "Candidates only: verify each with tools, dismiss false positives silently. "
    "Entries marked [context] are unchanged neighbours of the PR: context, never findings "
    "by themselves. This list is incomplete: always run your own sweep too."
)

_SIG_CACHE: dict[type, type] = {}


def with_hotspots(signature: type[dspy.Signature]) -> type[dspy.Signature]:
    """Return ``signature`` with a ``hotspots`` string input prepended (cached)."""
    if "hotspots" in signature.input_fields:
        return signature
    cached = _SIG_CACHE.get(signature)
    if cached is None:
        cached = signature.prepend("hotspots", dspy.InputField(desc=HOTSPOTS_DESC), type_=str)
        _SIG_CACHE[signature] = cached
    return cached
