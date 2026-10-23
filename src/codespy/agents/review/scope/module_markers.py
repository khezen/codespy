"""Module markers: explicit sub-module declarations in single-manifest modular monoliths.

A marker turns a directory into a scope without a package manifest (dependencies stay
on the enclosing manifest). Only explicit tool configuration counts (Packwerk, tach,
import-linter, go-arch-lint, Spring Modulith, JPMS, Go ``cmd/*``, TS project references,
Laravel modules, Deptrac, sbt and Mill). A marker that fails to parse or points to a
missing directory is ignored and never breaks scope resolution.
"""

from __future__ import annotations

import configparser
import logging
import re
import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from codespy.agents.review.scope.manifest_parser import load_jsonc

logger = logging.getLogger(__name__)

# subroot -> (kind, marker path)
Markers = dict[str, tuple[str, str]]

_GLOB_CHARS = "*?[{"
_REGEX_CHARS = ".*+?[(^$\\"
_SPRING_APP = re.compile(r".+Application\.(?:java|kt)$")
_SPRING_ANNOTATION = re.compile(r"@SpringBootApplication\b")
_SBT_PROJECT = re.compile(r"""project(?:\.in\s*\(|\s+in\s+)\s*file\s*\(\s*["']([^"']+)["']\s*\)""")
_MILL_OBJECT = re.compile(r"^object\s+`?(\w+)`?\s+extends\s+([^\n{]*)", re.MULTILINE)


def normalize_subroot(path_str: str, repo_path: Path) -> str | None:
    """Normalize a marker path to a repo-relative subroot; None when unsafe or missing.

    Strips ``./`` and trailing ``/``, rejects absolute paths and ``..``, and requires
    an existing directory. The repo root is returned as ``"."``.
    """
    path_str = path_str.strip().replace("\\", "/")
    if not path_str or path_str.startswith("/") or re.match(r"^[A-Za-z]:", path_str):
        return None
    parts = [p for p in path_str.split("/") if p not in ("", ".")]
    if ".." in parts:
        return None
    subroot = "/".join(parts) or "."
    if not _is_dir_inside(repo_path, subroot):
        return None
    return subroot


def _is_dir_inside(repo_path: Path, subroot: str) -> bool:
    """True when ``subroot`` is a directory that stays inside the repo after symlinks.

    Scope subroots root the review agents' filesystem tools, so a repo symlink such as
    ``evil -> /`` must never become a scope.
    """
    try:
        root = repo_path.resolve()
        target = (repo_path / subroot).resolve()
    except (OSError, RuntimeError):
        return False
    if not target.is_dir() or not target.is_relative_to(root):
        return False
    return subroot == "." or target != root


def _literal_prefix(pattern: str, special: str) -> str:
    """Return ``pattern`` up to the first special char, trimmed to whole path components."""
    for i, ch in enumerate(pattern):
        if ch in special:
            head = pattern[:i]
            return head.rsplit("/", 1)[0] if "/" in head else ""
    return pattern


def _ancestor_dirs(changed_files: Iterable[str]) -> list[str]:
    """Non-root ancestor directories of the changed files (sorted, shallow first)."""
    dirs: set[str] = set()
    for filename in changed_files:
        parts = filename.split("/")
        for depth in range(1, len(parts)):
            dirs.add("/".join(parts[:depth]))
    return sorted(dirs, key=lambda d: (d.count("/"), d))


def _add(markers: Markers, subroot: str | None, kind: str, marker_path: str) -> None:
    if subroot and subroot != ".":
        markers.setdefault(subroot, (kind, marker_path))


# --- per-module marker files (scanned in ancestor dirs of changed files) ---------------


def _packwerk(repo_path: Path, ancestors: list[str]) -> Markers:
    markers: Markers = {}
    if not (repo_path / "packwerk.yml").is_file():
        return markers
    for d in ancestors:
        if (repo_path / d / "package.yml").is_file():
            _add(markers, d, "packwerk", f"{d}/package.yml")
    return markers


def _laravel(repo_path: Path, ancestors: list[str]) -> Markers:
    markers: Markers = {}
    has_statuses = (repo_path / "modules_statuses.json").is_file()
    for d in ancestors:
        in_modules_dir = Path(d).parent.name == "Modules"
        if (in_modules_dir or has_statuses) and (repo_path / d / "module.json").is_file():
            _add(markers, d, "laravel", f"{d}/module.json")
    return markers


def _jpms(repo_path: Path, ancestors: list[str]) -> Markers:
    markers: Markers = {}
    for d in ancestors:
        if not (repo_path / d / "module-info.java").is_file():
            continue
        subroot = d
        if subroot == "src/main/java":
            continue  # root module: already the root scope
        if subroot.endswith("/src/main/java"):
            subroot = subroot[: -len("/src/main/java")]
        _add(markers, subroot, "jpms", f"{d}/module-info.java")
    return markers


def _spring_modulith(repo_path: Path, changed_files: list[str]) -> Markers:
    """Package of a ``@SpringBootApplication`` dir D -> ``D/<next component>``."""
    markers: Markers = {}
    app_dirs: dict[str, str] = {}  # dir -> application file
    for d in ["."] + _ancestor_dirs(changed_files):
        dir_path = repo_path / d
        try:
            candidates = sorted(p for p in dir_path.iterdir() if _SPRING_APP.match(p.name))
        except OSError:
            continue
        for app in candidates:
            try:
                if _SPRING_ANNOTATION.search(app.read_text(encoding="utf-8", errors="ignore")):
                    app_dirs[d] = app.name if d == "." else f"{d}/{app.name}"
                    break
            except OSError:
                continue
    if not app_dirs:
        return markers
    for filename in changed_files:
        parts = filename.split("/")
        # Deepest application dir enclosing the file
        for depth in range(len(parts) - 2, -1, -1):
            d = "/".join(parts[:depth]) or "."
            if d in app_dirs:
                subroot = "/".join(parts[: depth + 1])
                if _is_dir_inside(repo_path, subroot):
                    _add(markers, subroot, "spring-modulith", app_dirs[d])
                break
    return markers


def _go_cmd(
    repo_path: Path, changed_files: list[str], manifests: dict[Path, tuple[str, str]]
) -> Markers:
    """``cmd/<name>`` binaries of a single root ``go.mod`` module."""
    markers: Markers = {}
    if manifests.get(Path(".")) != ("go", "go.mod") or Path("cmd") in manifests:
        return markers
    for filename in changed_files:
        parts = filename.split("/")
        if len(parts) >= 3 and parts[0] == "cmd":
            subroot = f"cmd/{parts[1]}"
            if (repo_path / subroot / "main.go").is_file():
                _add(markers, subroot, "go-cmd", f"{subroot}/main.go")
    return markers


def _mill_packages(repo_path: Path, ancestors: list[str]) -> Markers:
    markers: Markers = {}
    for d in ancestors:
        if (repo_path / d / "package.mill").is_file():
            _add(markers, d, "mill", f"{d}/package.mill")
    return markers


# --- root configuration files -----------------------------------------------------------


def _tach(repo_path: Path) -> Markers:
    markers: Markers = {}
    path = repo_path / "tach.toml"
    if not path.is_file():
        return markers
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    source_roots = data.get("source_roots") or ["."]
    for module in data.get("modules") or []:
        mod_path = module.get("path") if isinstance(module, dict) else None
        if not isinstance(mod_path, str) or mod_path == "<root>":
            continue
        rel = mod_path.replace(".", "/")
        for root in source_roots:
            if not isinstance(root, str):
                continue
            subroot = normalize_subroot(f"{root}/{rel}", repo_path)
            if subroot:
                _add(markers, subroot, "tach", "tach.toml")
                break
    return markers


def _split_names(value: Any) -> list[str]:
    """import-linter list value (multi-line string or list) -> dotted names."""
    items = value.splitlines() if isinstance(value, str) else value
    if not isinstance(items, list):
        return []
    names: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        for part in item.split("|"):
            part = part.strip().strip("()").strip()
            if part:
                names.append(part)
    return names


def _importlinter_contracts(repo_path: Path) -> tuple[str, list[dict[str, Any]]] | None:
    ini_sources = [
        (repo_path / ".importlinter", ".importlinter"),
        (repo_path / "setup.cfg", "setup.cfg"),
    ]
    for path, label in ini_sources:
        if not path.is_file():
            continue
        parser = configparser.ConfigParser()
        parser.read(path, encoding="utf-8")
        if not parser.has_section("importlinter"):
            continue
        contracts = [
            dict(parser.items(section))
            for section in parser.sections()
            if section.startswith("importlinter:contract:")
        ]
        return label, contracts
    pyproject = repo_path / "pyproject.toml"
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        config = data.get("tool", {}).get("importlinter")
        if isinstance(config, dict):
            contracts = [c for c in config.get("contracts") or [] if isinstance(c, dict)]
            return "pyproject.toml", contracts
    return None


def _import_linter(repo_path: Path) -> Markers:
    markers: Markers = {}
    found = _importlinter_contracts(repo_path)
    if found is None:
        return markers
    label, contracts = found
    for contract in contracts:
        containers = _split_names(contract.get("containers", []))
        names = _split_names(contract.get("modules", [])) + _split_names(
            contract.get("source_modules", [])
        )
        names += containers
        for layer in _split_names(contract.get("layers", [])):
            if containers:
                names.extend(f"{c}.{layer}" for c in containers)
            else:
                names.append(layer)
        for name in names:
            rel = name.replace(".", "/")
            subroot = normalize_subroot(rel, repo_path) or normalize_subroot(
                f"src/{rel}", repo_path
            )
            _add(markers, subroot, "import-linter", label)
    return markers


def _load_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _go_arch_lint(repo_path: Path) -> Markers:
    markers: Markers = {}
    path = repo_path / ".go-arch-lint.yml"
    if not path.is_file():
        return markers
    data = _load_yaml(path)
    if not isinstance(data, dict):
        return markers
    workdir = data.get("workdir") if isinstance(data.get("workdir"), str) else "."
    components = data.get("components")
    if not isinstance(components, dict):
        return markers
    for component in components.values():
        globs = component.get("in") if isinstance(component, dict) else None
        if isinstance(globs, str):
            globs = [globs]
        for glob in globs if isinstance(globs, list) else []:
            if not isinstance(glob, str):
                continue
            prefix = _literal_prefix(glob, _GLOB_CHARS)
            if prefix:
                _add(
                    markers,
                    normalize_subroot(f"{workdir}/{prefix}", repo_path),
                    "go-arch-lint",
                    ".go-arch-lint.yml",
                )
    return markers


def _ts_references(repo_path: Path) -> Markers:
    markers: Markers = {}
    path = repo_path / "tsconfig.json"
    if not path.is_file():
        return markers
    data = load_jsonc(path)
    refs = data.get("references") if isinstance(data, dict) else None
    for ref in refs if isinstance(refs, list) else []:
        ref_path = ref.get("path") if isinstance(ref, dict) else None
        if not isinstance(ref_path, str):
            continue
        if ref_path.endswith(".json"):
            ref_path = ref_path.rsplit("/", 1)[0] if "/" in ref_path else "."
        _add(markers, normalize_subroot(ref_path, repo_path), "ts-references", "tsconfig.json")
    return markers


def _deptrac(repo_path: Path) -> Markers:
    markers: Markers = {}
    for name in ("deptrac.yaml", "deptrac.yml", "depfile.yaml"):
        path = repo_path / name
        if path.is_file():
            break
    else:
        return markers
    data = _load_yaml(path)
    if not isinstance(data, dict):
        return markers
    section = data.get("deptrac")
    layers = section.get("layers") if isinstance(section, dict) else data.get("layers")
    for layer in layers if isinstance(layers, list) else []:
        collectors = layer.get("collectors") if isinstance(layer, dict) else None
        for collector in collectors if isinstance(collectors, list) else []:
            if not isinstance(collector, dict) or collector.get("type") != "directory":
                continue
            value = collector.get("value") or collector.get("regex")
            if isinstance(value, str):
                prefix = _literal_prefix(value, _REGEX_CHARS)
                _add(
                    markers,
                    normalize_subroot(prefix, repo_path) if prefix else None,
                    "deptrac",
                    name,
                )
    return markers


def _sbt(repo_path: Path) -> Markers:
    markers: Markers = {}
    path = repo_path / "build.sbt"
    if not path.is_file():
        return markers
    for match in _SBT_PROJECT.finditer(path.read_text(encoding="utf-8")):
        _add(markers, normalize_subroot(match.group(1), repo_path), "sbt", "build.sbt")
    return markers


def _mill_root(repo_path: Path) -> Markers:
    markers: Markers = {}
    for name in ("build.mill", "build.sc"):
        path = repo_path / name
        if not path.is_file():
            continue
        for match in _MILL_OBJECT.finditer(path.read_text(encoding="utf-8")):
            if "Module" in match.group(2):
                _add(markers, normalize_subroot(match.group(1), repo_path), "mill", name)
        break
    return markers


def discover_module_markers(
    repo_path: Path,
    changed_files: Iterable[str],
    manifests: dict[Path, tuple[str, str]],
) -> Markers:
    """Discover module-marker scopes relevant to the changed files.

    Args:
        repo_path: Repository root
        changed_files: Changed file paths relative to the repo root
        manifests: Manifest dir -> (package manager, filename) from manifest discovery

    Returns:
        Subroot -> (kind, marker path). Subroots that already have a manifest are
        omitted (the manifest scope wins). Markers without changed files are harmless:
        the resolver drops scopes that own no changed file.
    """
    files = list(changed_files)
    ancestors = _ancestor_dirs(files)
    parsers: list[tuple[str, Callable[[], Markers]]] = [
        ("packwerk", lambda: _packwerk(repo_path, ancestors)),
        ("tach", lambda: _tach(repo_path)),
        ("import-linter", lambda: _import_linter(repo_path)),
        ("go-arch-lint", lambda: _go_arch_lint(repo_path)),
        ("spring-modulith", lambda: _spring_modulith(repo_path, files)),
        ("jpms", lambda: _jpms(repo_path, ancestors)),
        ("go-cmd", lambda: _go_cmd(repo_path, files, manifests)),
        ("ts-references", lambda: _ts_references(repo_path)),
        ("laravel", lambda: _laravel(repo_path, ancestors)),
        ("deptrac", lambda: _deptrac(repo_path)),
        ("sbt", lambda: _sbt(repo_path)),
        ("mill", lambda: _mill_root(repo_path)),
        ("mill", lambda: _mill_packages(repo_path, ancestors)),
    ]
    found: Markers = {}
    for kind, parse in parsers:
        try:
            markers = parse()
        except Exception as e:  # malformed config must never break resolution
            logger.debug("Ignoring %s module markers: %s", kind, e)
            continue
        for subroot, marker in markers.items():
            if Path(subroot) in manifests or subroot in found:
                continue
            if not _is_dir_inside(repo_path, subroot):
                logger.debug(
                    "Ignoring %s marker %s: not a directory inside the repo", kind, subroot
                )
                continue
            found[subroot] = marker
    return found
