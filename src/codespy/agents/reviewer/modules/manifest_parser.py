"""Package manifest parser for extracting package names from various manifest files."""

from __future__ import annotations

import configparser
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

# Mapping of package manager to ecosystem name
PACKAGE_MANAGER_TO_ECOSYSTEM: dict[str, str] = {
    "npm": "npm",
    "go": "Go",
    "pip": "PyPI",
    "cargo": "crates.io",
    "maven": "Maven",
    "gradle": "Maven",
    "sbt": "Maven",
    "composer": "Packagist",
    "bundler": "RubyGems",
    "dotnet": "NuGet",
    "swift": "SwiftURL",
    "pub": "Pub",
    "mix": "Hex",
    "helm": "Helm",
    "clojure": "Clojure",
    "leiningen": "Clojure",
    "stack": "Hackage",
    "cabal": "Hackage",
    "dune": "opam",
    "zig": "Zig",
    "cpan": "CPAN",
    "r": "CRAN",
}

# Git hosts for repo inference
_GIT_HOSTS = ("github.com/", "gitlab.com/", "bitbucket.org/")


def _infer_repo_from_url(url: str) -> str | None:
    """Extract owner/repo from git URL (e.g., https://github.com/owner/repo.git)."""
    for host in _GIT_HOSTS:
        for scheme in (f"https://{host}", f"http://{host}", f"git@{host.rstrip('/')}:"):
            if url.startswith(scheme):
                path = url[len(scheme):].rstrip("/").removesuffix(".git")
                parts = path.split("/")
                if len(parts) >= 2:
                    return f"{parts[0]}/{parts[1]}"
    return None


def _infer_repo_from_name(name: str) -> str | None:
    """Extract owner/repo from a name with git host prefix (e.g., Go module)."""
    for host in _GIT_HOSTS:
        if name.startswith(host):
            parts = name[len(host):].split("/")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
    return None


def _infer_repo_from_path(path: str) -> str | None:
    """Extract owner/repo from a file path containing a git host (vendored deps)."""
    for host in _GIT_HOSTS:
        idx = path.find(host)
        if idx >= 0:
            remainder = path[idx + len(host):]
            parts = remainder.split("/")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
    return None


def infer_repo_from_name(name: str) -> str | None:
    """Public wrapper for _infer_repo_from_name."""
    return _infer_repo_from_name(name)


def extract_dependencies(manifest_path: str, repo_path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract production dependency names and inferred source repos from manifest.

    Args:
        manifest_path: Relative path to manifest file from repo root
        repo_path: Path to the repository root

    Returns:
        Tuple of:
        - dependency_names: list of all production dep names
        - dependency_repos: dict mapping dep name -> owner/repo for identifiable deps
    """
    full_path = repo_path / manifest_path
    if not full_path.exists():
        return [], {}

    filename = Path(manifest_path).name

    try:
        if filename == "package.json":
            return _extract_deps_from_package_json(full_path)
        elif filename == "go.mod":
            return _extract_deps_from_go_mod(full_path)
        elif filename == "pyproject.toml":
            return _extract_deps_from_pyproject_toml(full_path)
        elif filename == "Cargo.toml":
            return _extract_deps_from_cargo_toml(full_path)
        elif filename == "pom.xml":
            return _extract_deps_from_pom_xml(full_path)
        elif filename == "composer.json":
            return _extract_deps_from_composer_json(full_path)
        elif filename == "pubspec.yaml":
            return _extract_deps_from_pubspec_yaml(full_path)
        elif filename == "Gemfile":
            return _extract_deps_from_gemfile(full_path)
        elif filename == "mix.exs":
            return _extract_deps_from_mix_exs(full_path)
        elif filename.endswith(".csproj"):
            return _extract_deps_from_csproj(full_path)
        elif filename == "Package.swift":
            return _extract_deps_from_swift_package(full_path)
        elif filename in ("build.gradle", "build.gradle.kts"):
            return _extract_deps_from_gradle(full_path)
        elif filename == "setup.cfg":
            return _extract_deps_from_setup_cfg(full_path)
    except Exception:
        return [], {}

    return [], {}


def _extract_deps_from_package_json(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from package.json (production only, skip dev/peer/optional)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        deps = data.get("dependencies", {})
        if not isinstance(deps, dict):
            return [], {}

        names = list(deps.keys())
        repos: dict[str, str] = {}

        for name, spec in deps.items():
            if isinstance(spec, str):
                # Parse git URLs: "github:owner/repo" or "git+https://..."
                if spec.startswith("github:"):
                    repo_path = spec[7:].split("#")[0]  # Remove any #ref
                    if "/" in repo_path:
                        repos[name] = repo_path
                elif spec.startswith("git+"):
                    inferred = _infer_repo_from_url(spec[4:])
                    if inferred:
                        repos[name] = inferred

        return names, repos
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return [], {}


def _extract_deps_from_go_mod(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from go.mod (filter // indirect lines)."""
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()

        names: list[str] = []
        repos: dict[str, str] = {}

        # Parse require block
        in_require = False
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("require ("):
                in_require = True
                continue
            if in_require and line == ")":
                in_require = False
                continue
            if not in_require and line.startswith("require "):
                # Single-line require
                parts = line[8:].strip().split()
                if parts:
                    line = parts[0]
                else:
                    continue

            if in_require or (line and not line.startswith("require ")):
                # Skip indirect deps
                if "// indirect" in line:
                    continue
                # Extract module path
                parts = line.split()
                if parts:
                    module_path = parts[0]
                    names.append(module_path)
                    # Go modules always have host in path
                    inferred = _infer_repo_from_name(module_path)
                    if inferred:
                        repos[module_path] = inferred

        return names, repos
    except (UnicodeDecodeError, OSError):
        return [], {}


def _strip_pep508_extras(name: str) -> str:
    """Strip extras and version specifiers from PEP 508 dependency name."""
    # Handle name[extra] -> name
    if "[" in name:
        name = name.split("[")[0]
    # Handle version specifiers (>=, ==, ~=, etc.)
    for op in (">=", "<=", ">", "<", "==", "!=", "~=", "==="):
        if op in name:
            name = name.split(op)[0].strip()
    return name.strip()


def _extract_deps_from_pyproject_toml(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from pyproject.toml (PEP 508 or Poetry)."""
    try:
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)

        names: list[str] = []
        repos: dict[str, str] = {}

        # Try [project.dependencies] (PEP 508)
        project = data.get("project")
        if isinstance(project, dict):
            deps = project.get("dependencies", [])
            if isinstance(deps, list):
                for dep in deps:
                    if isinstance(dep, str):
                        name = _strip_pep508_extras(dep)
                        if name and name != "python":
                            names.append(name)

        # Try [tool.poetry.dependencies]
        tool = data.get("tool")
        if isinstance(tool, dict):
            poetry = tool.get("poetry")
            if isinstance(poetry, dict):
                poetry_deps = poetry.get("dependencies", {})
                if isinstance(poetry_deps, dict):
                    for name, spec in poetry_deps.items():
                        if name == "python":
                            continue
                        names.append(name)
                        # Check for git or path source
                        if isinstance(spec, dict):
                            git_url = spec.get("git")
                            if git_url and isinstance(git_url, str):
                                inferred = _infer_repo_from_url(git_url)
                                if inferred:
                                    repos[name] = inferred
                            path_val = spec.get("path")
                            if path_val and isinstance(path_val, str):
                                inferred = _infer_repo_from_path(path_val)
                                if inferred:
                                    repos[name] = inferred

        return names, repos
    except Exception:
        return [], {}


def _extract_deps_from_cargo_toml(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from Cargo.toml (skip dev-dependencies, build-dependencies)."""
    try:
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)

        names: list[str] = []
        repos: dict[str, str] = {}

        deps = data.get("dependencies", {})
        if isinstance(deps, dict):
            for name, spec in deps.items():
                names.append(name)
                if isinstance(spec, dict):
                    git_url = spec.get("git")
                    if git_url and isinstance(git_url, str):
                        inferred = _infer_repo_from_url(git_url)
                        if inferred:
                            repos[name] = inferred

        return names, repos
    except Exception:
        return [], {}


def _extract_deps_from_pom_xml(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from pom.xml (skip test scope)."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        ns = {"m": "http://maven.apache.org/POM/4.0.0"}

        names: list[str] = []

        deps = root.find("m:dependencies", ns)
        if deps is None:
            deps = root.find("dependencies")

        if deps is not None:
            for dep in deps.findall("m:dependency", ns) if deps else []:
                scope = dep.find("m:scope", ns)
                if scope is not None and scope.text == "test":
                    continue
                group = dep.find("m:groupId", ns)
                artifact = dep.find("m:artifactId", ns)
                if group is not None and artifact is not None:
                    names.append(f"{group.text}:{artifact.text}")

        return names, {}
    except ET.ParseError:
        return [], {}


def _extract_deps_from_composer_json(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from composer.json (exclude php, ext-*)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        require = data.get("require", {})
        if not isinstance(require, dict):
            return [], {}

        names = [name for name in require
                 if not name.startswith("php") and not name.startswith("ext-")]

        return names, {}
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return [], {}


def _extract_deps_from_pubspec_yaml(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from pubspec.yaml (exclude flutter packages)."""
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        deps = data.get("dependencies", {})
        if not isinstance(deps, dict):
            return [], {}

        excluded = {"flutter", "flutter_test", "flutter_localizations"}
        names: list[str] = []
        repos: dict[str, str] = {}

        for name, spec in deps.items():
            if name in excluded:
                continue
            names.append(name)
            if isinstance(spec, dict):
                git_url = spec.get("git")
                if isinstance(git_url, str):
                    inferred = _infer_repo_from_url(git_url)
                    if inferred:
                        repos[name] = inferred
                elif isinstance(git_url, dict):
                    url = git_url.get("url")
                    if url and isinstance(url, str):
                        inferred = _infer_repo_from_url(url)
                        if inferred:
                            repos[name] = inferred

        return names, repos
    except Exception:
        return [], {}


def _extract_deps_from_gemfile(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from Gemfile (skip dev/test groups)."""
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()

        names: list[str] = []
        repos: dict[str, str] = {}

        # Track if we're in a dev/test group
        in_dev_group = False
        group_depth = 0

        for line in content.split("\n"):
            line_stripped = line.strip()

            # Track group blocks
            if line_stripped.startswith("group "):
                if ":development" in line_stripped or ":test" in line_stripped:
                    in_dev_group = True
                group_depth += 1
                continue

            if line_stripped == "end" and group_depth > 0:
                group_depth -= 1
                if group_depth == 0:
                    in_dev_group = False
                continue

            if in_dev_group:
                continue

            # Parse gem lines
            match = re.match(r"gem\s+['\"]([^'\"]+)['\"]", line_stripped)
            if match:
                name = match.group(1)
                names.append(name)

                # Check for git or github option
                git_match = re.search(r"git:\s*['\"]([^'\"]+)['\"]", line)
                if git_match:
                    inferred = _infer_repo_from_url(git_match.group(1))
                    if inferred:
                        repos[name] = inferred

                github_match = re.search(r"github:\s*['\"]([^'\"]+)['\"]", line)
                if github_match:
                    gh_path = github_match.group(1)
                    repos[name] = gh_path if "/" in gh_path else f"{gh_path}/{gh_path}"

        return names, repos
    except (UnicodeDecodeError, OSError):
        return [], {}


def _extract_deps_from_mix_exs(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from mix.exs (skip dev/test only)."""
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()

        names: list[str] = []
        repos: dict[str, str] = {}

        # Find deps function
        deps_match = re.search(r"defp?\s+deps\s*do\s*\[(.*?)\]\s*end", content, re.DOTALL)
        if not deps_match:
            return [], {}

        deps_block = deps_match.group(1)

        # Parse each dep tuple
        for dep_match in re.finditer(r"\{([^}]+)\}", deps_block):
            dep_str = dep_match.group(1)
            # Skip if only: :dev or only: :test
            if "only: :dev" in dep_str or "only: :test" in dep_str:
                continue

            # Extract name (first atom or string)
            name_match = re.match(r":([a-z_][a-zA-Z0-9_]*)|\"([^\"]+)\"", dep_str.strip())
            if name_match:
                name = name_match.group(1) or name_match.group(2)
                if name:
                    names.append(name)

                    # Check for github option
                    gh_match = re.search(r"github:\s*\"([^\"]+)\"", dep_str)
                    if gh_match:
                        gh_path = gh_match.group(1)
                        repos[name] = gh_path if "/" in gh_path else f"{gh_path}/{gh_path}"

        return names, repos
    except (UnicodeDecodeError, OSError):
        return [], {}


def _extract_deps_from_csproj(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from .csproj (skip PrivateAssets=All)."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        ns = {"p": "http://schemas.microsoft.com/developer/msbuild/2003"}

        names: list[str] = []

        for ref in root.findall(".//p:PackageReference", ns):
            private = ref.get("PrivateAssets")
            if private == "All":
                continue
            include = ref.get("Include")
            if include:
                names.append(include)

        # Also try without namespace
        if not names:
            for ref in root.findall(".//PackageReference"):
                private = ref.get("PrivateAssets")
                if private == "All":
                    continue
                include = ref.get("Include")
                if include:
                    names.append(include)

        return names, {}
    except ET.ParseError:
        return [], {}


def _extract_deps_from_swift_package(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from Package.swift."""
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()

        names: list[str] = []
        repos: dict[str, str] = {}

        # Match .package(url: "...", ...)
        for match in re.finditer(r'\.package\s*\([^)]*url:\s*["\']([^"\']+)["\']', content):
            url = match.group(1)
            inferred = _infer_repo_from_url(url)
            if inferred:
                # Use repo name as dep name for Swift
                dep_name = inferred.split("/")[-1]
                names.append(dep_name)
                repos[dep_name] = inferred
            else:
                # Extract name from URL
                parts = url.rstrip("/").split("/")
                if parts:
                    name = parts[-1].removesuffix(".git")
                    names.append(name)

        return names, repos
    except (UnicodeDecodeError, OSError):
        return [], {}


def _extract_deps_from_gradle(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from build.gradle/build.gradle.kts (skip test/debug)."""
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()

        names: list[str] = []

        # Match implementation, api, compile dependencies
        for match in re.finditer(r"(implementation|api|compile)\s*['\"]([^'\"]+)['\"]", content):
            coord = match.group(2)
            # Skip test/debug variants
            if not coord.startswith("test") and not coord.startswith("debug"):
                names.append(coord)

        # Match Kotlin DSL: implementation("...")
        for match in re.finditer(r"(implementation|api)\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", content):
            coord = match.group(2)
            if not coord.startswith("test") and not coord.startswith("debug"):
                names.append(coord)

        return names, {}
    except (UnicodeDecodeError, OSError):
        return [], {}


def _extract_deps_from_setup_cfg(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract deps from setup.cfg [options] install_requires."""
    try:
        config = configparser.ConfigParser()
        config.read(path, encoding="utf-8")

        names: list[str] = []

        if config.has_option("options", "install_requires"):
            deps_str = config.get("options", "install_requires")
            for line in deps_str.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    name = _strip_pep508_extras(line)
                    if name:
                        names.append(name)

        return names, {}
    except (configparser.Error, UnicodeDecodeError, OSError):
        return [], {}


def extract_package_name(manifest_path: str, repo_path: Path) -> str | None:
    """Extract package name from manifest file. Returns None on failure.

    Args:
        manifest_path: Relative path to manifest file from repo root
        repo_path: Path to the repository root

    Returns:
        Package name string or None if extraction fails
    """
    full_path = repo_path / manifest_path
    if not full_path.exists():
        return None

    filename = Path(manifest_path).name

    try:
        if filename == "package.json" or filename == "composer.json":
            return _extract_from_json(full_path, ["name"])
        elif filename == "go.mod":
            return _extract_from_go_mod(full_path)
        elif filename == "pyproject.toml":
            return _extract_from_pyproject_toml(full_path)
        elif filename == "Cargo.toml":
            return _extract_from_toml(full_path, ["package", "name"])
        elif filename == "pubspec.yaml" or filename == "Chart.yaml":
            return _extract_from_yaml(full_path, ["name"])
        elif filename == "pom.xml":
            return _extract_from_pom_xml(full_path)
        elif filename == "setup.cfg":
            return _extract_from_setup_cfg(full_path)
        elif filename.endswith(".csproj") or filename.endswith(".fsproj") or filename.endswith(".vbproj"):
            return _extract_from_dotnet_proj(full_path)
        elif filename in ("build.gradle", "build.gradle.kts"):
            return _extract_from_gradle(repo_path, manifest_path)
        elif filename == "Gemfile":
            return _extract_from_gemfile(full_path)
        elif filename == "Package.swift":
            return _extract_from_swift_package(full_path)
        elif filename == "mix.exs":
            return _extract_from_mix_exs(full_path)
    except Exception:
        return None

    return None


def _extract_from_json(path: Path, keys: list[str]) -> str | None:
    """Extract value from JSON file following key path."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        value = data
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
            if value is None:
                return None
        return value if isinstance(value, str) else None
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def _extract_from_go_mod(path: Path) -> str | None:
    """Extract module name from go.mod file."""
    try:
        with open(path, encoding="utf-8") as f:
            first_line = f.readline().strip()
        match = re.match(r"^module\s+(\S+)", first_line)
        return match.group(1) if match else None
    except (UnicodeDecodeError, OSError):
        return None


def _extract_from_toml(path: Path, keys: list[str]) -> str | None:
    """Extract value from TOML file following key path."""
    try:
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
        value = data
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
            if value is None:
                return None
        return value if isinstance(value, str) else None
    except Exception:
        return None


def _extract_from_pyproject_toml(path: Path) -> str | None:
    """Extract package name from pyproject.toml.

    Tries [project][name] first, then [tool][poetry][name].
    """
    try:
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Try [project][name] first
        project = data.get("project")
        if isinstance(project, dict):
            name = project.get("name")
            if isinstance(name, str):
                return name

        # Fall back to [tool][poetry][name]
        tool = data.get("tool")
        if isinstance(tool, dict):
            poetry = tool.get("poetry")
            if isinstance(poetry, dict):
                name = poetry.get("name")
                if isinstance(name, str):
                    return name

        return None
    except Exception:
        return None


def _extract_from_yaml(path: Path, keys: list[str]) -> str | None:
    """Extract value from YAML file following key path."""
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        value = data
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
            if value is None:
                return None
        return value if isinstance(value, str) else None
    except Exception:
        return None


def _extract_from_pom_xml(path: Path) -> str | None:
    """Extract package name from Maven pom.xml as 'groupId:artifactId'."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        # Handle namespaced XML
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}

        group_id = root.find("m:groupId", ns)
        if group_id is None:
            group_id = root.find("groupId")

        artifact_id = root.find("m:artifactId", ns)
        if artifact_id is None:
            artifact_id = root.find("artifactId")

        if group_id is not None and artifact_id is not None:
            return f"{group_id.text}:{artifact_id.text}"
        elif artifact_id is not None:
            return artifact_id.text
        return None
    except ET.ParseError:
        return None


def _extract_from_setup_cfg(path: Path) -> str | None:
    """Extract package name from setup.cfg [metadata] section."""
    try:
        config = configparser.ConfigParser()
        config.read(path, encoding="utf-8")
        if config.has_option("metadata", "name"):
            return config.get("metadata", "name")
        return None
    except (configparser.Error, UnicodeDecodeError, OSError):
        return None


def _extract_from_dotnet_proj(path: Path) -> str | None:
    """Extract package name from .NET project file.

    Tries PackageId first, then RootNamespace, then filename stem.
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        # Handle namespaced XML
        ns = {"p": "http://schemas.microsoft.com/developer/msbuild/2003"}

        # Try PackageId
        package_id = root.find(".//p:PackageId", ns)
        if package_id is None:
            package_id = root.find(".//PackageId")
        if package_id is not None and package_id.text:
            return package_id.text

        # Try RootNamespace
        root_ns = root.find(".//p:RootNamespace", ns)
        if root_ns is None:
            root_ns = root.find(".//RootNamespace")
        if root_ns is not None and root_ns.text:
            return root_ns.text

        # Fall back to filename stem
        return path.stem
    except ET.ParseError:
        return path.stem


def _extract_from_gradle(repo_path: Path, manifest_path: str) -> str | None:
    """Extract project name from Gradle settings.gradle or settings.gradle.kts.

    Gradle projects don't store the name in build.gradle - it's in settings.gradle.
    This is a best-effort extraction.
    """
    manifest_dir = Path(manifest_path).parent

    # Look for settings.gradle or settings.gradle.kts
    settings_files = ["settings.gradle", "settings.gradle.kts"]

    for settings_file in settings_files:
        settings_path = repo_path / manifest_dir / settings_file
        if settings_path.exists():
            try:
                with open(settings_path, encoding="utf-8") as f:
                    content = f.read()
                # Look for rootProject.name = 'name' or rootProject.name = "name"
                match = re.search(
                    r"rootProject\.name\s*=\s*['\"]([^'\"]+)['\"]",
                    content,
                )
                if match:
                    return match.group(1)
            except (UnicodeDecodeError, OSError):
                continue

    return None


def _extract_from_gemfile(path: Path) -> str | None:
    """Extract gem name from Gemfile.

    Looks for 'source' line to extract the gem name, but this is often
    not present. Returns None as Gemfiles don't reliably contain a package name.
    """
    # Gemfiles don't typically contain a reliable package name
    # They reference gems to install, not the current package name
    return None


def _extract_from_swift_package(path: Path) -> str | None:
    """Extract package name from Package.swift.

    Looks for 'name: "..."' in the Package initialization.
    """
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # Look for Package(name: "...")
        match = re.search(
            r"Package\s*\([^)]*name:\s*['\"]([^'\"]+)['\"]",
            content,
            re.DOTALL,
        )
        if match:
            return match.group(1)
        return None
    except (UnicodeDecodeError, OSError):
        return None


def _extract_from_mix_exs(path: Path) -> str | None:
    """Extract project name from mix.exs.

    Looks for 'def project do' and extracts the 'app:' value.
    """
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # Look for app: :name or app: "name"
        match = re.search(r"app:\s*[:\"]([^\"\s,)]+)[\"\s,)]", content)
        if match:
            return match.group(1)
        return None
    except (UnicodeDecodeError, OSError):
        return None
