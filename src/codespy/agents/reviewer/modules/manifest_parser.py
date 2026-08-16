"""Package manifest parser for extracting package names from various manifest files."""

from __future__ import annotations

import configparser
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET


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
        if filename == "package.json":
            return _extract_from_json(full_path, ["name"])
        elif filename == "composer.json":
            return _extract_from_json(full_path, ["name"])
        elif filename == "go.mod":
            return _extract_from_go_mod(full_path)
        elif filename == "pyproject.toml":
            return _extract_from_pyproject_toml(full_path)
        elif filename == "Cargo.toml":
            return _extract_from_toml(full_path, ["package", "name"])
        elif filename == "pubspec.yaml":
            return _extract_from_yaml(full_path, ["name"])
        elif filename == "Chart.yaml":
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
        with open(path, "r", encoding="utf-8") as f:
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
        with open(path, "r", encoding="utf-8") as f:
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

        with open(path, "r", encoding="utf-8") as f:
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
                with open(settings_path, "r", encoding="utf-8") as f:
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
        with open(path, "r", encoding="utf-8") as f:
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
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Look for app: :name or app: "name"
        match = re.search(r"app:\s*[:\"]([^\"\s,)]+)[\"\s,)]", content)
        if match:
            return match.group(1)
        return None
    except (UnicodeDecodeError, OSError):
        return None
