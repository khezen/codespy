"""Git platform configuration and auto-discovery (GitHub and GitLab)."""

import logging
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global variables to track token sources
_github_token_source: str = "not set"
_gitlab_token_source: str = "not set"


def get_github_token_source() -> str:
    """Get the source of the GitHub token."""
    return _github_token_source


def get_gitlab_token_source() -> str:
    """Get the source of the GitLab token."""
    return _gitlab_token_source


def set_github_token_source(source: str) -> None:
    """Set the source of the GitHub token."""
    global _github_token_source
    _github_token_source = source


def set_gitlab_token_source(source: str) -> None:
    """Set the source of the GitLab token."""
    global _gitlab_token_source
    _gitlab_token_source = source


def discover_github_token() -> tuple[str | None, str]:
    """Try to discover GitHub token from local environment.

    Returns:
        Tuple of (token, source) where source describes where the token was found.
    """
    # 1. Check environment variables
    for env_var in ("GITHUB_TOKEN", "GH_TOKEN"):
        if token := os.environ.get(env_var):
            return token, f"environment variable ${env_var}"

    # 2. Try GitHub CLI (gh auth token)
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip(), "GitHub CLI (gh auth token)"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Try git credential helper
    try:
        result = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n",
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("password="):
                    return line.split("=", 1)[1], "git credential helper"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 4. Try .netrc file
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            content = netrc_path.read_text()
            in_github = False
            for line in content.split("\n"):
                line = line.strip()
                if "github.com" in line:
                    in_github = True
                if in_github and line.startswith("password"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1], "~/.netrc file"
                if in_github and line.startswith("machine") and "github.com" not in line:
                    in_github = False
        except Exception:
            pass

    return None, "not found"


def discover_gitlab_token() -> tuple[str | None, str]:
    """Try to discover GitLab token from local environment.

    Returns:
        Tuple of (token, source) where source describes where the token was found.
    """
    # 1. Check environment variables
    for env_var in ("GITLAB_TOKEN", "GITLAB_PRIVATE_TOKEN", "CI_JOB_TOKEN"):
        if token := os.environ.get(env_var):
            return token, f"environment variable ${env_var}"

    # 2. Try GitLab CLI (glab auth token) - if installed
    try:
        result = subprocess.run(
            ["glab", "auth", "status", "-t"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse token from output
            for line in result.stdout.split("\n"):
                if "Token:" in line:
                    token = line.split("Token:")[-1].strip()
                    if token and token != "***":
                        return token, "GitLab CLI (glab auth status)"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Try git credential helper for gitlab.com
    try:
        result = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=gitlab.com\n",
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("password="):
                    return line.split("=", 1)[1], "git credential helper"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 4. Try .netrc file
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            content = netrc_path.read_text()
            in_gitlab = False
            for line in content.split("\n"):
                line = line.strip()
                if "gitlab.com" in line:
                    in_gitlab = True
                if in_gitlab and line.startswith("password"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1], "~/.netrc file"
                if in_gitlab and line.startswith("machine") and "gitlab.com" not in line:
                    in_gitlab = False
        except Exception:
            pass

    # 5. Try python-gitlab config file
    gitlab_config_paths = [
        Path.home() / ".python-gitlab.cfg",
        Path("/etc/python-gitlab.cfg"),
    ]
    for config_path in gitlab_config_paths:
        if config_path.exists():
            try:
                import configparser
                config = configparser.ConfigParser()
                config.read(config_path)
                # Try [gitlab.com] section first, then [global]
                for section in ["gitlab.com", "global"]:
                    if section in config and "private_token" in config[section]:
                        return config[section]["private_token"], f"{config_path}"
            except Exception:
                pass

    return None, "not found"


class GitHubConfig(BaseModel):
    """GitHub configuration."""

    token: str | None = Field(default=None, repr=False)
    auto_discover_token: bool = True


class GitLabConfig(BaseModel):
    """GitLab configuration."""

    token: str | None = Field(default=None, repr=False)
    url: str = "https://gitlab.com"  # Can be changed for self-hosted instances
    auto_discover_token: bool = True