"""GitHub configuration and auto-discovery."""

import logging
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global variable to track token source
_token_source: str = "not set"


def get_token_source() -> str:
    """Get the source of the GitHub token."""
    return _token_source


def set_token_source(source: str) -> None:
    """Set the source of the GitHub token."""
    global _token_source
    _token_source = source


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


class GitHubConfig(BaseModel):
    """GitHub configuration."""

    token: str | None = Field(default=None, repr=False)
    auto_discover_token: bool = True  # Set to False to disable auto-discovery