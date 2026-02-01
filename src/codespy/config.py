"""Configuration management for codespy."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


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


# Global variable to track token source
_token_source: str = "not set"


def get_token_source() -> str:
    """Get the source of the GitHub token."""
    return _token_source


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # GitHub settings
    github_token: str = Field(
        default="",
        description="GitHub personal access token for API access",
    )
    gh_token: str = Field(
        default="",
        description="Alternative GitHub token (GH_TOKEN)",
    )

    @model_validator(mode="after")
    def resolve_github_token(self) -> "Settings":
        """Auto-discover GitHub token if not explicitly set."""
        global _token_source

        def is_placeholder(token: str) -> bool:
            """Check if token looks like a placeholder."""
            placeholders = ["xxx", "your", "token", "example", "placeholder"]
            token_lower = token.lower()
            return any(p in token_lower for p in placeholders)

        # If github_token is set and not a placeholder, use it
        if self.github_token and not is_placeholder(self.github_token):
            _token_source = "GITHUB_TOKEN environment variable or .env file"
            return self

        # If GH_TOKEN is set and not a placeholder, use it
        if self.gh_token and not is_placeholder(self.gh_token):
            self.github_token = self.gh_token
            _token_source = "GH_TOKEN environment variable"
            return self

        # Clear placeholder if present
        if self.github_token and is_placeholder(self.github_token):
            self.github_token = ""

        # Try auto-discovery
        token, source = discover_github_token()
        if token and not is_placeholder(token):
            self.github_token = token
            _token_source = source
            logger.debug(f"GitHub token discovered from: {source}")
        else:
            _token_source = "not found"

        return self

    # LLM settings (LiteLLM format)
    litellm_model: str = Field(
        default="gpt-4o",
        description="LiteLLM model identifier (e.g., gpt-4o, bedrock/anthropic.claude-3-sonnet)",
    )

    # AWS settings (for Bedrock)
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock",
    )
    aws_access_key_id: str | None = Field(
        default=None,
        description="AWS access key ID (optional, uses default credentials if not set)",
    )
    aws_secret_access_key: str | None = Field(
        default=None,
        description="AWS secret access key (optional, uses default credentials if not set)",
    )

    # OpenAI settings
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )

    # Anthropic settings
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key",
    )

    # Review settings
    max_context_size: int = Field(
        default=50000,
        description="Maximum context size in characters for LLM input",
    )
    include_repo_context: bool = Field(
        default=True,
        description="Whether to include repository context (imports, dependencies)",
    )

    # Output settings
    output_format: Literal["markdown", "json"] = Field(
        default="markdown",
        description="Default output format for review results",
    )

    # Cache settings
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "codespy",
        description="Directory for caching cloned repositories",
    )


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the current settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings (useful after environment changes)."""
    global settings
    settings = Settings()
    return settings
