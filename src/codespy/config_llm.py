"""LLM provider configuration and auto-discovery."""

import logging
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def discover_aws_credentials() -> tuple[str | None, str | None, str | None, str | None, str]:
    """Try to discover AWS credentials from local environment.

    Returns:
        Tuple of (access_key_id, secret_access_key, region, profile, source)
    """
    import configparser

    access_key_id = None
    secret_access_key = None
    region = None
    profile = None
    source = "not found"

    # 1. Check environment variables
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        profile = os.environ.get("AWS_PROFILE")
        return access_key_id, secret_access_key, region, profile, "environment variables"

    # 2. Try AWS credentials file
    credentials_path = Path.home() / ".aws" / "credentials"
    config_path = Path.home() / ".aws" / "config"
    profile_name = os.environ.get("AWS_PROFILE", "default")

    if credentials_path.exists():
        try:
            config = configparser.ConfigParser()
            config.read(credentials_path)

            if profile_name in config:
                access_key_id = config[profile_name].get("aws_access_key_id")
                secret_access_key = config[profile_name].get("aws_secret_access_key")
                profile = profile_name if profile_name != "default" else None

                # Also check config file for region
                if config_path.exists():
                    config_file = configparser.ConfigParser()
                    config_file.read(config_path)
                    # Config file uses "profile X" format for non-default profiles
                    config_section = f"profile {profile_name}" if profile_name != "default" else "default"
                    if config_section in config_file:
                        region = config_file[config_section].get("region")

                if access_key_id and secret_access_key:
                    source = f"~/.aws/credentials [{profile_name}]"
                    return access_key_id, secret_access_key, region, profile, source
        except Exception:
            pass

    # 3. Try AWS CLI
    try:
        result = subprocess.run(
            ["aws", "configure", "get", "aws_access_key_id"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            access_key_id = result.stdout.strip()

            result = subprocess.run(
                ["aws", "configure", "get", "aws_secret_access_key"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                secret_access_key = result.stdout.strip()

                # Try to get region
                result = subprocess.run(
                    ["aws", "configure", "get", "region"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    region = result.stdout.strip()

                return access_key_id, secret_access_key, region, profile, "AWS CLI (aws configure)"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None, None, None, None, "not found"


def discover_openai_api_key() -> tuple[str | None, str]:
    """Try to discover OpenAI API key from local environment.

    Returns:
        Tuple of (api_key, source)
    """
    # 1. Check environment variable
    if key := os.environ.get("OPENAI_API_KEY"):
        return key, "environment variable $OPENAI_API_KEY"

    # 2. Try config files
    config_paths = [
        Path.home() / ".config" / "openai" / "config",
        Path.home() / ".openai" / "config",
        Path.home() / ".openai" / "api_key",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                content = config_path.read_text().strip()
                # Check if it's a simple key file or INI-style config
                if content.startswith("sk-"):
                    return content, f"{config_path}"
                # Try parsing as key=value or INI
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("api_key") or line.startswith("OPENAI_API_KEY"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            key = parts[1].strip().strip("'\"")
                            if key:
                                return key, f"{config_path}"
            except Exception:
                pass

    return None, "not found"


def discover_anthropic_api_key() -> tuple[str | None, str]:
    """Try to discover Anthropic API key from local environment.

    Returns:
        Tuple of (api_key, source)
    """
    # 1. Check environment variable
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return key, "environment variable $ANTHROPIC_API_KEY"

    # 2. Try config files
    config_paths = [
        Path.home() / ".config" / "anthropic" / "config",
        Path.home() / ".anthropic" / "config",
        Path.home() / ".anthropic" / "api_key",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                content = config_path.read_text().strip()
                # Check if it's a simple key file
                if content.startswith("sk-ant-"):
                    return content, f"{config_path}"
                # Try parsing as key=value or INI
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("api_key") or line.startswith("ANTHROPIC_API_KEY"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            key = parts[1].strip().strip("'\"")
                            if key:
                                return key, f"{config_path}"
            except Exception:
                pass

    return None, "not found"


def discover_gemini_api_key() -> tuple[str | None, str]:
    """Try to discover Google Gemini API key from local environment.

    Returns:
        Tuple of (api_key, source)
    """
    # 1. Check environment variables
    for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if key := os.environ.get(env_var):
            return key, f"environment variable ${env_var}"

    # 2. Try config files
    config_paths = [
        Path.home() / ".config" / "gemini" / "config",
        Path.home() / ".config" / "google" / "api_key",
        Path.home() / ".gemini" / "api_key",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                content = config_path.read_text().strip()
                # Check if it's a simple key file
                if content and not content.startswith("#"):
                    # Single line = api key
                    first_line = content.split("\n")[0].strip()
                    if first_line and "=" not in first_line:
                        return first_line, f"{config_path}"
                # Try parsing as key=value
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("api_key") or line.startswith("GEMINI_API_KEY") or line.startswith("GOOGLE_API_KEY"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            key = parts[1].strip().strip("'\"")
                            if key:
                                return key, f"{config_path}"
            except Exception:
                pass

    # 3. Check for Google Application Default Credentials (for Vertex AI)
    adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if adc_path and Path(adc_path).exists():
        return None, f"Google ADC available at {adc_path} (use Vertex AI)"

    # 4. Try gcloud for application default credentials
    try:
        result = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            # ADC is available but we return None since it's OAuth, not API key
            return None, "Google ADC via gcloud (use Vertex AI)"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None, "not found"


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    # OpenAI
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_api_base: str | None = None
    auto_discover_openai: bool = True  # Set to False to disable auto-discovery

    # Anthropic
    anthropic_api_key: str | None = Field(default=None, repr=False)
    auto_discover_anthropic: bool = True  # Set to False to disable auto-discovery

    # Google Gemini
    gemini_api_key: str | None = Field(default=None, repr=False)
    auto_discover_gemini: bool = True  # Set to False to disable auto-discovery

    # AWS Bedrock
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = Field(default=None, repr=False)
    aws_secret_access_key: str | None = Field(default=None, repr=False)
    aws_profile: str | None = None
    auto_discover_aws: bool = True  # Set to False to disable auto-discovery

    # Azure OpenAI
    azure_api_key: str | None = Field(default=None, repr=False)
    azure_api_base: str | None = None
    azure_api_version: str | None = None

    # Enable provider-side prompt caching (Anthropic, OpenAI, Bedrock, etc.)
    enable_prompt_caching: bool = True

    def sync_from_flat(
        self,
        *,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        gemini_api_key: str | None = None,
        aws_region: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> dict[str, object]:
        """Sync flat settings into this LLMConfig and return values to propagate back.

        Priority: flat fields (from env vars via pydantic-settings) > nested defaults.
        Returns dict of field values that should be set on the parent Settings.
        """
        # Step 1: Flat → nested (env vars win over nested defaults)
        if openai_api_key:
            self.openai_api_key = openai_api_key
        if anthropic_api_key:
            self.anthropic_api_key = anthropic_api_key
        if gemini_api_key:
            self.gemini_api_key = gemini_api_key
        if aws_region:
            self.aws_region = aws_region
        if aws_access_key_id:
            self.aws_access_key_id = aws_access_key_id
        if aws_secret_access_key:
            self.aws_secret_access_key = aws_secret_access_key

        # Step 2: Return merged values (nested fills gaps where flat is not set)
        return {
            "openai_api_key": openai_api_key or self.openai_api_key,
            "anthropic_api_key": anthropic_api_key or self.anthropic_api_key,
            "gemini_api_key": gemini_api_key or self.gemini_api_key,
            "aws_access_key_id": aws_access_key_id or self.aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key or self.aws_secret_access_key,
            "enable_prompt_caching": self.enable_prompt_caching,
        }
