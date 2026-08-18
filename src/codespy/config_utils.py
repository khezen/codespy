"""Lightweight config utilities. No internal imports to avoid circular deps."""

from __future__ import annotations

from pydantic import SecretStr


def secret_value(s: SecretStr | None) -> str:
    """Extract the plain-text value from a SecretStr, or return empty string.

    Use at API boundaries where a raw string is needed (library calls,
    environment variables, URL interpolation). The empty-string return for
    None/empty ensures callers can use standard truthiness checks:

        token = secret_value(settings.github_token)
        if token:
            # token is configured and non-empty
    """
    if s is None:
        return ""
    return s.get_secret_value()
