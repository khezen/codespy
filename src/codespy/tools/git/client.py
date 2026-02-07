"""Git client factory for automatic platform detection."""

import logging
from typing import TYPE_CHECKING

from codespy.tools.git.base import GitClient
from codespy.tools.git.github_client import GitHubClient
from codespy.tools.git.gitlab_client import GitLabClient

if TYPE_CHECKING:
    from codespy.config import Settings

logger = logging.getLogger(__name__)

# Registry of available clients in priority order
_CLIENT_CLASSES: list[type[GitClient]] = [
    GitHubClient,
    GitLabClient,
]


def get_client(url: str, settings: "Settings | None" = None) -> GitClient:
    """Get appropriate Git client based on URL.

    Automatically detects whether the URL is for GitHub or GitLab
    and returns the appropriate client instance.

    Args:
        url: Merge request URL (GitHub PR or GitLab MR)
        settings: Application settings. Uses global settings if not provided.

    Returns:
        GitClient instance (GitHubClient or GitLabClient)

    Raises:
        ValueError: If URL doesn't match any supported platform
    """
    for client_class in _CLIENT_CLASSES:
        if client_class.can_handle(url):
            client = client_class(settings)
            logger.debug(f"Using {client.platform_name} client for URL: {url}")
            return client

    raise ValueError(
        f"Unsupported Git platform URL: {url}\n"
        "Supported formats:\n"
        "  - GitHub: https://github.com/owner/repo/pull/123\n"
        "  - GitLab: https://gitlab.com/namespace/project/-/merge_requests/123"
    )


def detect_platform(url: str) -> str:
    """Detect the Git platform from a URL.

    Args:
        url: Merge request URL

    Returns:
        Platform name ('github' or 'gitlab')

    Raises:
        ValueError: If URL doesn't match any supported platform
    """
    for client_class in _CLIENT_CLASSES:
        if client_class.can_handle(url):
            return client_class(None).platform_name.lower()

    raise ValueError(f"Unsupported Git platform URL: {url}")


def is_supported_url(url: str) -> bool:
    """Check if a URL is supported by any Git client.

    Args:
        url: URL to check

    Returns:
        True if URL is supported, False otherwise
    """
    return any(client_class.can_handle(url) for client_class in _CLIENT_CLASSES)