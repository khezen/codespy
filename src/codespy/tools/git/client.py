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
        url: Pull request URL (GitHub PR or GitLab MR)
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
        url: Pull request URL

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


def parse_repo_url(url: str) -> tuple[str, str, str, str]:
    """Parse a repository URL into (platform, host, owner, repo).

    Accepts ``https://github.com/<owner>/<repo>[.git][/]`` and
    ``https://<gitlab host>/<group>[/<subgroup>...]/<project>[.git][/]``.
    PR/MR, tree and blob URLs are rejected: pass ``--path``/``--ref`` instead.

    Raises:
        ValueError: If the URL is not a plain repository URL.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        raise ValueError(f"Invalid repository URL (expected https://...): {url}")
    path = parsed.path.strip("/").removesuffix(".git")
    parts = [p for p in path.split("/") if p]
    bad = [p for p in parts if p in (".", "..") or p.startswith("-")]
    if bad:
        raise ValueError(f"Invalid repository URL: {url} (unsafe path segment {bad[0]!r})")

    if host in ("github.com", "www.github.com"):
        if len(parts) != 2:
            raise ValueError(
                f"Invalid GitHub repository URL: {url}. Expected https://github.com/owner/repo"
            )
        return "github", "github.com", parts[0], parts[1]

    if "-" in parts or len(parts) < 2:
        raise ValueError(
            f"Invalid GitLab repository URL: {url}. "
            "Expected https://gitlab.example.com/group[/subgroup]/project"
        )
    return "gitlab", host, "/".join(parts[:-1]), parts[-1]


def get_repo_client(url: str, settings: "Settings | None" = None) -> GitClient:
    """Get the Git client for a plain repository URL (see ``parse_repo_url``).

    GitLab repos must live on the configured instance (``gitlab_url``,
    default https://gitlab.com), since that is where the client sends API calls.

    Raises:
        ValueError: If the URL is invalid or its GitLab host is not the configured one.
    """
    from urllib.parse import urlparse

    platform, host, _, _ = parse_repo_url(url)
    if platform == "github":
        return GitHubClient(settings)

    client = GitLabClient(settings)
    configured = (urlparse(client.gitlab_url).hostname or "").lower()
    if host != configured:
        raise ValueError(
            f"Repository host {host} does not match the configured GitLab instance "
            f"{configured}. Set GITLAB_URL=https://{host}."
        )
    return client
