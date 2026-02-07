"""MCP server for Git operations (GitHub and GitLab)."""

import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

from codespy.config import Settings
from codespy.tools.git.client import get_client, detect_platform, is_supported_url
from codespy.tools.git.base import GitClient

logger = logging.getLogger(__name__)

# Get caller module from environment (set by mcp_utils.py)
_caller_module = os.environ.get("MCP_CALLER_MODULE", "unknown")

mcp = FastMCP("git")
_settings: Settings | None = None


def _get_client(url: str) -> GitClient:
    """Get the appropriate GitClient based on URL."""
    if _settings is None:
        raise RuntimeError("Settings not initialized")
    return get_client(url, _settings)


@mcp.tool()
def parse_mr_url(url: str) -> dict:
    """Parse a Git merge request URL into owner, repo, and MR number.

    Works with both GitHub Pull Requests and GitLab Merge Requests.

    Args:
        url: Git MR URL
             - GitHub: https://github.com/owner/repo/pull/123
             - GitLab: https://gitlab.com/group/project/-/merge_requests/123

    Returns:
        Dict with owner (or namespace), repo, mr_number, and platform
    """
    platform = detect_platform(url)
    client = _get_client(url)
    owner, repo, mr_number = client.parse_url(url)
    return {
        "owner": owner,
        "repo": repo,
        "mr_number": mr_number,
        "platform": platform.value,
    }


@mcp.tool()
def fetch_merge_request(mr_url: str) -> dict:
    """Fetch merge request data from GitHub or GitLab.

    Works with both GitHub Pull Requests and GitLab Merge Requests.

    Args:
        mr_url: Git MR URL
                - GitHub: https://github.com/owner/repo/pull/123
                - GitLab: https://gitlab.com/group/project/-/merge_requests/123

    Returns:
        Dict with MR data including title, body, changed files, etc.
    """
    client = _get_client(mr_url)
    mr = client.fetch_merge_request(mr_url)
    return mr.model_dump()


@mcp.tool()
def clone_repository(
    owner: str,
    repo_name: str,
    ref: str,
    target_path: str,
    depth: int | None = 1,
    sparse_paths: list[str] | None = None,
    platform: str = "github",
) -> str:
    """Clone or update a repository and checkout a specific ref.

    Use this tool to efficiently clone a repository before exploring it.
    For large repos, use sparse_paths to only checkout directories relevant to your task.

    Args:
        owner: Repository owner/namespace
               - GitHub: "facebook"
               - GitLab: "group/subgroup" (supports nested namespaces)
        repo_name: Repository name (e.g., "react")
        ref: Git ref (branch, tag, or commit SHA) to checkout
        target_path: Absolute path where the repository should be cloned
        depth: Shallow clone depth (1 = only latest commit, None = full history). Default 1 for efficiency.
        sparse_paths: List of directory paths for sparse checkout (e.g., ["src/", "lib/common/"]).
                      Use this for large monorepos to only fetch needed directories.
                      Derive sparse paths from changed file paths to minimize clone size.
        platform: Git platform - "github" or "gitlab" (default: "github")

    Returns:
        Path to the cloned repository
    """
    from pathlib import Path
    from codespy.tools.git.models import GitPlatform
    
    if _settings is None:
        raise RuntimeError("Settings not initialized")
    
    # Build a URL to get the right client
    if platform.lower() == "gitlab":
        # Use the configured GitLab URL or default
        gitlab_base = _settings.gitlab_url.rstrip("/") if _settings.gitlab_url else "https://gitlab.com"
        dummy_url = f"{gitlab_base}/{owner}/{repo_name}/-/merge_requests/1"
    else:
        dummy_url = f"https://github.com/{owner}/{repo_name}/pull/1"
    
    client = _get_client(dummy_url)
    logger.info(f"[GIT] {_caller_module} -> clone_repository: {owner}/{repo_name}@{ref[:8]} ({platform})")
    path = client.clone_repository(
        owner=owner,
        repo_name=repo_name,
        ref=ref,
        target_path=Path(target_path),
        depth=depth,
        sparse_paths=sparse_paths,
    )
    return str(path)


@mcp.tool()
def detect_git_platform(url: str) -> dict:
    """Detect which Git platform a URL belongs to.

    Args:
        url: Any URL that might be a Git platform URL

    Returns:
        Dict with platform name and whether the URL is supported
    """
    supported = is_supported_url(url)
    if supported:
        platform = detect_platform(url)
        return {
            "platform": platform.value,
            "supported": True,
        }
    return {
        "platform": "unknown",
        "supported": False,
    }


if __name__ == "__main__":
    # Suppress noisy MCP server "Processing request" logs
    logging.getLogger("mcp.server").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)
    
    # Initialize with settings from environment
    _settings = Settings()
    mcp.run()