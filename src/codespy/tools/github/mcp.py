"""MCP server for GitHub operations."""

import os
import sys

from mcp.server.fastmcp import FastMCP

from codespy.config import Settings
from codespy.tools.github.client import GitHubClient

mcp = FastMCP("github-server")
_client: GitHubClient | None = None


def _get_client() -> GitHubClient:
    """Get the GitHubClient instance, raising if not initialized."""
    if _client is None:
        raise RuntimeError("GitHubClient not initialized")
    return _client


@mcp.tool()
def parse_pr_url(url: str) -> dict:
    """Parse a GitHub PR URL into owner, repo, and PR number.

    Args:
        url: GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)

    Returns:
        Dict with owner, repo, and pr_number
    """
    owner, repo, pr_number = _get_client().parse_pr_url(url)
    return {"owner": owner, "repo": repo, "pr_number": pr_number}


@mcp.tool()
def fetch_pull_request(pr_url: str) -> dict:
    """Fetch pull request data from GitHub.

    Args:
        pr_url: GitHub PR URL

    Returns:
        Dict with PR data including title, body, changed files, etc.
    """
    pr = _get_client().fetch_pull_request(pr_url)
    return pr.model_dump()


@mcp.tool()
def clone_repository(owner: str, repo_name: str, ref: str) -> str:
    """Clone or update a repository and checkout a specific ref.

    Args:
        owner: Repository owner
        repo_name: Repository name
        ref: Git ref (branch, tag, or commit) to checkout

    Returns:
        Path to the cloned repository
    """
    path = _get_client().clone_repository(owner, repo_name, ref)
    return str(path)


if __name__ == "__main__":
    # Initialize with settings from environment
    settings = Settings()
    _client = GitHubClient(settings)
    mcp.run()