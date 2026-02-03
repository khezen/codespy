"""MCP server for GitHub operations."""

import os
import sys

from mcp.server.fastmcp import FastMCP

from codespy.config import Settings
from codespy.tools.github.client import GitHubClient

mcp = FastMCP("github")
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
def clone_repository(
    owner: str,
    repo_name: str,
    ref: str,
    target_path: str,
    depth: int | None = 1,
    sparse_paths: list[str] | None = None,
) -> str:
    """Clone or update a repository and checkout a specific ref.

    Use this tool to efficiently clone a repository before exploring it.
    For large repos, use sparse_paths to only checkout directories relevant to your task.

    Args:
        owner: Repository owner (e.g., "facebook")
        repo_name: Repository name (e.g., "react")
        ref: Git ref (branch, tag, or commit SHA) to checkout
        target_path: Absolute path where the repository should be cloned
        depth: Shallow clone depth (1 = only latest commit, None = full history). Default 1 for efficiency.
        sparse_paths: List of directory paths for sparse checkout (e.g., ["src/", "lib/common/"]).
                      Use this for large monorepos to only fetch needed directories.
                      Derive sparse paths from changed file paths to minimize clone size.

    Returns:
        Path to the cloned repository
    """
    from pathlib import Path
    path = _get_client().clone_repository(
        owner=owner,
        repo_name=repo_name,
        ref=ref,
        target_path=Path(target_path),
        depth=depth,
        sparse_paths=sparse_paths,
    )
    return str(path)


if __name__ == "__main__":
    # Initialize with settings from environment
    settings = Settings()
    _client = GitHubClient(settings)
    mcp.run()