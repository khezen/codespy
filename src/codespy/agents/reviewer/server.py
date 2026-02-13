"""MCP server for codespy code review — review local changes or remote PRs from your editor."""

import asyncio
import concurrent.futures
import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from codespy.config import Settings

logger = logging.getLogger(__name__)

mcp = FastMCP("codespy-reviewer")
_settings: Settings | None = None
_pipeline: Any = None

# Thread pool for running the review pipeline (which uses asyncio.run() internally)
# without conflicting with the MCP server's own event loop.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _get_pipeline() -> Any:
    """Get or create the ReviewPipeline (lazy init to avoid slow startup)."""
    global _pipeline
    if _pipeline is None:
        from codespy.agents.reviewer.reviewer import ReviewPipeline

        _pipeline = ReviewPipeline(_settings)
    return _pipeline


def _run_in_thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a function in the thread pool and return the result.

    The ReviewPipeline uses asyncio.run() internally, which can't be called
    from within the MCP server's event loop. Running in a separate thread
    gives the pipeline its own event loop.
    """
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


def _do_local_review(repo_path: str, base_ref: str, output_format: str) -> str:
    """Synchronous local review — runs in thread pool."""
    import json

    from codespy.tools.git.local_diff import build_mr_from_diff

    repo = Path(repo_path).resolve()
    mr = build_mr_from_diff(repo, base_ref=base_ref, include_uncommitted=False)

    if not mr.changed_files:
        return f"No changes found between {base_ref} and HEAD in {repo_path}"

    pipeline = _get_pipeline()
    result = pipeline(mr=mr, repo_path=repo, verify_model=True)

    if output_format == "json":
        return json.dumps(result.to_json_dict(), indent=2)
    return result.to_markdown()


def _do_uncommitted_review(repo_path: str, output_format: str) -> str:
    """Synchronous uncommitted review — runs in thread pool."""
    import json

    from codespy.tools.git.local_diff import build_mr_from_diff

    repo = Path(repo_path).resolve()
    mr = build_mr_from_diff(repo, include_uncommitted=True)

    if not mr.changed_files:
        return f"No uncommitted changes found in {repo_path}"

    pipeline = _get_pipeline()
    result = pipeline(mr=mr, repo_path=repo, verify_model=True)

    if output_format == "json":
        return json.dumps(result.to_json_dict(), indent=2)
    return result.to_markdown()


def _do_pr_review(mr_url: str, output_format: str) -> str:
    """Synchronous PR review — runs in thread pool."""
    import json

    pipeline = _get_pipeline()
    result = pipeline(mr_url=mr_url, verify_model=True)

    if output_format == "json":
        return json.dumps(result.to_json_dict(), indent=2)
    return result.to_markdown()


@mcp.tool()
async def review_local_changes(
    repo_path: str,
    base_ref: str = "main",
    output_format: str = "markdown",
) -> str:
    """Review local git changes (current branch vs base) for security, bugs, and documentation issues.

    No PR or remote platform required — works with any local git repository.
    Diffs the current HEAD against the base_ref to find changed files, then runs
    the full codespy review pipeline (scope identification, code & doc review,
    supply chain audit, deduplication, and summarization).

    Args:
        repo_path: Absolute path to the local git repository to review
        base_ref: Git ref to diff against (e.g., "main", "develop", "origin/main", "HEAD~5").
                  Defaults to "main".
        output_format: Output format — "markdown" for human-readable or "json" for structured data.
                       Defaults to "markdown".

    Returns:
        Review results as markdown or JSON string
    """
    try:
        return await _run_in_thread(_do_local_review, repo_path, base_ref, output_format)
    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.exception("Review failed")
        return f"Review failed: {e}"


@mcp.tool()
async def review_uncommitted(
    repo_path: str,
    output_format: str = "markdown",
) -> str:
    """Review uncommitted changes (staged + unstaged) in a local git repository.

    Reviews all modifications in the working tree that haven't been committed yet.
    Useful for checking your work before committing.

    Args:
        repo_path: Absolute path to the local git repository to review
        output_format: Output format — "markdown" for human-readable or "json" for structured data.
                       Defaults to "markdown".

    Returns:
        Review results as markdown or JSON string
    """
    try:
        return await _run_in_thread(_do_uncommitted_review, repo_path, output_format)
    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.exception("Review failed")
        return f"Review failed: {e}"


@mcp.tool()
async def review_pr(
    mr_url: str,
    output_format: str = "markdown",
) -> str:
    """Review a GitHub Pull Request or GitLab Merge Request by URL.

    Fetches the PR/MR data from the platform, clones the repository, and runs
    the full codespy review pipeline.

    Args:
        mr_url: Full URL of the PR/MR to review.
                GitHub: https://github.com/owner/repo/pull/123
                GitLab: https://gitlab.com/namespace/project/-/merge_requests/123
        output_format: Output format — "markdown" for human-readable or "json" for structured data.
                       Defaults to "markdown".

    Returns:
        Review results as markdown or JSON string
    """
    try:
        return await _run_in_thread(_do_pr_review, mr_url, output_format)
    except Exception as e:
        logger.exception("Review failed")
        return f"Review failed: {e}"


def run_server() -> None:
    """Start the MCP server (called from CLI or __main__)."""
    global _settings

    # Suppress noisy MCP server logs — keep the transport clean
    logging.getLogger("mcp.server").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)

    # Send application logs to stderr (stdout is the MCP transport)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    _settings = Settings()
    logger.info(f"codespy-reviewer MCP server starting (model: {_settings.default_model})")
    mcp.run()


if __name__ == "__main__":
    run_server()
