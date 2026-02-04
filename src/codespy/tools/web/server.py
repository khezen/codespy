"""MCP server for web browsing operations."""

import logging
import os

from mcp.server.fastmcp import FastMCP

from codespy.tools.web.client import WebBrowser

logger = logging.getLogger(__name__)
_caller_module = os.environ.get("MCP_CALLER_MODULE", "unknown")

mcp = FastMCP("web")
_browser: WebBrowser | None = None


def _get_browser() -> WebBrowser:
    """Get the WebBrowser instance, raising if not initialized."""
    if _browser is None:
        raise RuntimeError("WebBrowser not initialized")
    return _browser


@mcp.tool()
def fetch_page(url: str) -> dict:
    """Fetch a web page and convert it to markdown.

    Args:
        url: URL to fetch

    Returns:
        Dict with url, title, content (markdown), status_code, error (if any)
    """
    logger.info(f"[WEB] {_caller_module} -> fetch_page: {url}")
    page = _get_browser().fetch_page(url)
    return page.model_dump()


@mcp.tool()
def search(query: str, num_results: int = 10) -> dict:
    """Perform a web search using DuckDuckGo.

    Args:
        query: Search query
        num_results: Maximum number of results to return

    Returns:
        Dict with query, results (list of {title, url, snippet}), error (if any)
    """
    logger.info(f"[WEB] {_caller_module} -> search: {query}")
    results = _get_browser().search(query, num_results)
    return results.model_dump()


@mcp.tool()
def search_and_fetch(query: str, num_results: int = 3) -> list[dict]:
    """Search and fetch the top results.

    Args:
        query: Search query
        num_results: Number of results to fetch

    Returns:
        List of WebPage dicts for each search result
    """
    logger.info(f"[WEB] {_caller_module} -> search_and_fetch: {query} (top {num_results})")
    pages = _get_browser().search_and_fetch(query, num_results)
    return [p.model_dump() for p in pages]


if __name__ == "__main__":
    # Suppress noisy MCP server "Processing request" logs
    logging.getLogger("mcp.server").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)
    
    _browser = WebBrowser()
    mcp.run()
