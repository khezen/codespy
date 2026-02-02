"""MCP server for web browsing operations."""

from mcp.server.fastmcp import FastMCP

from codespy.tools.web.client import WebBrowser

mcp = FastMCP("web-server")
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
    pages = _get_browser().search_and_fetch(query, num_results)
    return [p.model_dump() for p in pages]


if __name__ == "__main__":
    _browser = WebBrowser()
    mcp.run()