"""Web browsing tool for fetching pages and searching."""

from codespy.tools.web.client import WebBrowser
from codespy.tools.web.models import SearchResult, SearchResults, WebPage

__all__ = [
    "WebBrowser",
    "WebPage",
    "SearchResult",
    "SearchResults",
]