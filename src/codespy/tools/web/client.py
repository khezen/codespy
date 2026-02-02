"""Web browser client for fetching pages and searching."""

import logging
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from codespy.tools.web.models import SearchResult, SearchResults, WebPage

logger = logging.getLogger(__name__)

# Default headers to mimic a real browser
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Elements to remove from HTML before conversion
REMOVE_ELEMENTS = [
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    "noscript",
    "iframe",
    "svg",
    "form",
    "button",
    "input",
    "select",
    "textarea",
]


class WebBrowser:
    """Client for web browsing operations.

    Provides methods to fetch web pages and convert them to markdown,
    and to perform web searches.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_content_length: int = 500_000,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the web browser client.

        Args:
            timeout: Request timeout in seconds
            max_content_length: Maximum content length to process (bytes)
            headers: Custom headers to use for requests
        """
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.headers = headers or DEFAULT_HEADERS

    def _clean_html(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove unwanted elements from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            Cleaned BeautifulSoup object
        """
        # Remove unwanted elements
        for element in soup.find_all(REMOVE_ELEMENTS):
            element.decompose()

        # Remove elements with common ad/tracking classes
        for element in soup.find_all(class_=re.compile(r"(ad|ads|advert|banner|sidebar|popup)")):
            element.decompose()

        # Remove hidden elements
        for element in soup.find_all(style=re.compile(r"display:\s*none")):
            element.decompose()

        return soup

    def _html_to_markdown(self, html: str, base_url: str) -> str:
        """Convert HTML to clean markdown.

        Args:
            html: HTML content
            base_url: Base URL for resolving relative links

        Returns:
            Markdown content
        """
        soup = BeautifulSoup(html, "html.parser")
        soup = self._clean_html(soup)

        # Try to find main content area
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"(content|main|article)"))
            or soup.find(class_=re.compile(r"(content|main|article)"))
            or soup.find("body")
            or soup
        )

        # Convert to markdown with link preservation
        markdown = md(
            str(main_content),
            heading_style="ATX",  # Use # style headings
            bullets="-",  # Use - for lists
            strip=["img", "video", "audio", "picture", "source", "canvas", "map", "area"],
        )

        # Clean up excessive whitespace
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        markdown = markdown.strip()

        return markdown

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            Page title
        """
        # Try various title sources
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # Try og:title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return str(og_title["content"]).strip()

        # Try h1
        h1 = soup.find("h1")
        if h1 and h1.get_text():
            return h1.get_text().strip()

        return ""

    def fetch_page(self, url: str) -> WebPage:
        """Fetch a web page and convert it to markdown.

        Args:
            url: URL to fetch

        Returns:
            WebPage with markdown content
        """
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme:
            url = f"https://{url}"
            parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return WebPage(
                url=url,
                content="",
                status_code=0,
                error=f"Invalid URL scheme: {parsed.scheme}",
            )

        try:
            with httpx.Client(
                timeout=self.timeout,
                headers=self.headers,
                follow_redirects=True,
            ) as client:
                response = client.get(url)

                content_type = response.headers.get("content-type", "")

                # Check if content is HTML
                if "text/html" not in content_type.lower():
                    return WebPage(
                        url=str(response.url),
                        content=response.text[:self.max_content_length],
                        status_code=response.status_code,
                        content_type=content_type,
                    )

                html = response.text[:self.max_content_length]
                soup = BeautifulSoup(html, "html.parser")
                title = self._extract_title(soup)
                markdown = self._html_to_markdown(html, str(response.url))

                return WebPage(
                    url=str(response.url),
                    title=title,
                    content=markdown,
                    status_code=response.status_code,
                    content_type=content_type,
                )

        except httpx.TimeoutException:
            return WebPage(
                url=url,
                content="",
                status_code=0,
                error=f"Request timed out after {self.timeout}s",
            )
        except httpx.RequestError as e:
            return WebPage(
                url=url,
                content="",
                status_code=0,
                error=f"Request failed: {e}",
            )
        except Exception as e:
            logger.exception(f"Error fetching {url}")
            return WebPage(
                url=url,
                content="",
                status_code=0,
                error=f"Unexpected error: {e}",
            )

    def search(self, query: str, num_results: int = 10) -> SearchResults:
        """Perform a web search using DuckDuckGo.

        Args:
            query: Search query
            num_results: Maximum number of results to return

        Returns:
            SearchResults with list of results
        """
        try:
            from ddgs import DDGS
        except ImportError:
            return SearchResults(
                query=query,
                error="ddgs package not installed. Run: pip install ddgs",
            )

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))

            search_results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("link", "")),
                    snippet=r.get("body", r.get("snippet", "")),
                )
                for r in results
            ]

            return SearchResults(
                query=query,
                results=search_results,
            )

        except Exception as e:
            logger.exception(f"Search error for query: {query}")
            return SearchResults(
                query=query,
                error=f"Search failed: {e}",
            )

    def search_and_fetch(self, query: str, num_results: int = 3) -> list[WebPage]:
        """Search and fetch the top results.

        Args:
            query: Search query
            num_results: Number of results to fetch

        Returns:
            List of WebPage objects for each search result
        """
        search_results = self.search(query, num_results=num_results)

        if not search_results.success:
            return []

        pages = []
        for result in search_results.results:
            page = self.fetch_page(result.url)
            pages.append(page)

        return pages