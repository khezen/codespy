"""Models for web browsing tool."""

from pydantic import BaseModel, Field


class WebPage(BaseModel):
    """Represents a fetched web page with markdown content."""

    url: str = Field(description="The URL of the page")
    title: str = Field(default="", description="Page title extracted from HTML")
    content: str = Field(description="Page content converted to markdown")
    status_code: int = Field(description="HTTP status code")
    content_type: str = Field(default="text/html", description="Content-Type header")
    error: str | None = Field(default=None, description="Error message if fetch failed")

    @property
    def success(self) -> bool:
        """Check if the page was fetched successfully."""
        return self.error is None and 200 <= self.status_code < 400


class SearchResult(BaseModel):
    """A single search result."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    snippet: str = Field(default="", description="Snippet/description of the result")


class SearchResults(BaseModel):
    """Search results from a web search."""

    query: str = Field(description="The search query")
    results: list[SearchResult] = Field(default_factory=list, description="List of results")
    error: str | None = Field(default=None, description="Error message if search failed")

    @property
    def success(self) -> bool:
        """Check if the search was successful."""
        return self.error is None

    def to_markdown(self) -> str:
        """Convert search results to markdown format."""
        if self.error:
            return f"**Search Error:** {self.error}"

        if not self.results:
            return f"No results found for: {self.query}"

        lines = [f"## Search Results for: {self.query}\n"]
        for i, result in enumerate(self.results, 1):
            lines.append(f"### {i}. [{result.title}]({result.url})")
            if result.snippet:
                lines.append(f"{result.snippet}\n")
            else:
                lines.append("")

        return "\n".join(lines)