"""Untrusted PR/model text must not be parsed as Rich markup."""

import io
from unittest.mock import MagicMock

from rich.console import Console
from rich.errors import MarkupError
from rich.markup import escape

from codespy.workflows.review.reporters.stdout import StdoutReporter

ROUTE = "GET /accounts/[/{accountID}] returns 404"


def test_stdout_markdown_with_markup_like_text() -> None:
    buf = io.StringIO()
    result = MagicMock()
    result.to_markdown.return_value = f"## Findings\n- {ROUTE}"
    StdoutReporter(format="markdown", console=Console(file=buf, width=200)).report(result)
    assert ROUTE in buf.getvalue()


def test_error_message_with_markup_like_text() -> None:
    buf = io.StringIO()
    console = Console(file=buf, width=200)
    err = MarkupError(f"closing tag '[/{{accountID}}]' at position 45 doesn't match any open tag")
    console.print(f"[red]Error during review:[/red] {escape(str(err))}")
    assert "[/{accountID}]" in buf.getvalue()
