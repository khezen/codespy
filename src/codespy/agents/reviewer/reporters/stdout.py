"""Stdout reporter for console output."""

import json
from typing import Literal

from rich.console import Console

from codespy.agents.reviewer.models import ReviewResult
from codespy.agents.reviewer.reporters.base import BaseReporter


class StdoutReporter(BaseReporter):
    """Reporter that outputs review results to stdout."""

    def __init__(
        self,
        format: Literal["markdown", "json"] = "markdown",
        console: Console | None = None,
    ) -> None:
        """Initialize stdout reporter.

        Args:
            format: Output format (markdown or json).
            console: Rich console for output. Creates new one if not provided.
        """
        self.format = format
        self.console = console or Console()

    def report(self, result: ReviewResult) -> None:
        """Output review result to console.

        Args:
            result: The review result to output.
        """
        if self.format == "json":
            self.console.print_json(json.dumps(result.to_json_dict(), indent=2))
        else:
            self.console.print(result.to_markdown())