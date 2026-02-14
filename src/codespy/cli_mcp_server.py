"""MCP server command for codespy."""

from typing import Annotated

import typer
from rich.console import Console

from codespy.config import get_settings

console = Console()


def serve(
    config_file: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-f",
            help="Path to a YAML config file (overrides default config locations).",
        ),
    ] = None,
) -> None:
    """Start the codespy MCP server for IDE integration.

    Runs an MCP (Model Context Protocol) server over stdin/stdout that exposes
    code review tools. Configure your IDE (e.g., VS Code with Cline) to connect
    to this server, then review local changes or remote PRs without leaving
    your editor.

    Available tools:
        review_local_changes  — Review local git changes (branch vs base)
        review_uncommitted    — Review uncommitted working tree changes
        review_pr             — Review a GitHub PR or GitLab MR by URL

    Examples:
        codespy serve
        codespy serve --config path/to/config.yaml

    MCP config for Cline (cline_mcp_settings.json):
        {
          "mcpServers": {
            "codespy-reviewer": {
              "command": "codespy",
              "args": ["serve"]
            }
          }
        }
    """
    try:
        settings = get_settings(config_file=config_file)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    from codespy.agents.reviewer.server import run_server
    run_server(settings=settings)
