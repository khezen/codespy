"""Command-line interface for codespy."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from codespy import __version__
from codespy.config import get_settings
from codespy.config_git import get_github_token_source, get_gitlab_token_source

# Import command functions from submodules
from codespy.cli_remote import review
from codespy.cli_local import review_local, review_uncommitted
from codespy.cli_mcp_server import serve

app = typer.Typer(
    name="codespy",
    help="Code review agent powered by DSPy",
    no_args_is_help=True,
)

console = Console()

# Register commands from submodules
app.command(name="review")(review)
app.command(name="review-local")(review_local)
app.command(name="review-uncommitted")(review_uncommitted)
app.command(name="serve")(serve)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"codespy version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """codespy - Code review agent powered by DSPy."""
    pass


@app.command()
def config(
    config_file: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-f",
            help="Path to a YAML config file (overrides default config locations).",
        ),
    ] = None,
) -> None:
    """Show current configuration."""
    try:
        settings = get_settings(config_file=config_file)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(Panel("[bold]Current Configuration[/bold]", title="codespy"))

    # Show non-sensitive settings
    console.print(f"[bold]Model:[/bold] {settings.default_model}")
    console.print(f"[bold]AWS Region:[/bold] {settings.aws_region}")
    console.print(f"[bold]Max Context Size:[/bold] {settings.default_max_context_size}")
    console.print(f"[bold]Output Format:[/bold] {settings.output_format}")
    console.print(f"[bold]Cache Directory:[/bold] {settings.cache_dir}")

    # Show GitHub token status
    github_token_source = get_github_token_source()
    if settings.github_token:
        console.print(f"[bold]GitHub Token:[/bold] [green]configured[/green] [dim]({github_token_source})[/dim]")
    else:
        console.print("[bold]GitHub Token:[/bold] [dim]not found[/dim]")

    # Show GitLab token status
    gitlab_token_source = get_gitlab_token_source()
    if settings.gitlab_token:
        console.print(f"[bold]GitLab Token:[/bold] [green]configured[/green] [dim]({gitlab_token_source})[/dim]")
    else:
        console.print("[bold]GitLab Token:[/bold] [dim]not found[/dim]")

    console.print(f"[bold]GitLab URL:[/bold] {settings.gitlab_url}")

    console.print(
        f"[bold]OpenAI API Key:[/bold] {'[green]configured[/green]' if settings.openai_api_key else '[dim]not set[/dim]'}"
    )
    console.print(
        f"[bold]Anthropic API Key:[/bold] {'[green]configured[/green]' if settings.anthropic_api_key else '[dim]not set[/dim]'}"
    )


if __name__ == "__main__":
    app()
