"""Command-line interface for codespy."""

import json
import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from codespy import __version__
from codespy.config import get_settings, get_token_source

app = typer.Typer(
    name="codespy",
    help="Code review agent powered by DSPy",
    no_args_is_help=True,
)

console = Console()


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
def review(
    pr_url: Annotated[
        str,
        typer.Argument(
            help="GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)",
        ),
    ],
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Output format: markdown or json",
        ),
    ] = "markdown",
    with_context: Annotated[
        bool,
        typer.Option(
            "--with-context",
            "-c",
            help="Include repository context (imports, dependencies)",
        ),
    ] = True,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="LLM model to use (overrides LITELLM_MODEL env var)",
        ),
    ] = None,
    include_vendor: Annotated[
        bool,
        typer.Option(
            "--include-vendor",
            help="Include vendor/dependency files in review (node_modules, vendor/, etc.)",
        ),
    ] = False,
) -> None:
    """Review a GitHub pull request for security, bugs, and documentation.

    Examples:
        codespy review https://github.com/owner/repo/pull/123
        codespy review https://github.com/owner/repo/pull/123 --output json
        codespy review https://github.com/owner/repo/pull/123 --model bedrock/anthropic.claude-3-sonnet
    """
    settings = get_settings()

    # Override settings if provided via CLI
    if model:
        settings.litellm_model = model
    if output:
        settings.output_format = output  # type: ignore
    settings.include_repo_context = with_context
    settings.include_vendor = include_vendor

    # Set up logging with timestamps
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    )

    # Validate GitHub token
    token_source = get_token_source()
    if not settings.github_token:
        console.print(
            "[red]Error:[/red] GitHub token not found. Tried:\n"
            "  • GITHUB_TOKEN / GH_TOKEN environment variables\n"
            "  • GitHub CLI (gh auth token)\n"
            "  • Git credential helper\n"
            "  • ~/.netrc file\n\n"
            "Please authenticate with GitHub CLI: [bold]gh auth login[/bold]"
        )
        raise typer.Exit(1)

    vendor_status = "[yellow]included[/yellow]" if include_vendor else "[green]excluded[/green]"
    console.print(
        Panel(
            f"[bold blue]Reviewing PR:[/bold blue] {pr_url}\n"
            f"[bold]Model:[/bold] {settings.litellm_model}\n"
            f"[bold]Output:[/bold] {settings.output_format}\n"
            f"[bold]Context:[/bold] {'enabled' if with_context else 'disabled'}\n"
            f"[bold]Vendor files:[/bold] {vendor_status}\n"
            f"[bold]GitHub Token:[/bold] [green]found[/green] [dim]({token_source})[/dim]",
            title="codespy",
        )
    )

    try:
        from codespy.agents.reviewer.pipeline import ReviewPipeline
        from codespy.agents import verify_model_access

        pipeline = ReviewPipeline(settings)

        # Verify model access
        console.print("[dim]Verifying model access...[/dim]")
        success, message = verify_model_access(settings)
        if success:
            console.print(f"[bold]Model:[/bold] [green]verified[/green] [dim]({settings.litellm_model})[/dim]")
        else:
            console.print(f"[red]Error:[/red] {message}")
            raise typer.Exit(1)

        result = pipeline.forward(pr_url, verify_model=False)  # Already verified

        # Show cost summary
        if result.llm_calls > 0:
            cost_str = f"${result.total_cost:.4f}" if result.total_cost > 0 else "N/A"
            console.print(
                Panel(
                    f"[bold]LLM Calls:[/bold] {result.llm_calls}\n"
                    f"[bold]Total Tokens:[/bold] {result.total_tokens:,}\n"
                    f"[bold]Total Cost:[/bold] {cost_str}",
                    title="Cost Summary",
                )
            )

        # Output results
        if output == "json":
            console.print_json(json.dumps(result.to_json_dict(), indent=2))
        else:
            console.print(result.to_markdown())

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error during review:[/red] {e}")
        logging.exception("Review failed")
        raise typer.Exit(1)


@app.command()
def config() -> None:
    """Show current configuration."""
    settings = get_settings()

    console.print(Panel("[bold]Current Configuration[/bold]", title="codespy"))

    # Show non-sensitive settings
    console.print(f"[bold]Model:[/bold] {settings.litellm_model}")
    console.print(f"[bold]AWS Region:[/bold] {settings.aws_region}")
    console.print(f"[bold]Max Context Size:[/bold] {settings.max_context_size}")
    console.print(f"[bold]Output Format:[/bold] {settings.output_format}")
    console.print(f"[bold]Cache Directory:[/bold] {settings.cache_dir}")

    # Show token status (not the actual token)
    token_source = get_token_source()
    if settings.github_token:
        console.print(f"[bold]GitHub Token:[/bold] [green]configured[/green] [dim]({token_source})[/dim]")
    else:
        console.print("[bold]GitHub Token:[/bold] [red]not found[/red]")
    console.print(
        f"[bold]OpenAI API Key:[/bold] {'[green]configured[/green]' if settings.openai_api_key else '[dim]not set[/dim]'}"
    )
    console.print(
        f"[bold]Anthropic API Key:[/bold] {'[green]configured[/green]' if settings.anthropic_api_key else '[dim]not set[/dim]'}"
    )


if __name__ == "__main__":
    app()
