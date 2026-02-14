"""CLI commands for remote PR/MR review (GitHub and GitLab)."""

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from codespy.config import get_settings
from codespy.config_git import get_github_token_source, get_gitlab_token_source
from codespy.tools.git.client import detect_platform, is_supported_url

console = Console()


def review(
    mr_url: Annotated[
        str,
        typer.Argument(
            help="Merge request URL (GitHub PR or GitLab MR)",
        ),
    ],
    config_file: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-f",
            help="Path to a YAML config file (overrides default config locations).",
        ),
    ] = None,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Output format: markdown or json",
        ),
    ] = "markdown",
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="LLM model to use (overrides DEFAULT_MODEL env var)",
        ),
    ] = None,
    stdout: Annotated[
        bool | None,
        typer.Option(
            "--stdout/--no-stdout",
            help="Enable/disable stdout output (overrides config)",
        ),
    ] = None,
    git_comment: Annotated[
        bool | None,
        typer.Option(
            "--git-comment/--no-git-comment",
            help="Enable/disable Git platform review comments (overrides config)",
        ),
    ] = None,
) -> None:
    """Review a GitHub PR or GitLab MR for security, bugs, and documentation.

    Examples:
        codespy review https://github.com/owner/repo/pull/123
        codespy review https://github.com/owner/repo/pull/123 --config path/to/config.yaml
        codespy review https://gitlab.com/namespace/project/-/merge_requests/123
        codespy review https://github.com/owner/repo/pull/123 --output json
        codespy review https://github.com/owner/repo/pull/123 --model bedrock/anthropic.claude-3-sonnet
        codespy review https://github.com/owner/repo/pull/123 --git-comment
        codespy review https://github.com/owner/repo/pull/123 --no-stdout --git-comment
    """
    # Set up logging with timestamps
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    )

    try:
        settings = get_settings(config_file=config_file)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Print config at startup (secrets are hidden via repr=False)
    logging.info(f"Loaded config: {settings}")
    
    # Log module configurations
    settings.log_signature_configs()

    # Override settings if provided via CLI (CLI > env > yaml > defaults)
    if model:
        settings.default_model = model
    if output:
        settings.output_format = output  # type: ignore
    if stdout is not None:
        settings.output_stdout = stdout
    if git_comment is not None:
        settings.output_git = git_comment

    # Validate URL format
    if not is_supported_url(mr_url):
        console.print(
            "[red]Error:[/red] Unsupported URL format.\n\n"
            "Supported formats:\n"
            "  • GitHub: https://github.com/owner/repo/pull/123\n"
            "  • GitLab: https://gitlab.com/namespace/project/-/merge_requests/123"
        )
        raise typer.Exit(1)

    # Detect platform and validate token
    platform = detect_platform(mr_url)
    
    if platform == "github":
        token = settings.github_token
        token_source = get_github_token_source()
        token_env_hint = "GITHUB_TOKEN / GH_TOKEN"
        cli_hint = "gh auth login"
    else:  # gitlab
        token = settings.gitlab_token
        token_source = get_gitlab_token_source()
        token_env_hint = "GITLAB_TOKEN / GITLAB_PRIVATE_TOKEN"
        cli_hint = "glab auth login"

    if not token:
        console.print(
            f"[red]Error:[/red] {platform.title()} token not found. Tried:\n"
            f"  • {token_env_hint} environment variables\n"
            f"  • {platform.title()} CLI ({cli_hint.split()[0]} auth token)\n"
            "  • Git credential helper\n"
            "  • ~/.netrc file\n\n"
            f"Please authenticate with {platform.title()} CLI: [bold]{cli_hint}[/bold]"
        )
        raise typer.Exit(1)

    # Build output destinations display
    output_destinations = []
    if settings.output_stdout:
        output_destinations.append(f"stdout ({settings.output_format})")
    if settings.output_git:
        output_destinations.append(f"{platform.title()} comment")
    output_display = ", ".join(output_destinations) if output_destinations else "[red]none[/red]"

    console.print(
        Panel(
            f"[bold blue]Reviewing MR:[/bold blue] {mr_url}\n"
            f"[bold]Platform:[/bold] {platform.title()}\n"
            f"[bold]Model:[/bold] {settings.default_model}\n"
            f"[bold]Output:[/bold] {output_display}\n"
            f"[bold]{platform.title()} Token:[/bold] [green]found[/green] [dim]({token_source})[/dim]",
            title="codespy",
        )
    )

    try:
        from codespy.agents.reviewer.reviewer import ReviewPipeline
        from codespy.agents.reviewer.models import RemoteReviewConfig

        pipeline = ReviewPipeline(settings)

        # Create remote review config
        config = RemoteReviewConfig(url=mr_url)
        
        # Run review (model access always verified in pipeline)
        result = pipeline(config)

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

        # Output results using reporters
        from codespy.agents.reviewer.reporters import GitReporter, StdoutReporter

        if settings.output_stdout:
            stdout_reporter = StdoutReporter(format=settings.output_format, console=console)
            stdout_reporter.report(result)

        if settings.output_git:
            console.print(f"[dim]Posting review to {platform.title()}...[/dim]")
            git_reporter = GitReporter(url=mr_url, settings=settings)
            git_reporter.report(result)
            console.print(f"[green]✓[/green] {platform.title()} review posted successfully")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error during review:[/red] {e}")
        logging.exception("Review failed")
        raise typer.Exit(1)
