"""CLI command for full-path review (``codespy review-path``)."""

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.panel import Panel

from codespy import __version__
from codespy.config import get_settings
from codespy.config_git import get_github_token_source, get_gitlab_token_source
from codespy.config_utils import secret_value
from codespy.tools.git.client import get_repo_client, parse_repo_url

console = Console()


def review_path(
    repo_url: Annotated[
        str,
        typer.Argument(
            help="Repository URL (https://github.com/owner/repo or a GitLab project URL)"
        ),
    ],
    path: Annotated[
        str,
        typer.Option(
            "--path", "-p", help="Path to review, relative to the repo root (default: whole repo)"
        ),
    ] = ".",
    ref: Annotated[
        str | None,
        typer.Option(
            "--ref",
            "-r",
            help="Branch to review; also the report PR base (default: repo default branch)",
        ),
    ] = None,
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
        typer.Option("--output", "-o", help="Output format: markdown or json"),
    ] = "markdown",
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LLM model to use (overrides DEFAULT_MODEL env var)"),
    ] = None,
    stdout: Annotated[
        bool | None,
        typer.Option(
            "--stdout/--no-stdout", help="Enable/disable stdout output (overrides config)"
        ),
    ] = None,
    open_pr: Annotated[
        bool | None,
        typer.Option(
            "--open-pr/--no-open-pr",
            help="Open/update the report PR (overrides review.output_git). "
            "--no-open-pr is a dry run.",
        ),
    ] = None,
    max_files: Annotated[
        int | None,
        typer.Option(
            "--max-files", min=1, help="Max reviewable files (overrides review.full_path.max_files)"
        ),
    ] = None,
) -> None:
    """Review every file under a repo path and deliver the report as a new PR/MR.

    There is no diff: every file under --path is reviewed as newly added code.
    codespy commits the report to .codespy/reviews/<path>.md on branch
    codespy/review/<path> and opens a PR (or updates the open one) with the
    report in its description.

    Examples:
        codespy review-path https://github.com/owner/repo
        codespy review-path https://github.com/owner/repo --path services/api
        codespy review-path https://gitlab.com/group/project --path src --ref develop
        codespy review-path https://github.com/owner/repo --path lib --no-open-pr
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    )

    try:
        settings = get_settings(config_file=config_file)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1) from None

    logging.info(f"Loaded config: {settings}")
    settings.log_signature_configs()

    if model:
        settings.llm.default_model = model
    if output:
        settings.review.output_format = output  # type: ignore
    if stdout is not None:
        settings.review.output_stdout = stdout
    if open_pr is not None:
        settings.review.output_git = open_pr

    try:
        platform, host, owner, repo_name = parse_repo_url(repo_url)
        get_repo_client(repo_url, settings)  # validates the GitLab host
    except ValueError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1) from None

    if platform == "github":
        token, token_source = secret_value(settings.github_token), get_github_token_source()
        hint = "Set GITHUB_TOKEN / GH_TOKEN or run [bold]gh auth login[/bold]"
        scopes = "contents: write + pull-requests: write"
    else:
        token, token_source = secret_value(settings.gitlab_token), get_gitlab_token_source()
        hint = "Set GITLAB_TOKEN / GITLAB_PRIVATE_TOKEN or run [bold]glab auth login[/bold]"
        scopes = "api"
    if not token:
        console.print(f"[red]Error:[/red] {platform.title()} token not found. {hint}.")
        raise typer.Exit(1)

    caps = settings.review.full_path
    output_destinations = []
    if settings.review.output_stdout:
        output_destinations.append(f"stdout ({settings.review.output_format})")
    if settings.review.output_git:
        output_destinations.append(f"{platform.title()} report PR (token needs {scopes})")
    console.print(
        Panel(
            f"[bold blue]Reviewing path:[/bold blue] {escape(path)}\n"
            f"[bold]Repository:[/bold] {host}/{owner}/{repo_name}\n"
            f"[bold]Branch:[/bold] {escape(ref or '<default branch>')}\n"
            f"[bold]Model:[/bold] {settings.llm.default_model}\n"
            f"[bold]Caps:[/bold] {max_files or caps.max_files} files, {caps.max_bytes:,} bytes\n"
            f"[bold]Output:[/bold] {', '.join(output_destinations) or '[red]none[/red]'}\n"
            f"[bold]{platform.title()} Token:[/bold] [green]found[/green] "
            f"[dim]({token_source})[/dim]",
            title=f"codespy v{__version__}",
        )
    )

    try:
        from codespy.workflows.review.models import PathReviewConfig
        from codespy.workflows.review.pipeline import ReviewPipeline
        from codespy.workflows.review.reporters.pull_request import PullRequestReporter
        from codespy.workflows.review.reporters.stdout import StdoutReporter

        pipeline = ReviewPipeline(settings)
        config = PathReviewConfig(repo_url=repo_url, path=path, ref=ref, max_files=max_files)
        result = pipeline(config)

        handle = None
        publish_error: Exception | None = None
        reporter = None
        try:
            if settings.review.output_git:
                console.print(f"[dim]Publishing report PR to {platform.title()}...[/dim]")
                reporter = PullRequestReporter(repo_url, settings)
                try:
                    handle = reporter.publish(result)
                except Exception as e:
                    # Keep going: memory phase and stdout report must still happen
                    publish_error = e
                    logging.exception("Report PR publication failed")
        finally:
            result = pipeline.finish_memory(result)

        if handle and reporter:
            reporter.update(handle, result)

        if result.llm_calls > 0:
            cost_str = f"${result.total_cost:.4f}" if result.total_cost > 0 else "N/A"
            total_in_tokens = sum(s.input_tokens for s in result.signature_stats)
            total_out_tokens = sum(s.output_tokens for s in result.signature_stats)
            total_in_cost = sum(s.input_cost for s in result.signature_stats)
            total_out_cost = sum(s.output_cost for s in result.signature_stats)
            console.print(
                Panel(
                    f"[bold]LLM Calls:[/bold] {result.llm_calls}\n"
                    f"[bold]In Tokens:[/bold] {total_in_tokens:,}  "
                    f"[bold]Out Tokens:[/bold] {total_out_tokens:,}\n"
                    f"[bold]In Cost:[/bold] ${total_in_cost:.4f}  "
                    f"[bold]Out Cost:[/bold] ${total_out_cost:.4f}\n"
                    f"[bold]Total Cost:[/bold] {cost_str}",
                    title="Cost Summary",
                )
            )

        if settings.review.output_stdout:
            StdoutReporter(format=settings.review.output_format, console=console).report(result)

        if handle:
            action = "Opened" if handle.created else "Updated"
            console.print(f"[green]✓[/green] {action} report PR: {handle.url}")
        if publish_error is not None:
            console.print(f"[red]Error publishing report PR:[/red] {escape(str(publish_error))}")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except ValueError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Error during review:[/red] {escape(str(e))}")
        logging.exception("Review failed")
        raise typer.Exit(1) from None
