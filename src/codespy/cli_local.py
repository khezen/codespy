"""CLI commands for local git review (without GitHub/GitLab)."""

import logging
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from codespy.config import get_settings

console = Console()


def review_local(
    repo_path: Annotated[
        str | None,
        typer.Argument(
            help="Path to git repository (defaults to current directory)",
        ),
    ] = None,
    base_ref: Annotated[
        str,
        typer.Option(
            "--base",
            "-b",
            help="Base git ref to diff against (e.g., 'main', 'develop', 'origin/main', 'HEAD~5')",
        ),
    ] = "main",
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
) -> None:
    """Review local git changes (current branch vs base) without GitHub/GitLab.

    Reviews committed changes on your current branch compared to a base ref
    (e.g., main, develop). No remote platform needed - works with any local repo.

    Examples:
        codespy review-local                    # Review current dir vs main
        codespy review-local /path/to/repo      # Review specific repo
        codespy review-local --base develop     # Compare against develop
        codespy review-local --base origin/main # Compare against origin/main
        codespy review-local --base HEAD~5      # Compare against 5 commits back
        codespy review-local --output json      # Output as JSON
    """
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

    if model:
        settings.default_model = model
    if output:
        settings.output_format = output  # type: ignore

    repo = Path(repo_path if repo_path else os.getcwd()).resolve()
    
    if not repo.exists():
        console.print(f"[red]Error:[/red] Directory does not exist: {repo}")
        raise typer.Exit(1)
    
    if not (repo / ".git").exists():
        console.print(f"[red]Error:[/red] Not a git repository: {repo}")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold blue]Reviewing local changes[/bold blue]\n"
            f"[bold]Repository:[/bold] {repo}\n"
            f"[bold]Base ref:[/bold] {base_ref}\n"
            f"[bold]Model:[/bold] {settings.default_model}\n"
            f"[bold]Output:[/bold] {settings.output_format}",
            title="codespy review-local",
        )
    )

    try:
        from codespy.agents.reviewer.reviewer import ReviewPipeline
        from codespy.agents.reviewer.models import LocalReviewConfig

        pipeline = ReviewPipeline(settings)

        # Create local review config
        config = LocalReviewConfig(
            repo_path=repo,
            base_ref=base_ref,
            uncommitted=False
        )
        
        # Run review (model access always verified in pipeline)
        result = pipeline(config)

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

        from codespy.agents.reviewer.reporters import StdoutReporter
        stdout_reporter = StdoutReporter(format=settings.output_format, console=console)
        stdout_reporter.report(result)

    except Exception as e:
        console.print(f"[red]Error during review:[/red] {e}")
        logging.exception("Review failed")
        raise typer.Exit(1)


def review_uncommitted(
    repo_path: Annotated[
        str | None,
        typer.Argument(
            help="Path to git repository (defaults to current directory)",
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
) -> None:
    """Review uncommitted changes (staged + unstaged) in working tree.

    Reviews all modifications in your working tree that haven't been committed yet.
    Useful for checking your work before committing.

    Examples:
        codespy review-uncommitted              # Review current dir
        codespy review-uncommitted /path/to/repo
        codespy review-uncommitted --output json
    """
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

    if model:
        settings.default_model = model
    if output:
        settings.output_format = output  # type: ignore

    repo = Path(repo_path if repo_path else os.getcwd()).resolve()
    
    if not repo.exists():
        console.print(f"[red]Error:[/red] Directory does not exist: {repo}")
        raise typer.Exit(1)
    
    if not (repo / ".git").exists():
        console.print(f"[red]Error:[/red] Not a git repository: {repo}")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold blue]Reviewing uncommitted changes[/bold blue]\n"
            f"[bold]Repository:[/bold] {repo}\n"
            f"[bold]Model:[/bold] {settings.default_model}\n"
            f"[bold]Output:[/bold] {settings.output_format}",
            title="codespy review-uncommitted",
        )
    )

    try:
        from codespy.agents.reviewer.reviewer import ReviewPipeline
        from codespy.agents.reviewer.models import LocalReviewConfig

        pipeline = ReviewPipeline(settings)

        # Create local review config for uncommitted changes
        config = LocalReviewConfig(
            repo_path=repo,
            uncommitted=True
        )
        
        # Run review (model access always verified in pipeline)
        result = pipeline(config)

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

        from codespy.agents.reviewer.reporters import StdoutReporter
        stdout_reporter = StdoutReporter(format=settings.output_format, console=console)
        stdout_reporter.report(result)

    except Exception as e:
        console.print(f"[red]Error during review:[/red] {e}")
        logging.exception("Review failed")
        raise typer.Exit(1)
