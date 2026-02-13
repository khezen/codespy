"""Command-line interface for codespy."""

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from codespy import __version__
from codespy.config import get_settings
from codespy.config_git import get_github_token_source, get_gitlab_token_source
from codespy.tools.git.client import detect_platform, is_supported_url

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
        from codespy.agents import verify_model_access

        pipeline = ReviewPipeline(settings)

        # Verify model access
        console.print("[dim]Verifying model access...[/dim]")
        success, message = verify_model_access(settings)
        if success:
            console.print(f"[bold]Model:[/bold] [green]verified[/green] [dim]({settings.default_model})[/dim]")
        else:
            console.print(f"[red]Error:[/red] {message}")
            raise typer.Exit(1)

        result = pipeline(mr_url, verify_model=False)  # Already verified

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


@app.command()
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
    import os
    from pathlib import Path

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

    # Override settings if provided via CLI
    if model:
        settings.default_model = model
    if output:
        settings.output_format = output  # type: ignore

    # Resolve repo path
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
        from codespy.agents import verify_model_access
        from codespy.tools.git.local_diff import build_mr_from_diff

        # Verify model access
        console.print("[dim]Verifying model access...[/dim]")
        success, message = verify_model_access(settings)
        if success:
            console.print(f"[bold]Model:[/bold] [green]verified[/green] [dim]({settings.default_model})[/dim]")
        else:
            console.print(f"[red]Error:[/red] {message}")
            raise typer.Exit(1)

        # Build MergeRequest from local diff
        console.print(f"[dim]Building diff: HEAD vs {base_ref}...[/dim]")
        mr = build_mr_from_diff(repo, base_ref=base_ref, include_uncommitted=False)

        if not mr.changed_files:
            console.print(f"[yellow]No changes found between {base_ref} and HEAD[/yellow]")
            raise typer.Exit(0)

        console.print(f"[dim]Found {len(mr.changed_files)} changed files[/dim]")

        # Run review pipeline
        pipeline = ReviewPipeline(settings)
        result = pipeline(mr=mr, repo_path=repo, verify_model=False)

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
        from codespy.agents.reviewer.reporters import StdoutReporter
        stdout_reporter = StdoutReporter(format=settings.output_format, console=console)
        stdout_reporter.report(result)

    except Exception as e:
        console.print(f"[red]Error during review:[/red] {e}")
        logging.exception("Review failed")
        raise typer.Exit(1)


@app.command()
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
    import os
    from pathlib import Path

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

    # Override settings if provided via CLI
    if model:
        settings.default_model = model
    if output:
        settings.output_format = output  # type: ignore

    # Resolve repo path
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
        from codespy.agents import verify_model_access
        from codespy.tools.git.local_diff import build_mr_from_diff

        # Verify model access
        console.print("[dim]Verifying model access...[/dim]")
        success, message = verify_model_access(settings)
        if success:
            console.print(f"[bold]Model:[/bold] [green]verified[/green] [dim]({settings.default_model})[/dim]")
        else:
            console.print(f"[red]Error:[/red] {message}")
            raise typer.Exit(1)

        # Build MergeRequest from uncommitted changes
        console.print("[dim]Building diff: working tree vs HEAD...[/dim]")
        mr = build_mr_from_diff(repo, include_uncommitted=True)

        if not mr.changed_files:
            console.print("[yellow]No uncommitted changes found[/yellow]")
            raise typer.Exit(0)

        console.print(f"[dim]Found {len(mr.changed_files)} changed files[/dim]")

        # Run review pipeline
        pipeline = ReviewPipeline(settings)
        result = pipeline(mr=mr, repo_path=repo, verify_model=False)

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
        from codespy.agents.reviewer.reporters import StdoutReporter
        stdout_reporter = StdoutReporter(format=settings.output_format, console=console)
        stdout_reporter.report(result)

    except Exception as e:
        console.print(f"[red]Error during review:[/red] {e}")
        logging.exception("Review failed")
        raise typer.Exit(1)


@app.command()
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
    # Load settings before starting (validates config file exists)
    if config_file is not None:
        import os
        os.environ["CODESPY_CONFIG_FILE"] = config_file

    from codespy.agents.reviewer.server import run_server
    run_server()


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