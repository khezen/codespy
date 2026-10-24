"""Site config file discovery and reading utilities for SEO review.

This module provides functions to discover and read crawl and hosting
configuration files anywhere in the repository, even outside the scope root.
"""

from __future__ import annotations

import subprocess
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


# Site config patterns (crawl and hosting files)
SITE_CONFIG_PATTERNS: list[str] = [
    "robots.txt",
    "robots.*",
    "sitemap*",
    "sitemap.xml*",
    "llms.txt",
    "llms-full.txt",
    "vercel.json",
    "netlify.toml",
    "_redirects",
    "_headers",
    ".htaccess",
    "nginx.conf",
    "**/nginx/**/*.conf",
    "**/apache*/**/*.conf",
    "firebase.json",
    "wrangler.toml",
    "middleware.*",
    "next.config.*",
    "nuxt.config.*",
    "astro.config.*",
    "svelte.config.*",
    "gatsby-config.*",
    "hugo.toml",
    "hugo.yaml",
    "hugo.json",
    "config.toml",
    "config.yaml",
    "_config.yml",
    ".eleventy.js",
    "eleventy.config.*",
]


def _matches_site_config(path: str, patterns: Sequence[str]) -> bool:
    """Check if path matches any site config pattern."""
    basename = Path(path).name

    for pattern in patterns:
        # Handle ** patterns for directory matching
        if "**" in pattern:
            # Split and check path parts
            parts = pattern.split("**")
            if len(parts) == 2:
                prefix, suffix = parts
                if path.startswith(prefix) and path.endswith(suffix):
                    return True
            continue
        # Check basename for simple patterns
        if fnmatch(basename, pattern) or fnmatch(path, pattern):
            return True
    return False


def find_site_config_files(
    repo_path: Path,
    excluded_directories: Sequence[str],
    max_files: int = 200,
) -> list[str]:
    """Find all site config files in the repository.

    Uses git ls-files to get tracked paths even in sparse checkout.

    Args:
        repo_path: Path to the repository root.
        excluded_directories: Directories to exclude from results.
        max_files: Maximum number of files to return.

    Returns:
        List of repo-relative paths to site config files.
    """
    # Get all tracked files using git ls-files
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        all_files = result.stdout.strip().split("\n")
    except (subprocess.CalledProcessError, OSError):
        # Fallback to listing files directly
        all_files = []
        try:
            for f in repo_path.rglob("*"):
                if f.is_file():
                    try:
                        rel_path = f.relative_to(repo_path)
                        all_files.append(str(rel_path))
                    except ValueError:
                        pass
        except OSError:
            pass

    # Filter to site config patterns
    site_files: list[str] = []
    for filepath in all_files:
        if not filepath:
            continue

        # Check excluded directories
        excluded = False
        for excluded_dir in excluded_directories:
            if excluded_dir in filepath.split("/"):
                excluded = True
                break
        if excluded:
            continue

        # Check if it matches site config patterns
        if _matches_site_config(filepath, SITE_CONFIG_PATTERNS):
            site_files.append(filepath)

    # Limit results
    return site_files[:max_files]


class SiteConfigTools:
    """Tools for reading site config files.

    These tools are not scope-restricted - they can read files anywhere
    in the repository, but only files that were discovered and in the allowlist.
    """

    def __init__(
        self,
        repo_path: Path,
        allowed_files: list[str],
        max_file_chars: int = 60_000,
    ) -> None:
        """Initialize the site config tools.

        Args:
            repo_path: Path to the repository root.
            allowed_files: List of allowed repo-relative file paths.
            max_file_chars: Maximum characters to read from a file.
        """
        self.repo_path = repo_path
        self.allowed_files = set(allowed_files)
        self.max_file_chars = max_file_chars

    def list_site_config(self) -> str:
        """List all site config files discovered in the repository.

        Returns:
            Newline-separated list of paths, each prefixed with "repo:".
        """
        if not self.allowed_files:
            return "no site config files found"
        lines = [f"repo:{f}" for f in sorted(self.allowed_files)]
        return "\n".join(lines)

    def read_site_config(self, path: str) -> str:
        """Read the content of a site config file.

        Args:
            path: File path (with or without "repo:" prefix).

        Returns:
            File content, truncated to max_file_chars.
        """
        # Strip repo: prefix if present
        if path.startswith("repo:"):
            clean_path = path[5:]
        else:
            clean_path = path

        # Security check: must be in allowed list
        if clean_path not in self.allowed_files:
            return f"error: {path} is not in the site config allowlist"

        # Security check: prevent path traversal
        try:
            full_path = self.repo_path / clean_path
            # Resolve and ensure it's under repo_path
            resolved = full_path.resolve()
            if not str(resolved).startswith(str(self.repo_path.resolve())):
                return f"error: path outside repository"
        except (OSError, ValueError):
            return f"error: invalid path"

        # Read the file
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            return content[: self.max_file_chars]
        except (OSError, FileNotFoundError):
            return f"error: file not found or unreadable"


def create_site_config_tools(
    repo_path: Path,
    excluded_directories: Sequence[str],
) -> SiteConfigTools:
    """Create site config tools for the repository.

    Args:
        repo_path: Path to the repository root.
        excluded_directories: Directories to exclude from discovery.

    Returns:
        SiteConfigTools instance ready to use.
    """
    allowed_files = find_site_config_files(repo_path, excluded_directories)
    return SiteConfigTools(repo_path, allowed_files)
