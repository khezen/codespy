"""I/O configuration: output settings and file exclusions."""

from typing import Literal

# Output format type
OutputFormat = Literal["markdown", "json"]

# Default excluded directories - directories to skip during code review
# Binary files, lock files, and minified files are always excluded automatically
DEFAULT_EXCLUDED_DIRECTORIES: list[str] = [
    # Vendor/dependency directories
    "vendor",
    "node_modules",
    "third_party",
    "external",
    "deps",
    "_vendor",
    "vendored",
    # Build output directories
    "dist",
    "build",
    "out",
    "target",
    # Package manager directories
    ".bundle",
    "Pods",
    "Carthage",
    "bower_components",
    "jspm_packages",
    # Version control
    ".git",
    ".svn",
    ".hg",
    # Cache directories
    "__pycache__",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
]