"""Data models for GitHub PR data."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class FileStatus(str, Enum):
    """Status of a file in a PR."""

    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"
    RENAMED = "renamed"


# Binary file extensions that should be excluded from review
BINARY_EXTENSIONS = {
    # Images
    "png", "jpg", "jpeg", "gif", "ico", "svg", "webp", "bmp", "tiff", "tif",
    # Fonts
    "ttf", "woff", "woff2", "eot", "otf",
    # Compiled binaries
    "exe", "dll", "so", "dylib", "class", "pyc", "pyo", "o", "obj", "a", "lib",
    # Archives
    "zip", "tar", "gz", "tgz", "rar", "7z", "jar", "war", "ear", "bz2", "xz",
    # Media
    "mp3", "mp4", "wav", "avi", "mov", "webm", "ogg", "flac", "mkv",
    # Documents
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
    # Other binary
    "bin", "dat", "db", "sqlite", "sqlite3",
}

# Lock files that are auto-generated and should be excluded from review
LOCK_FILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "cargo.lock",
    "go.sum",
    "gemfile.lock",
    "composer.lock",
    "uv.lock",
    "pipfile.lock",
    "shrinkwrap.json",
    "npm-shrinkwrap.json",
}

# Directories containing vendored/generated code
EXCLUDED_DIRECTORIES = {
    "vendor",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
}


class ChangedFile(BaseModel):
    """Represents a file changed in a pull request."""

    filename: str = Field(description="Path to the file")
    status: FileStatus = Field(description="Type of change (added, modified, removed, renamed)")
    additions: int = Field(default=0, description="Number of lines added")
    deletions: int = Field(default=0, description="Number of lines deleted")
    patch: str | None = Field(default=None, description="The diff patch for this file")
    previous_filename: str | None = Field(
        default=None, description="Previous filename if renamed"
    )

    @property
    def extension(self) -> str:
        """Get the file extension."""
        if "." in self.filename:
            return self.filename.rsplit(".", 1)[-1].lower()
        return ""

    @property
    def basename(self) -> str:
        """Get the file basename (filename without path)."""
        return self.filename.rsplit("/", 1)[-1].lower()

    @property
    def is_code_file(self) -> bool:
        """Check if this is a code file based on extension."""
        code_extensions = {
            "py",
            "js",
            "ts",
            "tsx",
            "jsx",
            "go",
            "rs",
            "java",
            "kt",
            "c",
            "cpp",
            "h",
            "hpp",
            "cs",
            "rb",
            "php",
            "swift",
            "scala",
            "sh",
            "bash",
            "sql",
            "vue",
            "svelte",
        }
        return self.extension in code_extensions

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary file based on extension."""
        return self.extension in BINARY_EXTENSIONS

    @property
    def is_lock_file(self) -> bool:
        """Check if this is a lock file (auto-generated dependency file)."""
        return self.basename in LOCK_FILE_NAMES

    @property
    def is_minified(self) -> bool:
        """Check if this is a minified file."""
        basename = self.basename
        return basename.endswith(".min.js") or basename.endswith(".min.css")

    @property
    def is_source_map(self) -> bool:
        """Check if this is a source map file."""
        return self.extension == "map" or self.basename.endswith((".js.map", ".css.map"))

    @property
    def is_in_excluded_directory(self) -> bool:
        """Check if this file is in an excluded directory (vendor, node_modules, etc.)."""
        path_parts = self.filename.lower().split("/")
        return any(part in EXCLUDED_DIRECTORIES for part in path_parts)

    @property
    def should_review(self) -> bool:
        """Check if this file should be included in code review.
        
        Returns False for binary files, lock files, minified files,
        source maps, and files in excluded directories.
        """
        if self.is_binary:
            return False
        if self.is_lock_file:
            return False
        if self.is_minified:
            return False
        if self.is_source_map:
            return False
        if self.is_in_excluded_directory:
            return False
        return True


class PullRequest(BaseModel):
    """Represents a GitHub pull request."""

    number: int = Field(description="PR number")
    title: str = Field(description="PR title")
    body: str | None = Field(default=None, description="PR description/body")
    state: str = Field(description="PR state (open, closed, merged)")
    author: str = Field(description="PR author username")
    base_branch: str = Field(description="Target branch")
    head_branch: str = Field(description="Source branch")
    base_sha: str = Field(description="Base commit SHA")
    head_sha: str = Field(description="Head commit SHA")
    created_at: datetime = Field(description="PR creation timestamp")
    updated_at: datetime = Field(description="PR last update timestamp")
    repo_owner: str = Field(description="Repository owner")
    repo_name: str = Field(description="Repository name")
    changed_files: list[ChangedFile] = Field(
        default_factory=list, description="List of changed files"
    )
    labels: list[str] = Field(default_factory=list, description="PR labels")

    @property
    def repo_full_name(self) -> str:
        """Get full repository name (owner/repo)."""
        return f"{self.repo_owner}/{self.repo_name}"

    @property
    def url(self) -> str:
        """Get the PR URL."""
        return f"https://github.com/{self.repo_full_name}/pull/{self.number}"

    @property
    def total_changes(self) -> int:
        """Get total number of changed lines."""
        return sum(f.additions + f.deletions for f in self.changed_files)

    @property
    def code_files(self) -> list[ChangedFile]:
        """Get only code files from changed files."""
        return [f for f in self.changed_files if f.is_code_file]


class CallerInfo(BaseModel):
    """Information about a caller of a function/method."""

    file: str = Field(description="File containing the caller")
    line_number: int = Field(description="Line number of the call")
    line_content: str = Field(description="Content of the line")
    function_name: str = Field(description="Name of the function being called")


class ReviewContext(BaseModel):
    """Context information for code review."""

    pull_request: PullRequest = Field(description="The pull request being reviewed")
    related_files: dict[str, str] = Field(
        default_factory=dict,
        description="Related files content (imports, dependencies)",
    )
    repository_structure: str | None = Field(
        default=None, description="Overview of repository structure"
    )
    callers: dict[str, list[CallerInfo]] = Field(
        default_factory=dict,
        description="Callers of changed functions, keyed by filename",
    )

    def get_context_for_file(self, filename: str) -> str:
        """Get context string for a specific file."""
        context_parts = []

        # Add related files
        for related_name, content in self.related_files.items():
            if related_name != filename:
                context_parts.append(f"=== Related file: {related_name} ===\n{content}")

        return "\n\n".join(context_parts)

    def get_callers_for_file(self, filename: str) -> str:
        """Get formatted caller information for a specific file.

        Args:
            filename: The file to get callers for

        Returns:
            Formatted string listing all callers of functions in this file
        """
        if filename not in self.callers or not self.callers[filename]:
            return "No callers found for functions in this file."

        callers = self.callers[filename]
        lines = ["=== Verified Callers of Changed Functions ==="]

        # Group by function name
        by_function: dict[str, list[CallerInfo]] = {}
        for caller in callers:
            if caller.function_name not in by_function:
                by_function[caller.function_name] = []
            by_function[caller.function_name].append(caller)

        for func_name, func_callers in by_function.items():
            lines.append(f"\nFunction: {func_name}")
            lines.append(f"  Called from {len(func_callers)} location(s):")
            for caller in func_callers[:10]:  # Limit to 10 callers per function
                lines.append(f"    - {caller.file}:{caller.line_number}: {caller.line_content.strip()}")
            if len(func_callers) > 10:
                lines.append(f"    ... and {len(func_callers) - 10} more callers")

        return "\n".join(lines)
