"""Data models for Git merge requests (GitHub PRs and GitLab MRs)."""

import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class FileStatus(str, Enum):
    """Status of a file in a merge request."""

    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"
    RENAMED = "renamed"


class GitPlatform(str, Enum):
    """Supported Git platforms."""

    GITHUB = "github"
    GITLAB = "gitlab"


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


class ChangedFile(BaseModel):
    """Represents a file changed in a merge request."""

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

    def is_in_excluded_directory(self, excluded_directories: list[str]) -> bool:
        """Check if this file is in an excluded directory.
        
        Args:
            excluded_directories: List of directory names to exclude (from settings)
        """
        path_parts = self.filename.lower().split("/")
        excluded_set = {d.lower() for d in excluded_directories}
        return any(part in excluded_set for part in path_parts)

    @property
    def valid_new_line_numbers(self) -> set[int]:
        """Get line numbers in the new file that are valid for inline comments.
        
        Parses the unified diff patch to extract line numbers where inline comments
        can be placed. Only lines that appear in the diff (additions and context lines)
        are valid for GitHub/GitLab review comments.
        
        Returns:
            Set of valid line numbers in the new version of the file
        """
        if not self.patch:
            return set()

        valid_lines: set[int] = set()
        current_new_line = 0

        for line in self.patch.split("\n"):
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            if line.startswith("@@"):
                match = re.search(r"\+(\d+)", line)
                if match:
                    current_new_line = int(match.group(1))
                continue

            if not current_new_line:
                continue

            # Context line (unchanged) - valid for comments
            if line.startswith(" "):
                valid_lines.add(current_new_line)
                current_new_line += 1
            # Addition line - valid for comments
            elif line.startswith("+"):
                valid_lines.add(current_new_line)
                current_new_line += 1
            # Deletion line - doesn't increment new line counter (not in new file)
            elif line.startswith("-"):
                pass
            # Other lines (like "\ No newline at end of file") - ignore
            else:
                pass

        return valid_lines

    def is_line_in_diff(self, line_number: int) -> bool:
        """Check if a line number is valid for inline comments.
        
        Args:
            line_number: Line number to check
            
        Returns:
            True if the line is part of the diff and can receive inline comments
        """
        return line_number in self.valid_new_line_numbers


def should_review_file(file: ChangedFile, excluded_directories: list[str]) -> bool:
    """Check if a file should be included in code review.
    
    Args:
        file: The ChangedFile to check
        excluded_directories: List of directory names to exclude (from settings)
        
    Returns:
        True if file should be reviewed, False if it should be skipped
    """
    if file.is_binary:
        return False
    if file.is_lock_file:
        return False
    if file.is_minified:
        return False
    if file.is_source_map:
        return False
    if file.is_in_excluded_directory(excluded_directories):
        return False
    return True


class MergeRequest(BaseModel):
    """Represents a merge request (GitHub PR or GitLab MR)."""

    number: int = Field(description="MR/PR number")
    title: str = Field(description="MR/PR title")
    body: str | None = Field(default=None, description="MR/PR description/body")
    state: str = Field(description="MR/PR state (open, closed, merged)")
    author: str = Field(description="MR/PR author username")
    base_branch: str = Field(description="Target branch")
    head_branch: str = Field(description="Source branch")
    base_sha: str = Field(description="Base commit SHA")
    head_sha: str = Field(description="Head commit SHA")
    created_at: datetime = Field(description="MR/PR creation timestamp")
    updated_at: datetime = Field(description="MR/PR last update timestamp")
    repo_owner: str = Field(description="Repository owner/namespace")
    repo_name: str = Field(description="Repository name")
    changed_files: list[ChangedFile] = Field(
        default_factory=list, description="List of changed files"
    )
    labels: list[str] = Field(default_factory=list, description="MR/PR labels")
    platform: GitPlatform = Field(description="Git platform (github, gitlab)")

    @property
    def repo_full_name(self) -> str:
        """Get full repository name (owner/repo)."""
        return f"{self.repo_owner}/{self.repo_name}"

    @property
    def url(self) -> str:
        """Get the MR/PR URL."""
        if self.platform == GitPlatform.GITLAB:
            return f"https://gitlab.com/{self.repo_full_name}/-/merge_requests/{self.number}"
        return f"https://github.com/{self.repo_full_name}/pull/{self.number}"

    @property
    def total_changes(self) -> int:
        """Get total number of changed lines."""
        return sum(f.additions + f.deletions for f in self.changed_files)

    @property
    def code_files(self) -> list[ChangedFile]:
        """Get only code files from changed files."""
        return [f for f in self.changed_files if f.is_code_file]


# Alias for backward compatibility
PullRequest = MergeRequest


class CallerInfo(BaseModel):
    """Information about a caller of a function/method."""

    file: str = Field(description="File containing the caller")
    line_number: int = Field(description="Line number of the call")
    line_content: str = Field(description="Content of the line")
    function_name: str = Field(description="Name of the function being called")


class ReviewContext(BaseModel):
    """Context information for code review."""

    merge_request: MergeRequest = Field(description="The merge request being reviewed")
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

    # Alias for backward compatibility
    @property
    def pull_request(self) -> MergeRequest:
        """Alias for merge_request (backward compatibility)."""
        return self.merge_request

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