"""Shared data models for storage operations (filesystem and S3)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class EntryType(StrEnum):
    """Type of storage entry."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


class Info(BaseModel):
    """Information about a file or directory."""

    path: str = Field(description="Relative path from root")
    name: str = Field(description="File or directory name")
    entry_type: EntryType = Field(description="Type of entry")
    size: int = Field(default=0, description="Size in bytes (0 for directories)")
    modified_at: datetime | None = Field(default=None, description="Last modified time")
    extension: str = Field(default="", description="File extension (empty for directories)")
    # S3-specific (empty strings for local filesystem)
    etag: str = Field(default="", description="ETag (S3 only)")
    storage_class: str = Field(default="", description="S3 storage class (S3 only)")

    @classmethod
    def from_path(cls, path: Path, root: Path) -> Info:
        """Create Info from a local filesystem Path.

        Args:
            path: The file path.
            root: Root directory to compute relative path.

        Returns:
            Info instance.
        """
        stat = path.stat()
        rel_path = str(path.relative_to(root))

        if path.is_symlink():
            entry_type = EntryType.SYMLINK
        elif path.is_dir():
            entry_type = EntryType.DIRECTORY
        else:
            entry_type = EntryType.FILE

        return cls(
            path=rel_path,
            name=path.name,
            entry_type=entry_type,
            size=stat.st_size if entry_type == EntryType.FILE else 0,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            extension=path.suffix.lstrip(".") if entry_type == EntryType.FILE else "",
        )


class Entry(BaseModel):
    """Entry in a directory listing."""

    name: str = Field(description="Entry name")
    entry_type: EntryType = Field(description="Type of entry")
    size: int = Field(default=0, description="Size in bytes (0 for directories)")
    modified_at: datetime | None = Field(default=None, description="Last modified time (UTC)")


class Listing(BaseModel):
    """Result of listing a directory."""

    path: str = Field(description="Directory path")
    entries: list[Entry] = Field(default_factory=list, description="Directory contents")
    total_files: int = Field(default=0, description="Number of files")
    total_directories: int = Field(default=0, description="Number of directories")


class TreeNode(BaseModel):
    """Node in a directory tree."""

    name: str = Field(description="Entry name")
    entry_type: EntryType = Field(description="Type of entry")
    children: list[TreeNode] = Field(default_factory=list, description="Child nodes")

    def to_string(self, prefix: str = "", is_last: bool = True) -> str:
        """Convert tree node to string representation.

        Args:
            prefix: Current line prefix.
            is_last: Whether this is the last sibling.

        Returns:
            String representation of the tree.
        """
        connector = "└── " if is_last else "├── "
        icon = "📁 " if self.entry_type == EntryType.DIRECTORY else "📄 "
        result = f"{prefix}{connector}{icon}{self.name}\n"

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            result += child.to_string(child_prefix, i == len(self.children) - 1)

        return result


class Content(BaseModel):
    """Result of reading a file."""

    path: str = Field(description="File path")
    content: str = Field(default="", description="File content as text")
    size: int = Field(default=0, description="Total file size in bytes")
    lines: int = Field(default=0, description="Total number of lines")
    truncated: bool = Field(default=False, description="Whether content was truncated")
    content_type: str = Field(default="", description="MIME type (S3) or empty for local files")
    error: str | None = Field(default=None, description="Error message if read failed")

    @property
    def success(self) -> bool:
        """Check if the file was read successfully."""
        return self.error is None


class OperationResult(BaseModel):
    """Result of a write or delete operation."""

    success: bool = Field(description="Whether the operation succeeded")
    path: str = Field(description="File path that was operated on")
    message: str = Field(default="", description="Human-readable result message")
    error: str | None = Field(default=None, description="Error message if operation failed")
