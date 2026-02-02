"""Data models for filesystem operations."""

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class EntryType(str, Enum):
    """Type of filesystem entry."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


class FileInfo(BaseModel):
    """Information about a file or directory."""

    path: str = Field(description="Relative path from root")
    name: str = Field(description="File or directory name")
    entry_type: EntryType = Field(description="Type of entry")
    size: int = Field(default=0, description="Size in bytes (0 for directories)")
    modified_at: datetime | None = Field(default=None, description="Last modified time")
    extension: str = Field(default="", description="File extension (empty for directories)")

    @classmethod
    def from_path(cls, path: Path, root: Path) -> "FileInfo":
        """Create FileInfo from a Path object.

        Args:
            path: The file path
            root: Root directory to compute relative path

        Returns:
            FileInfo instance
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


class DirectoryEntry(BaseModel):
    """Entry in a directory listing."""

    name: str = Field(description="Entry name")
    entry_type: EntryType = Field(description="Type of entry")
    size: int = Field(default=0, description="Size in bytes")


class DirectoryListing(BaseModel):
    """Result of listing a directory."""

    path: str = Field(description="Directory path")
    entries: list[DirectoryEntry] = Field(default_factory=list, description="Directory contents")
    total_files: int = Field(default=0, description="Number of files")
    total_directories: int = Field(default=0, description="Number of directories")


class TreeNode(BaseModel):
    """Node in a directory tree."""

    name: str = Field(description="Entry name")
    entry_type: EntryType = Field(description="Type of entry")
    children: list["TreeNode"] = Field(default_factory=list, description="Child nodes")

    def to_string(self, prefix: str = "", is_last: bool = True) -> str:
        """Convert tree node to string representation.

        Args:
            prefix: Current line prefix
            is_last: Whether this is the last sibling

        Returns:
            String representation of the tree
        """
        connector = "└── " if is_last else "├── "
        icon = "📁 " if self.entry_type == EntryType.DIRECTORY else "📄 "
        result = f"{prefix}{connector}{icon}{self.name}\n"

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            result += child.to_string(child_prefix, i == len(self.children) - 1)

        return result


class FileContent(BaseModel):
    """Result of reading a file."""

    path: str = Field(description="File path")
    content: str = Field(description="File content")
    size: int = Field(description="Total file size in bytes")
    lines: int = Field(description="Total number of lines")
    truncated: bool = Field(default=False, description="Whether content was truncated")