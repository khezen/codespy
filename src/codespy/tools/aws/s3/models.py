"""Data models for S3 operations (filesystem-like)."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EntryType(str, Enum):
    """Type of S3 entry."""

    FILE = "file"
    DIRECTORY = "directory"


class S3Info(BaseModel):
    """Information about an S3 object or prefix (analogous to FileInfo)."""

    path: str = Field(description="Key relative to bucket root (prefix)")
    name: str = Field(description="Object name (last component of key)")
    entry_type: EntryType = Field(description="Type of entry")
    size: int = Field(default=0, description="Size in bytes (0 for directories/prefixes)")
    modified_at: datetime | None = Field(default=None, description="Last modified time")
    extension: str = Field(default="", description="File extension (empty for directories)")
    etag: str = Field(default="", description="ETag of the object (empty for directories)")
    storage_class: str = Field(default="", description="S3 storage class")


class S3Entry(BaseModel):
    """Entry in a directory listing (analogous to DirectoryEntry)."""

    name: str = Field(description="Entry name")
    entry_type: EntryType = Field(description="Type of entry")
    size: int = Field(default=0, description="Size in bytes (0 for directories)")


class S3Listing(BaseModel):
    """Result of listing an S3 prefix (analogous to DirectoryListing)."""

    path: str = Field(description="Key prefix (directory path)")
    entries: list[S3Entry] = Field(default_factory=list, description="Prefix contents")
    total_files: int = Field(default=0, description="Number of objects")
    total_directories: int = Field(default=0, description="Number of sub-prefixes")


class S3TreeNode(BaseModel):
    """Node in an S3 prefix tree (analogous to TreeNode)."""

    name: str = Field(description="Entry name")
    entry_type: EntryType = Field(description="Type of entry")
    children: list["S3TreeNode"] = Field(default_factory=list, description="Child nodes")

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


class S3Content(BaseModel):
    """Result of reading an S3 object (analogous to FileContent)."""

    path: str = Field(description="Object key")
    content: str = Field(default="", description="Object content as text")
    size: int = Field(default=0, description="Total object size in bytes")
    lines: int = Field(default=0, description="Total number of lines")
    truncated: bool = Field(default=False, description="Whether content was truncated")
    content_type: str = Field(default="", description="Content-Type of the object")
    error: str | None = Field(default=None, description="Error message if read failed")

    @property
    def success(self) -> bool:
        """Check if the object was read successfully."""
        return self.error is None


class OperationResult(BaseModel):
    """Result of a write or delete operation."""

    success: bool = Field(description="Whether the operation succeeded")
    path: str = Field(description="File path (S3 object key) that was operated on")
    message: str = Field(default="", description="Human-readable result message")
    error: str | None = Field(default=None, description="Error message if operation failed")
