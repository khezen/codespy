"""Abstract base class defining the unified storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from codespy.tools.storage.models import (
    Content,
    Entry,
    Info,
    Listing,
    OperationResult,
    TreeNode,
)


class Storage(ABC):
    """Abstract storage backend — implemented by both FileSystem and S3Client.

    Provides a uniform interface for reading, writing, and navigating file-like
    storage, whether backed by the local filesystem or an S3 bucket.
    """

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    def exists(self, path: str = "") -> bool:
        """Check whether a path exists.

        Args:
            path: Relative path to check.

        Returns:
            True if the path exists.
        """
        ...

    @abstractmethod
    def get_info(self, path: str = "") -> Info:
        """Get metadata about a file or directory.

        Args:
            path: Relative path.

        Returns:
            Info with metadata.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        ...

    @abstractmethod
    def list_directory(
        self,
        path: str = "",
        include_hidden: bool = False,
    ) -> Listing:
        """List the contents of a directory (one level deep).

        Args:
            path: Relative directory path.
            include_hidden: Whether to include hidden entries.

        Returns:
            Listing with entries.
        """
        ...

    @abstractmethod
    def read_file(
        self,
        path: str,
        max_bytes: int = 100_000,
        max_lines: int | None = None,
    ) -> Content:
        """Read a file as text.

        Args:
            path: Relative file path.
            max_bytes: Maximum bytes to read.
            max_lines: Maximum lines to read.

        Returns:
            Content with file data (error field set on failure).
        """
        ...

    @abstractmethod
    def get_tree(
        self,
        path: str = "",
        max_depth: int = 3,
        include_hidden: bool = False,
    ) -> TreeNode:
        """Get a tree representation of a directory.

        Args:
            path: Relative directory path.
            max_depth: Maximum recursion depth.
            include_hidden: Whether to include hidden entries.

        Returns:
            TreeNode representing the directory tree.
        """
        ...

    def get_tree_string(
        self,
        path: str = "",
        max_depth: int = 3,
        include_hidden: bool = False,
    ) -> str:
        """Get a string representation of the directory tree.

        Args:
            path: Relative directory path.
            max_depth: Maximum recursion depth.
            include_hidden: Whether to include hidden entries.

        Returns:
            String representation of the tree.
        """
        tree = self.get_tree(path, max_depth, include_hidden)
        return tree.to_string()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @abstractmethod
    def write_file(
        self,
        path: str,
        content: str,
        content_type: str = "text/plain",
    ) -> OperationResult:
        """Write text content to a file.

        Args:
            path: Relative file path.
            content: Text content to write (UTF-8 encoded).
            content_type: MIME type hint (used by S3; ignored by filesystem).

        Returns:
            OperationResult indicating success or failure.
        """
        ...

    @abstractmethod
    def delete_file(self, path: str) -> OperationResult:
        """Delete a file.

        Args:
            path: Relative file path to delete.

        Returns:
            OperationResult indicating success or failure.
        """
        ...
