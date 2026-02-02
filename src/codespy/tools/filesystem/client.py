"""FileSystem client for file operations."""

import logging
from pathlib import Path

from codespy.tools.filesystem.models import (
    DirectoryEntry,
    DirectoryListing,
    EntryType,
    FileContent,
    FileInfo,
    TreeNode,
)

logger = logging.getLogger(__name__)


class FileSystem:
    """Client for filesystem operations.

    Provides secure file operations restricted to a root directory.
    """

    # Directories to skip when traversing
    SKIP_DIRS = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        "dist",
        "build",
        ".eggs",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }

    def __init__(self, root: str | Path) -> None:
        """Initialize the filesystem client.

        Args:
            root: Root directory for all operations
        """
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise ValueError(f"Root directory does not exist: {self.root}")
        if not self.root.is_dir():
            raise ValueError(f"Root is not a directory: {self.root}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to root, with security checks.

        Args:
            path: Relative path

        Returns:
            Absolute path

        Raises:
            ValueError: If path escapes root directory
        """
        if not path or path == ".":
            return self.root

        resolved = (self.root / path).resolve()

        # Security check: ensure path is within root
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ValueError(f"Path escapes root directory: {path}")

        return resolved

    def exists(self, path: str = "") -> bool:
        """Check if a path exists.

        Args:
            path: Relative path to check

        Returns:
            True if path exists
        """
        try:
            resolved = self._resolve_path(path)
            return resolved.exists()
        except ValueError:
            return False

    def get_info(self, path: str = "") -> FileInfo:
        """Get information about a file or directory.

        Args:
            path: Relative path

        Returns:
            FileInfo with metadata

        Raises:
            FileNotFoundError: If path does not exist
        """
        resolved = self._resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        return FileInfo.from_path(resolved, self.root)

    def list_directory(
        self,
        path: str = "",
        include_hidden: bool = False,
    ) -> DirectoryListing:
        """List contents of a directory.

        Args:
            path: Relative path to directory
            include_hidden: Whether to include hidden files (starting with .)

        Returns:
            DirectoryListing with entries

        Raises:
            FileNotFoundError: If path does not exist
            NotADirectoryError: If path is not a directory
        """
        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        entries: list[DirectoryEntry] = []
        total_files = 0
        total_directories = 0

        try:
            for entry in sorted(resolved.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                # Skip hidden files unless requested
                if not include_hidden and entry.name.startswith("."):
                    continue

                if entry.is_symlink():
                    entry_type = EntryType.SYMLINK
                elif entry.is_dir():
                    entry_type = EntryType.DIRECTORY
                    total_directories += 1
                else:
                    entry_type = EntryType.FILE
                    total_files += 1

                size = entry.stat().st_size if entry_type == EntryType.FILE else 0

                entries.append(
                    DirectoryEntry(
                        name=entry.name,
                        entry_type=entry_type,
                        size=size,
                    )
                )
        except PermissionError as e:
            logger.warning(f"Permission denied listing {path}: {e}")

        rel_path = str(resolved.relative_to(self.root)) if resolved != self.root else "."

        return DirectoryListing(
            path=rel_path,
            entries=entries,
            total_files=total_files,
            total_directories=total_directories,
        )

    def read_file(
        self,
        path: str,
        max_bytes: int = 100_000,
        max_lines: int | None = None,
    ) -> FileContent:
        """Read contents of a file.

        Args:
            path: Relative path to file
            max_bytes: Maximum bytes to read (default 100KB)
            max_lines: Maximum lines to read (optional)

        Returns:
            FileContent with file data

        Raises:
            FileNotFoundError: If file does not exist
            IsADirectoryError: If path is a directory
        """
        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if resolved.is_dir():
            raise IsADirectoryError(f"Cannot read directory: {path}")

        file_size = resolved.stat().st_size
        truncated = False

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with latin-1 for binary-ish files
            try:
                content = resolved.read_text(encoding="latin-1")
            except Exception:
                raise ValueError(f"Cannot read file as text: {path}")

        total_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

        # Truncate by bytes
        if len(content) > max_bytes:
            content = content[:max_bytes]
            truncated = True

        # Truncate by lines
        if max_lines is not None:
            lines = content.split("\n")
            if len(lines) > max_lines:
                content = "\n".join(lines[:max_lines])
                truncated = True

        rel_path = str(resolved.relative_to(self.root))

        return FileContent(
            path=rel_path,
            content=content,
            size=file_size,
            lines=total_lines,
            truncated=truncated,
        )

    def get_tree(
        self,
        path: str = "",
        max_depth: int = 3,
        include_hidden: bool = False,
    ) -> TreeNode:
        """Get a tree representation of a directory.

        Args:
            path: Relative path to directory
            max_depth: Maximum depth to traverse
            include_hidden: Whether to include hidden files

        Returns:
            TreeNode representing the directory structure

        Raises:
            FileNotFoundError: If path does not exist
            NotADirectoryError: If path is not a directory
        """
        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        return self._build_tree(resolved, max_depth, include_hidden, 0)

    def _build_tree(
        self,
        path: Path,
        max_depth: int,
        include_hidden: bool,
        current_depth: int,
    ) -> TreeNode:
        """Recursively build a tree structure.

        Args:
            path: Current path
            max_depth: Maximum depth
            include_hidden: Include hidden files
            current_depth: Current recursion depth

        Returns:
            TreeNode for this directory
        """
        entry_type = EntryType.DIRECTORY if path.is_dir() else EntryType.FILE
        children: list[TreeNode] = []

        if path.is_dir() and current_depth < max_depth:
            try:
                entries = sorted(
                    path.iterdir(),
                    key=lambda x: (x.is_file(), x.name.lower()),
                )

                for entry in entries:
                    # Skip hidden files
                    if not include_hidden and entry.name.startswith("."):
                        continue

                    # Skip common uninteresting directories
                    if entry.is_dir() and entry.name in self.SKIP_DIRS:
                        continue

                    child = self._build_tree(
                        entry,
                        max_depth,
                        include_hidden,
                        current_depth + 1,
                    )
                    children.append(child)

            except PermissionError:
                pass

        return TreeNode(
            name=path.name or str(path),
            entry_type=entry_type,
            children=children,
        )

    def get_tree_string(
        self,
        path: str = "",
        max_depth: int = 3,
        include_hidden: bool = False,
    ) -> str:
        """Get a string representation of the directory tree.

        Args:
            path: Relative path to directory
            max_depth: Maximum depth to traverse
            include_hidden: Whether to include hidden files

        Returns:
            String representation of the tree
        """
        tree = self.get_tree(path, max_depth, include_hidden)
        return tree.to_string()