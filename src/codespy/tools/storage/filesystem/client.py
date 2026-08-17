"""FileSystem client for local filesystem operations."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from codespy.tools.storage.base import Storage
from codespy.tools.storage.models import (
    Content,
    Entry,
    EntryType,
    Info,
    Listing,
    OperationResult,
    TreeNode,
)

logger = logging.getLogger(__name__)


class FileSystem(Storage):
    """Local filesystem storage client.

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

    def __init__(self, root: str | Path, create_if_missing: bool = True) -> None:
        """Initialize the filesystem client.

        Args:
            root: Root directory for all operations.
            create_if_missing: Create the root directory if it doesn't exist.
        """
        self.root = Path(root).resolve()
        if not self.root.exists():
            if create_if_missing:
                self.root.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created root directory: {self.root}")
            else:
                raise ValueError(f"Root directory does not exist: {self.root}")
        if not self.root.is_dir():
            raise ValueError(f"Root is not a directory: {self.root}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to root, with security checks.

        Args:
            path: Relative path.

        Returns:
            Absolute path.

        Raises:
            ValueError: If path escapes root directory.
        """
        if not path or path == ".":
            return self.root

        resolved = (self.root / path).resolve()

        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ValueError(f"Path escapes root directory: {path}")

        return resolved

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def exists(self, path: str = "") -> bool:
        """Check if a path exists."""
        try:
            resolved = self._resolve_path(path)
            return resolved.exists()
        except ValueError:
            return False

    def verify_access(self) -> None:
        """Verify the filesystem root is accessible.

        Raises:
            FileNotFoundError: If the memory root does not exist.
            NotADirectoryError: If the memory root is not a directory.
            PermissionError: If the memory root is not readable.
        """
        if not self.root.exists():
            raise FileNotFoundError(f"Memory root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Memory root is not a directory: {self.root}")
        if not os.access(self.root, os.R_OK):
            raise PermissionError(f"Memory root is not readable: {self.root}")

    def get_info(self, path: str = "") -> Info:
        """Get information about a file or directory."""
        resolved = self._resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        return Info.from_path(resolved, self.root)

    def list_directory(
        self,
        path: str = "",
        include_hidden: bool = False,
    ) -> Listing:
        """List contents of a directory."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        entries: list[Entry] = []
        total_files = 0
        total_directories = 0

        try:
            for entry in sorted(resolved.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
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

                stat = entry.stat()
                size = stat.st_size if entry_type == EntryType.FILE else 0
                modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC) if entry_type == EntryType.FILE else None

                entries.append(Entry(name=entry.name, entry_type=entry_type, size=size, modified_at=modified_at))
        except PermissionError as e:
            logger.warning(f"Permission denied listing {path}: {e}")

        rel_path = str(resolved.relative_to(self.root)) if resolved != self.root else "."

        return Listing(
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
    ) -> Content:
        """Read contents of a file."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            return Content(path=path, error=f"File not found: {path}")
        if resolved.is_dir():
            return Content(path=path, error=f"Cannot read directory: {path}")

        file_size = resolved.stat().st_size
        truncated = False

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = resolved.read_text(encoding="latin-1")
            except Exception:
                return Content(path=path, error=f"Cannot read file as text: {path}", size=file_size)

        total_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

        if len(content) > max_bytes:
            content = content[:max_bytes]
            truncated = True

        if max_lines is not None:
            lines = content.split("\n")
            if len(lines) > max_lines:
                content = "\n".join(lines[:max_lines])
                truncated = True

        rel_path = str(resolved.relative_to(self.root))

        return Content(
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
        """Get a tree representation of a directory."""
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
        entry_type = EntryType.DIRECTORY if path.is_dir() else EntryType.FILE
        children: list[TreeNode] = []

        if path.is_dir() and current_depth < max_depth:
            try:
                entries = sorted(
                    path.iterdir(),
                    key=lambda x: (x.is_file(), x.name.lower()),
                )
                for entry in entries:
                    if not include_hidden and entry.name.startswith("."):
                        continue
                    if entry.is_dir() and entry.name in self.SKIP_DIRS:
                        continue
                    child = self._build_tree(entry, max_depth, include_hidden, current_depth + 1)
                    children.append(child)
            except PermissionError:
                pass

        return TreeNode(name=path.name or str(path), entry_type=entry_type, children=children)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write_file(
        self,
        path: str,
        content: str,
        content_type: str = "text/plain",
    ) -> OperationResult:
        """Write text content to a file.

        Creates parent directories as needed.

        Args:
            path: Relative file path to write to.
            content: Text content to write (UTF-8).
            content_type: Ignored for local files; accepted for interface compatibility.

        Returns:
            OperationResult indicating success or failure.
        """
        try:
            resolved = self._resolve_path(path)
        except ValueError as e:
            return OperationResult(success=False, path=path, error=str(e))

        if not path:
            return OperationResult(success=False, path=path, error="Cannot write: path is empty")

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return OperationResult(
                success=True,
                path=str(resolved.relative_to(self.root)),
                message=f"Written {len(content.encode('utf-8'))} bytes to {resolved}",
            )
        except Exception as e:
            logger.error(f"write_file failed for {path!r}: {e}")
            return OperationResult(success=False, path=path, error=str(e))

    def delete_file(self, path: str) -> OperationResult:
        """Delete a file.

        Args:
            path: Relative file path to delete.

        Returns:
            OperationResult indicating success or failure.
        """
        try:
            resolved = self._resolve_path(path)
        except ValueError as e:
            return OperationResult(success=False, path=path, error=str(e))

        if not resolved.exists():
            return OperationResult(success=False, path=path, error=f"File not found: {path}")
        if resolved.is_dir():
            return OperationResult(success=False, path=path, error=f"Path is a directory: {path}")

        try:
            resolved.unlink()
            return OperationResult(
                success=True,
                path=path,
                message=f"Deleted {resolved}",
            )
        except Exception as e:
            logger.error(f"delete_file failed for {path!r}: {e}")
            return OperationResult(success=False, path=path, error=str(e))
