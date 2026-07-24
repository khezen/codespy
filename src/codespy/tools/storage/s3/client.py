"""S3 client for filesystem-like operations over a single bucket."""

from __future__ import annotations

import logging

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


class S3Client(Storage):
    """Client for S3 operations rooted at a single bucket.

    Treats S3 key prefixes as directories and individual object keys as files,
    mirroring the FileSystem client interface. All path arguments are relative
    to the bucket root (no leading slash needed).

    Authentication uses the standard boto3 credential chain:
    env vars (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY), ~/.aws/credentials,
    IAM instance role, etc.
    """

    def __init__(
        self,
        bucket: str,
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        """Initialize the S3 client rooted at a bucket.

        Args:
            bucket: S3 bucket name — all operations are scoped to this bucket.
            region: AWS region (e.g. 'us-east-1'). Falls back to boto3 defaults.
            endpoint_url: Custom endpoint URL for S3-compatible stores (e.g. MinIO).
        """
        import boto3  # type: ignore[import-untyped]

        self.bucket = bucket

        kwargs: dict = {}
        if region:
            kwargs["region_name"] = region
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url

        self._s3 = boto3.client("s3", **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str) -> str:
        normalised = path.lstrip("/")
        parts = normalised.split("/")
        resolved: list[str] = []
        for part in parts:
            if part == "..":
                raise ValueError(f"Path escapes bucket root: {path!r}")
            if part and part != ".":
                resolved.append(part)
        return "/".join(resolved)

    def _file_name(self, path: str) -> str:
        return path.rstrip("/").rsplit("/", 1)[-1]

    def _file_extension(self, path: str) -> str:
        name = self._file_name(path)
        if "." in name:
            return name.rsplit(".", 1)[-1]
        return ""

    def _client_error_code(self, exc: Exception) -> str:
        try:
            return exc.response["Error"]["Code"]  # type: ignore[attr-defined]
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def exists(self, path: str = "") -> bool:
        """Check whether a file or directory exists in the bucket."""
        file_path = self._resolve_path(path)

        if not file_path:
            return True

        try:
            self._s3.head_object(Bucket=self.bucket, Key=file_path)
            return True
        except Exception as e:
            if self._client_error_code(e) not in ("404", "NoSuchKey"):
                logger.warning(f"HeadObject error for {file_path!r}: {e}")

        dir_prefix = file_path if file_path.endswith("/") else file_path + "/"
        try:
            resp = self._s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=dir_prefix,
                MaxKeys=1,
            )
            return bool(resp.get("Contents") or resp.get("CommonPrefixes"))
        except Exception as e:
            logger.warning(f"ListObjectsV2 error for prefix {dir_prefix!r}: {e}")
            return False

    def get_info(self, path: str = "") -> Info:
        """Get metadata about a file or directory."""
        file_path = self._resolve_path(path)

        if not file_path:
            return Info(
                path=".",
                name=self.bucket,
                entry_type=EntryType.DIRECTORY,
            )

        try:
            resp = self._s3.head_object(Bucket=self.bucket, Key=file_path)
            return Info(
                path=file_path,
                name=self._file_name(file_path),
                entry_type=EntryType.FILE,
                size=resp.get("ContentLength", 0),
                modified_at=resp.get("LastModified"),
                extension=self._file_extension(file_path),
                etag=resp.get("ETag", "").strip('"'),
                storage_class=resp.get("StorageClass", "STANDARD"),
            )
        except Exception as e:
            if self._client_error_code(e) not in ("404", "NoSuchKey"):
                raise

        dir_prefix = file_path if file_path.endswith("/") else file_path + "/"
        resp = self._s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=dir_prefix,
            MaxKeys=1,
        )
        if resp.get("Contents") or resp.get("CommonPrefixes"):
            return Info(
                path=file_path,
                name=self._file_name(file_path),
                entry_type=EntryType.DIRECTORY,
            )

        raise FileNotFoundError(f"Path not found in bucket {self.bucket!r}: {file_path!r}")

    def list_directory(
        self,
        path: str = "",
        include_hidden: bool = False,
    ) -> Listing:
        """List files and subdirectories directly under a path (one level deep)."""
        dir_path = self._resolve_path(path)
        prefix = (dir_path + "/") if dir_path else ""

        entries: list[Entry] = []
        total_files = 0
        total_directories = 0
        continuation_token: str | None = None

        while True:
            kwargs: dict = {
                "Bucket": self.bucket,
                "Delimiter": "/",
                "Prefix": prefix,
                "MaxKeys": 1000,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            try:
                resp = self._s3.list_objects_v2(**kwargs)
            except Exception as e:
                logger.error(f"Error listing {prefix!r}: {e}")
                break

            for cp in resp.get("CommonPrefixes", []):
                sub = cp.get("Prefix", "")
                name = sub.rstrip("/").rsplit("/", 1)[-1]
                if not include_hidden and name.startswith("."):
                    continue
                entries.append(Entry(name=name, entry_type=EntryType.DIRECTORY))
                total_directories += 1

            for obj in resp.get("Contents", []):
                file_key = obj.get("Key", "")
                if file_key == prefix:
                    continue
                name = file_key.rsplit("/", 1)[-1]
                if not include_hidden and name.startswith("."):
                    continue
                entries.append(
                    Entry(name=name, entry_type=EntryType.FILE, size=obj.get("Size", 0))
                )
                total_files += 1

            if not resp.get("IsTruncated"):
                break
            continuation_token = resp.get("NextContinuationToken")

        entries.sort(key=lambda e: (e.entry_type == EntryType.FILE, e.name.lower()))

        return Listing(
            path=dir_path or ".",
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
        """Read a file from S3 as text."""
        file_path = self._resolve_path(path)
        if not file_path:
            return Content(path=path, error="Cannot read: path is empty (bucket root)")

        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=file_path)
        except Exception as e:
            return Content(path=file_path, error=f"GetObject failed: {e}")

        size: int = resp.get("ContentLength", 0)
        content_type: str = resp.get("ContentType", "")
        truncated = False

        try:
            raw: bytes = resp["Body"].read(max_bytes + 1)
        except Exception as e:
            return Content(path=file_path, error=f"Error reading body: {e}", size=size, content_type=content_type)

        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
            truncated = True

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = raw.decode("latin-1")
            except Exception:
                return Content(
                    path=file_path,
                    error="Cannot decode file as text (binary content)",
                    size=size,
                    content_type=content_type,
                )

        total_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

        if max_lines is not None:
            lines = content.split("\n")
            if len(lines) > max_lines:
                content = "\n".join(lines[:max_lines])
                truncated = True

        return Content(
            path=file_path,
            content=content,
            size=size,
            lines=total_lines,
            truncated=truncated,
            content_type=content_type,
        )

    def get_tree(
        self,
        path: str = "",
        max_depth: int = 3,
        include_hidden: bool = False,
    ) -> TreeNode:
        """Get a tree representation of a directory in the bucket."""
        dir_path = self._resolve_path(path)
        name = self._file_name(dir_path) if dir_path else self.bucket
        return self._build_tree(dir_path, name, max_depth, include_hidden, 0)

    def _build_tree(
        self,
        dir_path: str,
        name: str,
        max_depth: int,
        include_hidden: bool,
        current_depth: int,
    ) -> TreeNode:
        if current_depth >= max_depth:
            return TreeNode(name=name, entry_type=EntryType.DIRECTORY)

        listing = self.list_directory(dir_path, include_hidden=include_hidden)
        children: list[TreeNode] = []

        for entry in listing.entries:
            if entry.entry_type == EntryType.DIRECTORY:
                child_path = f"{dir_path}/{entry.name}" if dir_path else entry.name
                child = self._build_tree(child_path, entry.name, max_depth, include_hidden, current_depth + 1)
            else:
                child = TreeNode(name=entry.name, entry_type=EntryType.FILE)
            children.append(child)

        return TreeNode(name=name, entry_type=EntryType.DIRECTORY, children=children)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write_file(
        self,
        path: str,
        content: str,
        content_type: str = "text/plain",
    ) -> OperationResult:
        """Write text content to a file in the bucket."""
        file_path = self._resolve_path(path)
        if not file_path:
            return OperationResult(
                success=False,
                path=path,
                error="Cannot write: path resolves to bucket root",
            )

        try:
            body = content.encode("utf-8")
            self._s3.put_object(
                Bucket=self.bucket,
                Key=file_path,
                Body=body,
                ContentType=content_type,
                ContentLength=len(body),
            )
            return OperationResult(
                success=True,
                path=file_path,
                message=f"Written {len(body)} bytes to s3://{self.bucket}/{file_path}",
            )
        except Exception as e:
            logger.error(f"PutObject failed for {file_path!r}: {e}")
            return OperationResult(success=False, path=file_path, error=str(e))

    def delete_file(self, path: str) -> OperationResult:
        """Delete a file from the bucket."""
        file_path = self._resolve_path(path)
        if not file_path:
            return OperationResult(
                success=False,
                path=path,
                error="Cannot delete: path resolves to bucket root",
            )

        try:
            self._s3.delete_object(Bucket=self.bucket, Key=file_path)
            return OperationResult(
                success=True,
                path=file_path,
                message=f"Deleted s3://{self.bucket}/{file_path}",
            )
        except Exception as e:
            logger.error(f"DeleteObject failed for {file_path!r}: {e}")
            return OperationResult(success=False, path=file_path, error=str(e))
