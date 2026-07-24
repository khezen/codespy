"""S3 client for filesystem-like operations over a single bucket."""

import logging

from codespy.tools.aws.s3.models import (
    EntryType,
    OperationResult,
    S3Content,
    S3Entry,
    S3Info,
    S3Listing,
    S3TreeNode,
)

logger = logging.getLogger(__name__)


class S3Client:
    """Client for S3 operations rooted at a single bucket.

    Treats S3 key prefixes as directories and individual object keys as files,
    mirroring the FileSystem client interface.  All path arguments are relative
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
        """Normalise a relative path, guarding against escaping the bucket root.

        Mirrors FileSystem._resolve_path(): strips leading slashes, collapses
        redundant separators, rejects '..' traversal.

        Args:
            path: User-supplied file path or directory prefix.

        Returns:
            Normalised S3 key string (no leading slash).

        Raises:
            ValueError: If path contains '..' components that escape the root.
        """
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
        """Return the last component of a path (filename or directory name).

        Args:
            path: S3 key or prefix.

        Returns:
            Last path component.
        """
        return path.rstrip("/").rsplit("/", 1)[-1]

    def _file_extension(self, path: str) -> str:
        """Return the file extension from a path (without leading dot).

        Args:
            path: S3 object key.

        Returns:
            Extension string, e.g. 'py', 'json', or '' if none.
        """
        name = self._file_name(path)
        if "." in name:
            return name.rsplit(".", 1)[-1]
        return ""

    def _client_error_code(self, exc: Exception) -> str:
        """Safely extract the error code from a botocore ClientError."""
        try:
            return exc.response["Error"]["Code"]  # type: ignore[attr-defined]
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def exists(self, path: str = "") -> bool:
        """Check whether a file or directory exists in the bucket.

        For files: uses HeadObject.
        For directories: lists with the prefix — exists if any entries found.

        Args:
            path: Relative file path or directory prefix to check.

        Returns:
            True if the path exists.
        """
        file_path = self._resolve_path(path)

        if not file_path:
            # Bucket root always exists
            return True

        # Try as an exact file first
        try:
            self._s3.head_object(Bucket=self.bucket, Key=file_path)
            return True
        except Exception as e:
            if self._client_error_code(e) not in ("404", "NoSuchKey"):
                logger.warning(f"HeadObject error for {file_path!r}: {e}")

        # Try as a directory prefix
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

    def get_info(self, path: str = "") -> S3Info:
        """Get metadata about a file or directory.

        Args:
            path: Relative file path or directory prefix.

        Returns:
            S3Info with metadata.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        file_path = self._resolve_path(path)

        if not file_path:
            return S3Info(
                path=".",
                name=self.bucket,
                entry_type=EntryType.DIRECTORY,
            )

        # Try as a file
        try:
            resp = self._s3.head_object(Bucket=self.bucket, Key=file_path)
            return S3Info(
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

        # Try as a directory prefix
        dir_prefix = file_path if file_path.endswith("/") else file_path + "/"
        resp = self._s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=dir_prefix,
            MaxKeys=1,
        )
        if resp.get("Contents") or resp.get("CommonPrefixes"):
            return S3Info(
                path=file_path,
                name=self._file_name(file_path),
                entry_type=EntryType.DIRECTORY,
            )

        raise FileNotFoundError(f"Path not found in bucket {self.bucket!r}: {file_path!r}")

    def list_directory(
        self,
        path: str = "",
        include_hidden: bool = False,
    ) -> S3Listing:
        """List files and subdirectories directly under a path (one level deep).

        Uses S3 Delimiter="/" so sub-prefixes are returned as directories.

        Args:
            path: Relative directory path to list (empty string = bucket root).
            include_hidden: Whether to include entries starting with '.'.

        Returns:
            S3Listing with entries sorted: directories first, then files.
        """
        dir_path = self._resolve_path(path)
        prefix = (dir_path + "/") if dir_path else ""

        entries: list[S3Entry] = []
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

            # CommonPrefixes → sub-directories
            for cp in resp.get("CommonPrefixes", []):
                sub = cp.get("Prefix", "")
                name = sub.rstrip("/").rsplit("/", 1)[-1]
                if not include_hidden and name.startswith("."):
                    continue
                entries.append(S3Entry(name=name, entry_type=EntryType.DIRECTORY))
                total_directories += 1

            # Contents → files (skip placeholder directory key)
            for obj in resp.get("Contents", []):
                file_key = obj.get("Key", "")
                if file_key == prefix:
                    continue  # zero-byte folder placeholder
                name = file_key.rsplit("/", 1)[-1]
                if not include_hidden and name.startswith("."):
                    continue
                entries.append(
                    S3Entry(
                        name=name,
                        entry_type=EntryType.FILE,
                        size=obj.get("Size", 0),
                    )
                )
                total_files += 1

            if not resp.get("IsTruncated"):
                break
            continuation_token = resp.get("NextContinuationToken")

        # Directories first, then files — same sort as FileSystem.list_directory
        entries.sort(key=lambda e: (e.entry_type == EntryType.FILE, e.name.lower()))

        return S3Listing(
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
    ) -> S3Content:
        """Read a file from S3 as text.

        Mirrors FileSystem.read_file(): utf-8 → latin-1 fallback, byte and line
        truncation, returns error in model rather than raising.

        Args:
            path: Relative file path.
            max_bytes: Maximum bytes to read (default 100 KB).
            max_lines: Maximum lines to read (optional).

        Returns:
            S3Content with file data.
        """
        file_path = self._resolve_path(path)
        if not file_path:
            return S3Content(path=path, error="Cannot read: path is empty (bucket root)")

        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=file_path)
        except Exception as e:
            return S3Content(path=file_path, error=f"GetObject failed: {e}")

        size: int = resp.get("ContentLength", 0)
        content_type: str = resp.get("ContentType", "")
        truncated = False

        try:
            raw: bytes = resp["Body"].read(max_bytes + 1)
        except Exception as e:
            return S3Content(
                path=file_path,
                error=f"Error reading body: {e}",
                size=size,
                content_type=content_type,
            )

        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
            truncated = True

        # Decode: utf-8 first, latin-1 fallback (same as FileSystem.read_file)
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = raw.decode("latin-1")
            except Exception:
                return S3Content(
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

        return S3Content(
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
    ) -> S3TreeNode:
        """Get a tree representation of a directory in the bucket.

        Args:
            path: Relative directory path to start from (empty = bucket root).
            max_depth: Maximum recursion depth.
            include_hidden: Whether to include entries starting with '.'.

        Returns:
            S3TreeNode representing the directory tree.
        """
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
    ) -> S3TreeNode:
        """Recursively build an S3TreeNode.

        Args:
            dir_path: Current directory path (S3 prefix).
            name: Display name for this node.
            max_depth: Maximum recursion depth.
            include_hidden: Include hidden entries.
            current_depth: Current recursion depth counter.

        Returns:
            S3TreeNode for this directory.
        """
        if current_depth >= max_depth:
            return S3TreeNode(name=name, entry_type=EntryType.DIRECTORY)

        listing = self.list_directory(dir_path, include_hidden=include_hidden)
        children: list[S3TreeNode] = []

        for entry in listing.entries:
            if entry.entry_type == EntryType.DIRECTORY:
                child_path = f"{dir_path}/{entry.name}" if dir_path else entry.name
                child = self._build_tree(
                    child_path,
                    entry.name,
                    max_depth,
                    include_hidden,
                    current_depth + 1,
                )
            else:
                child = S3TreeNode(name=entry.name, entry_type=EntryType.FILE)
            children.append(child)

        return S3TreeNode(
            name=name,
            entry_type=EntryType.DIRECTORY,
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
            path: Relative directory path to start from.
            max_depth: Maximum recursion depth.
            include_hidden: Whether to include hidden entries.

        Returns:
            String representation of the tree (same style as FileSystem.get_tree_string).
        """
        tree = self.get_tree(path, max_depth, include_hidden)
        return tree.to_string()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write_file(
        self,
        path: str,
        content: str,
        content_type: str = "text/plain",
    ) -> OperationResult:
        """Write text content to a file in the bucket.

        Args:
            path: Relative file path to write to.
            content: Text content to write (encoded as UTF-8).
            content_type: MIME type for the object (default 'text/plain').

        Returns:
            OperationResult indicating success or failure.
        """
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
        """Delete a file from the bucket.

        Args:
            path: Relative file path to delete.

        Returns:
            OperationResult indicating success or failure.
        """
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
