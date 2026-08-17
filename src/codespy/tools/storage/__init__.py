"""Unified storage — local filesystem and S3 backends with a shared interface."""

from codespy.tools.storage.base import Storage
from codespy.tools.storage.filesystem.client import FileSystem
from codespy.tools.storage.models import (
    Content,
    Entry,
    EntryType,
    Info,
    Listing,
    OperationResult,
    TreeNode,
)
from codespy.tools.storage.s3.client import S3Client

__all__ = [
    "Content",
    "Entry",
    "EntryType",
    "FileSystem",
    "Info",
    "Listing",
    "OperationResult",
    "S3Client",
    "Storage",
    "TreeNode",
]
