"""S3 tool — filesystem-like access to a single S3 bucket."""

from codespy.tools.aws.s3.client import S3Client
from codespy.tools.aws.s3.models import (
    EntryType,
    OperationResult,
    S3Content,
    S3Entry,
    S3Info,
    S3Listing,
    S3TreeNode,
)

__all__ = [
    "S3Client",
    "EntryType",
    "S3Info",
    "S3Entry",
    "S3Listing",
    "S3TreeNode",
    "S3Content",
    "OperationResult",
]
