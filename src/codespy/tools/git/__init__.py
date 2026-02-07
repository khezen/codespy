"""Unified Git tools for GitHub and GitLab integration."""

from codespy.tools.git.base import GitClient
from codespy.tools.git.client import detect_platform, get_client, is_supported_url
from codespy.tools.git.models import (
    CallerInfo,
    ChangedFile,
    FileStatus,
    GitPlatform,
    MergeRequest,
    ReviewContext,
    should_review_file,
)

# Note: GitReporter is not exported here to avoid circular imports.
# Import directly: from codespy.tools.git.reporter import GitReporter

__all__ = [
    "GitClient",
    "get_client",
    "detect_platform",
    "is_supported_url",
    "GitPlatform",
    "MergeRequest",
    "ChangedFile",
    "FileStatus",
    "ReviewContext",
    "CallerInfo",
    "should_review_file",
]
