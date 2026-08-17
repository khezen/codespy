"""Tests for memory storage access verification."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from codespy.config_memory import verify_memory_access
from codespy.tools.storage.filesystem.client import FileSystem
from codespy.tools.storage.s3.client import S3Client


class TestVerifyMemoryAccess:
    """Tests for verify_memory_access function."""

    def test_verify_memory_access_all_disabled(self):
        """When all signatures disabled, returns success with skip message."""
        settings = MagicMock()
        settings.is_signature_enabled.return_value = False
        settings.get_memory_enabled.return_value = False

        success, message = verify_memory_access(settings)

        assert success is True
        assert "Memory disabled" in message

    def test_verify_memory_access_enabled_sig_disabled(self):
        """When signature has memory enabled but signature itself is disabled."""
        settings = MagicMock()

        def is_enabled(sig):
            return False  # All signatures disabled

        def memory_enabled(sig):
            return True  # But memory is configured

        settings.is_signature_enabled.side_effect = is_enabled
        settings.get_memory_enabled.side_effect = memory_enabled

        success, message = verify_memory_access(settings)

        assert success is True
        assert "Memory disabled" in message

    def test_verify_memory_access_store_none(self):
        """When memory active but store is None (missing S3 bucket)."""
        settings = MagicMock()
        settings.memory.backend = "s3"

        def is_enabled(sig):
            return sig == "summary"  # Only summary enabled

        def memory_enabled(sig):
            return sig == "summary"  # Memory enabled for summary

        settings.is_signature_enabled.side_effect = is_enabled
        settings.get_memory_enabled.side_effect = memory_enabled

        with patch("codespy.config_memory.get_memory_store", return_value=None):
            success, message = verify_memory_access(settings)

        assert success is False
        assert "not configured" in message

    def test_verify_memory_access_filesystem_ok(self, tmp_path):
        """When memory active with valid filesystem store."""
        settings = MagicMock()
        settings.memory.backend = "filesystem"

        def is_enabled(sig):
            return sig == "summary"

        def memory_enabled(sig):
            return sig == "summary"

        settings.is_signature_enabled.side_effect = is_enabled
        settings.get_memory_enabled.side_effect = memory_enabled

        # Create a real FileSystem with tmp_path
        fs = FileSystem(tmp_path)

        with patch("codespy.config_memory.get_memory_store", return_value=fs):
            success, message = verify_memory_access(settings)

        assert success is True
        assert "verified" in message
        assert "filesystem" in message

    def test_verify_memory_access_s3_ok(self):
        """When memory active with S3 store that verifies successfully."""
        settings = MagicMock()
        settings.memory.backend = "s3"

        def is_enabled(sig):
            return sig == "summary"

        def memory_enabled(sig):
            return sig == "summary"

        settings.is_signature_enabled.side_effect = is_enabled
        settings.get_memory_enabled.side_effect = memory_enabled

        # Mock S3 client that verifies successfully
        mock_store = MagicMock()
        mock_store.verify_access.return_value = None

        with patch("codespy.config_memory.get_memory_store", return_value=mock_store):
            success, message = verify_memory_access(settings)

        assert success is True
        assert "verified" in message
        assert "s3" in message
        mock_store.verify_access.assert_called_once()

    def test_verify_memory_access_raises(self):
        """When store's verify_access raises an exception."""
        settings = MagicMock()
        settings.memory.backend = "filesystem"

        def is_enabled(sig):
            return sig == "summary"

        def memory_enabled(sig):
            return sig == "summary"

        settings.is_signature_enabled.side_effect = is_enabled
        settings.get_memory_enabled.side_effect = memory_enabled

        # Mock store that raises on verify_access
        mock_store = MagicMock()
        mock_store.verify_access.side_effect = PermissionError("Access denied")

        with patch("codespy.config_memory.get_memory_store", return_value=mock_store):
            success, message = verify_memory_access(settings)

        assert success is False
        assert "not accessible" in message
        assert "Access denied" in message


class TestFileSystemVerifyAccess:
    """Tests for FileSystem.verify_access method."""

    def test_verify_access_filesystem_valid(self, tmp_path):
        """Valid filesystem root - no exception raised."""
        fs = FileSystem(tmp_path)
        # Should not raise
        fs.verify_access()

    def test_verify_access_filesystem_deleted_root(self, tmp_path):
        """Root deleted after initialization - raises FileNotFoundError."""
        fs = FileSystem(tmp_path)
        # Delete the root directory
        import shutil

        shutil.rmtree(tmp_path)

        with pytest.raises(FileNotFoundError, match="does not exist"):
            fs.verify_access()

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod 000 not supported on Windows")
    def test_verify_access_filesystem_not_readable(self, tmp_path):
        """Root not readable - raises PermissionError."""
        import os

        fs = FileSystem(tmp_path)
        # Remove read permission
        os.chmod(tmp_path, 0o000)

        try:
            with pytest.raises(PermissionError, match="not readable"):
                fs.verify_access()
        finally:
            # Restore permission for cleanup
            os.chmod(tmp_path, 0o755)


class TestS3ClientVerifyAccess:
    """Tests for S3Client.verify_access method."""

    def test_verify_access_s3_success(self):
        """Successful S3 access - no exception raised."""
        # Create S3Client without calling __init__ (no boto3)
        client = S3Client.__new__(S3Client)
        client.bucket = "test-bucket"
        client._s3 = MagicMock()
        client._s3.list_objects_v2.return_value = {}

        # Should not raise
        client.verify_access()

        # Verify the call was made correctly
        client._s3.list_objects_v2.assert_called_once_with(Bucket="test-bucket", MaxKeys=1)

    def test_verify_access_s3_client_error(self):
        """S3 client error propagates."""
        from unittest.mock import MagicMock

        # Create S3Client without calling __init__
        client = S3Client.__new__(S3Client)
        client.bucket = "test-bucket"
        client._s3 = MagicMock()

        # Simulate boto3 ClientError
        class ClientError(Exception):
            def __init__(self, error_response, operation_name):
                self.response = error_response
                self.operation_name = operation_name
                super().__init__(str(error_response))

        client._s3.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "The specified bucket does not exist"}},
            "ListObjectsV2",
        )

        with pytest.raises(ClientError):
            client.verify_access()
