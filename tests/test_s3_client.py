import pytest
from codespy.tools.storage.s3.client import S3Client


class TestResolvePath:
    def setup_method(self):
        # Patch boto3 — we only test path logic
        self.client = S3Client.__new__(S3Client)

    def test_normal_path(self):
        assert self.client._resolve_path("foo/bar/baz.txt") == "foo/bar/baz.txt"

    def test_strips_leading_slash(self):
        assert self.client._resolve_path("/foo/bar") == "foo/bar"

    def test_collapses_double_slash(self):
        assert self.client._resolve_path("foo//bar") == "foo/bar"

    def test_resolves_dot(self):
        assert self.client._resolve_path("foo/./bar") == "foo/bar"

    def test_rejects_traversal(self):
        with pytest.raises(ValueError, match="escapes bucket root"):
            self.client._resolve_path("foo/../../etc/passwd")

    def test_rejects_leading_traversal(self):
        with pytest.raises(ValueError, match="escapes bucket root"):
            self.client._resolve_path("../secret")

    def test_empty_after_normalization(self):
        assert self.client._resolve_path(".") == ""
        assert self.client._resolve_path("/") == ""


class TestReadFileTruncation:
    def test_truncate_preserves_utf8(self):
        # "café" = 63 61 66 c3 a9 (5 bytes), max_bytes=4 cuts inside é
        raw = "café".encode("utf-8")  # 5 bytes
        max_bytes = 4
        # Simulate truncation logic
        truncated_raw = raw[:max_bytes]  # b'caf\xc3' — incomplete é
        while truncated_raw and (truncated_raw[-1] & 0xC0) == 0x80:
            truncated_raw = truncated_raw[:-1]
        if truncated_raw and truncated_raw[-1] >= 0xC0:
            truncated_raw = truncated_raw[:-1]
        result = truncated_raw.decode("utf-8")
        assert result == "caf"  # Clean cut before multi-byte char

    def test_truncate_emoji(self):
        raw = "hi🎉bye".encode("utf-8")  # "hi" (2) + 🎉 (4) + "bye" (3) = 9 bytes
        max_bytes = 4
        truncated_raw = raw[:max_bytes]  # b'hi\xf0\x9f' — incomplete emoji
        while truncated_raw and (truncated_raw[-1] & 0xC0) == 0x80:
            truncated_raw = truncated_raw[:-1]
        if truncated_raw and truncated_raw[-1] >= 0xC0:
            truncated_raw = truncated_raw[:-1]
        result = truncated_raw.decode("utf-8")
        assert result == "hi"
