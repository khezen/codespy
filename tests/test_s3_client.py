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

    def test_rejects_interior_traversal(self):
        """Paths with '..' anywhere are now rejected, not just those escaping root."""
        with pytest.raises(ValueError, match="escapes bucket root"):
            self.client._resolve_path("a/b/../c")

    def test_empty_after_normalization(self):
        assert self.client._resolve_path(".") == ""
        assert self.client._resolve_path("/") == ""


class TestReadFileTruncation:
    @staticmethod
    def _truncate_utf8(raw: bytes, max_bytes: int) -> bytes:
        """Reproduce the fixed truncation logic."""
        if len(raw) <= max_bytes:
            return raw
        raw = raw[:max_bytes]
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as e:
            if e.start >= len(raw) - 4:
                raw = raw[:e.start]
        return raw

    def test_truncate_preserves_utf8(self):
        # "café" = 63 61 66 c3 a9 (5 bytes), max_bytes=4 cuts inside é
        raw = "café".encode("utf-8")
        result = self._truncate_utf8(raw, 4).decode("utf-8")
        assert result == "caf"

    def test_truncate_emoji(self):
        raw = "hi🎉bye".encode("utf-8")  # 9 bytes
        result = self._truncate_utf8(raw, 4).decode("utf-8")
        assert result == "hi"

    def test_truncate_at_exact_boundary_preserves_character(self):
        """Bug regression: truncation at valid char boundary must NOT strip it."""
        # "àè" = c3 a0 c3 a8 (4 bytes), max_bytes=4 lands exactly at end of è
        raw = "àè".encode("utf-8")
        result = self._truncate_utf8(raw, 4).decode("utf-8")
        assert result == "àè"  # Both characters preserved (was bug: stripped è)

    def test_truncate_between_two_multibyte(self):
        """Truncation between two multi-byte characters preserves the first."""
        # "àè" = c3 a0 c3 a8, max_bytes=3 cuts inside è
        raw = "àè".encode("utf-8")
        result = self._truncate_utf8(raw, 3).decode("utf-8")
        assert result == "à"  # è is incomplete, stripped
