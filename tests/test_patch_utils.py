"""Tests for patch_utils module."""

from pathlib import Path
from unittest.mock import MagicMock

from codespy.agents.reviewer.models import ScopeResult, ScopeType
from codespy.tools.git.models import ChangedFile, FileStatus
from codespy.tools.git.patch_utils import (
    _expand_hunk_to_functions,
    _merge_hunks,
    _parse_hunks,
    _rebuild_patch,
    _should_compact_file,
    compact_patch,
    compact_patches,
)
from codespy.tools.parsers.treesitter import FunctionInfo


class TestShouldCompactFile:
    """Tests for _should_compact_file function."""

    def test_returns_true_for_code_file_with_patch(self):
        file = ChangedFile(
            filename="src/main.py",
            status=FileStatus.MODIFIED,
            patch="@@ -1,3 +1,3 @@\n-old\n+new",
        )
        assert _should_compact_file(file) is True

    def test_returns_false_for_no_patch(self):
        file = ChangedFile(
            filename="src/main.py",
            status=FileStatus.MODIFIED,
            patch=None,
        )
        assert _should_compact_file(file) is False

    def test_returns_false_for_deleted_file(self):
        file = ChangedFile(
            filename="src/main.py",
            status=FileStatus.REMOVED,
            patch="@@ -1,3 +0,0 @@\n-old",
        )
        assert _should_compact_file(file) is False

    def test_returns_false_for_non_code_file(self):
        file = ChangedFile(
            filename="README.md",
            status=FileStatus.MODIFIED,
            patch="@@ -1,3 +1,3 @@\n-old\n+new",
        )
        assert _should_compact_file(file) is False

    def test_returns_false_for_binary_file(self):
        file = ChangedFile(
            filename="image.png",
            status=FileStatus.ADDED,
            patch=None,
        )
        assert _should_compact_file(file) is False

    def test_returns_false_for_lock_file(self):
        file = ChangedFile(
            filename="package-lock.json",
            status=FileStatus.MODIFIED,
            patch="@@ -1,3 +1,3 @@\n-old\n+new",
        )
        assert _should_compact_file(file) is False


class TestParseHunks:
    """Tests for _parse_hunks function."""

    def test_parses_single_hunk(self):
        patch = """@@ -1,5 +1,5 @@
 def hello():
-    print("old")
+    print("new")
     pass"""
        hunks = _parse_hunks(patch)

        assert len(hunks) == 1
        assert hunks[0]["old_start"] == 1
        assert hunks[0]["old_count"] == 5
        assert hunks[0]["new_start"] == 1
        assert hunks[0]["new_count"] == 5
        assert len(hunks[0]["lines"]) == 4  # Includes the context line

    def test_parses_multiple_hunks(self):
        patch = """@@ -1,3 +1,3 @@
 line1
-line2
+line2_changed
 line3
@@ -10,3 +10,3 @@
 line10
-line11
+line11_changed
 line12"""
        hunks = _parse_hunks(patch)

        assert len(hunks) == 2
        assert hunks[0]["new_start"] == 1
        assert hunks[1]["new_start"] == 10

    def test_handles_hunk_without_old_count(self):
        patch = """@@ -1 +1 @@
-old
+new"""
        hunks = _parse_hunks(patch)

        assert len(hunks) == 1
        assert hunks[0]["old_count"] == 1
        assert hunks[0]["new_count"] == 1

    def test_tracks_changed_new_lines(self):
        patch = """@@ -5,5 +5,6 @@
 context1
-added1
 context2
+added2
 context3"""
        hunks = _parse_hunks(patch)

        # Changed lines in new file: line 7 (added2) - deleted lines aren't in new file
        # Line 5: context1 (line 5), Line 6: deleted (not in new), Line 6: context2 (line 6)
        # Line 7: added2 (line 7)
        assert hunks[0]["changed_new_lines"] == [7]

    def test_returns_empty_list_for_no_hunks(self):
        patch = "no hunk headers here"
        hunks = _parse_hunks(patch)
        assert hunks == []

    def test_handles_no_newline_at_end_marker(self):
        patch = """@@ -1,3 +1,3 @@
 line1
 line2
-line3
+line3_new
\\ No newline at end of file"""
        hunks = _parse_hunks(patch)

        assert len(hunks) == 1
        # The "\ No newline" line is included in hunk lines
        # Lines: line1 (context), -line3 (deleted), +line3_new (added), \ marker
        assert len(hunks[0]["lines"]) == 5


class TestExpandHunkToFunctions:
    """Tests for _expand_hunk_to_functions function."""

    def test_expands_to_enclosing_function(self):
        hunk = {
            "header": "@@ -17,7 +17,7 @@",
            "old_start": 17,
            "old_count": 7,
            "new_start": 17,
            "new_count": 7,
            "lines": ["     line1", "-    old_line", "+    new_line", "     line2"],
            "changed_new_lines": [18],
        }
        # Function from line 10 to 30
        functions = [FunctionInfo(
            name="test_func",
            file="test.py",
            line_start=10,
            line_end=30,
            parameters=[],
        )]
        source_lines = [f"line {i}" for i in range(1, 35)]

        result = _expand_hunk_to_functions(hunk, functions, source_lines)

        assert result is not None
        assert result["expansion_start"] == 10
        assert result["expansion_end"] == 30

    def test_no_expansion_for_change_outside_function(self):
        hunk = {
            "header": "@@ -1,3 +1,4 @@",
            "old_start": 1,
            "old_count": 3,
            "new_start": 1,
            "new_count": 4,
            "lines": ["+import new_module", " import os"],
            "changed_new_lines": [1],
        }
        # Function starts at line 10
        functions = [FunctionInfo(
            name="test_func",
            file="test.py",
            line_start=10,
            line_end=30,
            parameters=[],
        )]
        source_lines = [f"line {i}" for i in range(1, 35)]

        result = _expand_hunk_to_functions(hunk, functions, source_lines)

        # Should return hunk unchanged (no enclosing function)
        assert result == hunk

    def test_uses_innermost_function_for_nested(self):
        hunk = {
            "header": "@@ -15,3 +15,3 @@",
            "old_start": 15,
            "old_count": 3,
            "new_start": 15,
            "new_count": 3,
            "lines": ["-    old", "+    new", "     pass"],
            "changed_new_lines": [15],
        }
        # Outer function: lines 5-25, inner function: lines 12-18
        functions = [
            FunctionInfo(name="outer", file="test.py", line_start=5, line_end=25, parameters=[]),
            FunctionInfo(name="inner", file="test.py", line_start=12, line_end=18, parameters=[]),
        ]
        source_lines = [f"line {i}" for i in range(1, 30)]

        result = _expand_hunk_to_functions(hunk, functions, source_lines)

        # Should use inner function boundaries
        assert result["expansion_start"] == 12
        assert result["expansion_end"] == 18

    def test_expands_multiple_functions_in_one_hunk(self):
        hunk = {
            "header": "@@ -20,10 +20,12 @@",
            "old_start": 20,
            "old_count": 10,
            "new_start": 20,
            "new_count": 12,
            "lines": ["-old", "+new"],
            "changed_new_lines": [22, 35],
        }
        functions = [
            FunctionInfo(name="func1", file="test.py", line_start=15, line_end=25, parameters=[]),
            FunctionInfo(name="func2", file="test.py", line_start=30, line_end=40, parameters=[]),
        ]
        source_lines = [f"line {i}" for i in range(1, 50)]

        result = _expand_hunk_to_functions(hunk, functions, source_lines)

        # Should expand to cover both functions
        assert result["expansion_start"] == 15
        assert result["expansion_end"] == 40


class TestMergeHunks:
    """Tests for _merge_hunks function."""

    def test_returns_single_hunk_unchanged(self):
        hunks = [{"expansion_start": 10, "expansion_end": 30}]
        merged = _merge_hunks(hunks)
        assert merged == hunks

    def test_merges_overlapping_hunks(self):
        hunks = [
            {"expansion_start": 10, "expansion_end": 30, "original_hunk": {"header": "@@ -17,7 +17,7 @@"}},
            {"expansion_start": 25, "expansion_end": 50, "original_hunk": {"header": "@@ -40,5 +40,5 @@"}},
        ]
        merged = _merge_hunks(hunks)

        assert len(merged) == 1
        assert merged[0]["expansion_start"] == 10
        assert merged[0]["expansion_end"] == 50

    def test_merges_adjacent_hunks(self):
        hunks = [
            {"expansion_start": 10, "expansion_end": 20},
            {"expansion_start": 21, "expansion_end": 30},  # Adjacent (gap = 1)
        ]
        merged = _merge_hunks(hunks)

        assert len(merged) == 1
        assert merged[0]["expansion_start"] == 10
        assert merged[0]["expansion_end"] == 30

    def test_keeps_separate_non_overlapping_hunks(self):
        hunks = [
            {"expansion_start": 10, "expansion_end": 20},
            {"expansion_start": 30, "expansion_end": 40},  # Gap of 9 lines
        ]
        merged = _merge_hunks(hunks)

        assert len(merged) == 2

    def test_sorts_hunks_by_start(self):
        hunks = [
            {"expansion_start": 50, "expansion_end": 60},
            {"expansion_start": 10, "expansion_end": 20},
        ]
        merged = _merge_hunks(hunks)

        assert merged[0]["expansion_start"] == 10
        assert merged[1]["expansion_start"] == 50


class TestRebuildPatch:
    """Tests for _rebuild_patch function."""

    def test_rebuilds_single_expanded_hunk(self):
        raw_patch = """@@ -17,7 +17,7 @@ def generate_token(user_id: str, ttl_hours: int = 24) -> str:
     expiry = datetime.utcnow() + timedelta(hours=ttl_hours)
     payload = f"{user_id}:{expiry.isoformat()}"
     signature = hashlib.sha256(f"{payload}{SECRET_KEY}".encode()).hexdigest()
-    return f"{payload}:{signature}"
+    return f"{payload}.{signature}"


 def verify_token(token: str) -> bool:"""

        source_lines = [f"line {i}" for i in range(1, 50)]
        merged_hunks = [
            {
                "original_hunk": {
                    "header": "@@ -17,7 +17,7 @@",
                    "lines": [
                        "     expiry = datetime.utcnow() + timedelta(hours=ttl_hours)",
                        '     payload = f"{user_id}:{expiry.isoformat()}"',
                        '     signature = hashlib.sha256(f"{payload}{SECRET_KEY}".encode()).hexdigest()',
                        '-    return f"{payload}:{signature}"',
                        '+    return f"{payload}.{signature}"',
                        "",
                        "",
                        " def verify_token(token: str) -> bool:",
                    ],
                },
                "expansion_start": 10,
                "expansion_end": 30,
                "hunk_start_new": 17,
                "hunk_end_new": 24,
            }
        ]

        result = _rebuild_patch(raw_patch, merged_hunks, source_lines)

        assert result is not None
        assert "@@ -10," in result  # New header starts at expansion_start

    def test_keeps_header_lines(self):
        raw_patch = """diff --git a/src/main.py b/src/main.py
index 123..456 789
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,3 @@
-old
+new"""

        source_lines = ["line1", "line2", "line3"]
        merged_hunks = [
            {
                "original_hunk": {
                    "header": "@@ -1,3 +1,3 @@",
                    "lines": ["-old", "+new"],
                },
                "expansion_start": 1,
                "expansion_end": 3,
                "hunk_start_new": 1,
                "hunk_end_new": 2,
            }
        ]

        result = _rebuild_patch(raw_patch, merged_hunks, source_lines)

        assert "diff --git" in result
        assert "index 123..456" in result

    def test_returns_none_for_empty_hunks(self):
        result = _rebuild_patch("patch", [], [])
        assert result is None


class TestCompactPatch:
    """Tests for compact_patch function."""

    def test_returns_original_for_empty_patch(self):
        result = compact_patch("", Path("test.py"), Path("/repo"))
        assert result == ""

    def test_returns_original_when_no_functions_found(self, tmp_path: Path):
        # Create a Python file without functions
        source_file = tmp_path / "test.py"
        source_file.write_text("x = 1\ny = 2\n")

        patch = """@@ -1,2 +1,2 @@
-x = 1
+x = 2
 y = 2"""

        mock_parser = MagicMock()
        mock_parser.available = True
        mock_parser.find_function_definitions.return_value = []

        result = compact_patch(patch, Path("test.py"), tmp_path, mock_parser)
        assert result == patch

    def test_returns_original_when_treesitter_unavailable(self, tmp_path: Path):
        patch = """@@ -1,3 +1,3 @@
-old
+new"""

        mock_parser = MagicMock()
        mock_parser.available = False

        result = compact_patch(patch, Path("test.py"), tmp_path, mock_parser)
        assert result == patch

    def test_returns_original_for_non_code_file(self, tmp_path: Path):
        patch = """@@ -1,3 +1,3 @@
-old
+new"""

        result = compact_patch(patch, Path("README.md"), tmp_path)
        assert result == patch


class TestCompactPatches:
    """Tests for compact_patches function."""

    def test_processes_multiple_scopes(self, tmp_path: Path, monkeypatch):
        # Create source files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("x = 1\n")
        (tmp_path / "src" / "utils.py").write_text("y = 2\n")

        scope1 = ScopeResult(
            repo="owner/repo",
            subroot="pkg1",
            scope_type=ScopeType.LIBRARY,
            reason="test",
            changed_files=[
                ChangedFile(
                    filename="src/main.py",
                    status=FileStatus.MODIFIED,
                    patch="@@ -1,3 +1,3 @@\n-old\n+new",
                )
            ],
        )
        scope2 = ScopeResult(
            repo="owner/repo",
            subroot="pkg2",
            scope_type=ScopeType.SERVICE,
            reason="test",
            changed_files=[
                ChangedFile(
                    filename="src/utils.py",
                    status=FileStatus.MODIFIED,
                    patch="@@ -5,3 +5,3 @@\n-old\n+new",
                )
            ],
        )
        scopes = [scope1, scope2]

        # Mock TreeSitterParser to return empty functions
        call_count = 0
        def mock_find_functions(self, file_path):
            nonlocal call_count
            call_count += 1
            return []

        monkeypatch.setattr(
            "codespy.tools.git.patch_utils.TreeSitterParser.find_function_definitions",
            mock_find_functions,
        )
        monkeypatch.setattr(
            "codespy.tools.git.patch_utils.TreeSitterParser.__init__", lambda self, path: None
        )
        monkeypatch.setattr(
            "codespy.tools.git.patch_utils.TreeSitterParser.available", True
        )

        compact_patches(scopes, tmp_path)

        # Should have tried to find functions in both files
        assert call_count == 2

    def test_skips_deleted_files(self, tmp_path: Path, monkeypatch):
        scope = ScopeResult(
            repo="owner/repo",
            subroot=".",
            scope_type=ScopeType.LIBRARY,
            reason="test",
            changed_files=[
                ChangedFile(
                    filename="src/deleted.py",
                    status=FileStatus.REMOVED,
                    patch="@@ -1,3 +0,0 @@\n-line1\n-line2\n-line3",
                )
            ],
        )

        call_count = 0
        def mock_find_functions(self, file_path):
            nonlocal call_count
            call_count += 1
            return []

        monkeypatch.setattr(
            "codespy.tools.git.patch_utils.TreeSitterParser.find_function_definitions",
            mock_find_functions,
        )
        monkeypatch.setattr(
            "codespy.tools.git.patch_utils.TreeSitterParser.__init__", lambda self, path: None
        )
        monkeypatch.setattr(
            "codespy.tools.git.patch_utils.TreeSitterParser.available", True
        )

        compact_patches([scope], tmp_path)

        # Should not try to find functions in deleted files
        assert call_count == 0

    def test_skips_files_without_patches(self, tmp_path: Path, monkeypatch):
        scope = ScopeResult(
            repo="owner/repo",
            subroot=".",
            scope_type=ScopeType.LIBRARY,
            reason="test",
            changed_files=[
                ChangedFile(
                    filename="src/empty.py",
                    status=FileStatus.MODIFIED,
                    patch=None,
                )
            ],
        )

        call_count = 0
        def mock_find_functions(self, file_path):
            nonlocal call_count
            call_count += 1
            return []

        monkeypatch.setattr(
            "codespy.tools.git.patch_utils.TreeSitterParser.find_function_definitions",
            mock_find_functions,
        )
        monkeypatch.setattr(
            "codespy.tools.git.patch_utils.TreeSitterParser.__init__", lambda self, path: None
        )
        monkeypatch.setattr(
            "codespy.tools.git.patch_utils.TreeSitterParser.available", True
        )

        compact_patches([scope], tmp_path)

        assert call_count == 0

    def test_handles_compaction_failure_gracefully(self, tmp_path: Path, monkeypatch):
        # Create source file
        source_file = tmp_path / "test.py"
        source_file.write_text("def func():\n    pass\n")

        scope = ScopeResult(
            repo="owner/repo",
            subroot=".",
            scope_type=ScopeType.LIBRARY,
            reason="test",
            changed_files=[
                ChangedFile(
                    filename="test.py",
                    status=FileStatus.MODIFIED,
                    patch="@@ -1,3 +1,3 @@\n-old\n+new",
                )
            ],
        )

        def mock_find_functions(self, file_path):
            raise Exception("parse error")

        monkeypatch.setattr(
            "codespy.tools.git.patch_utils.TreeSitterParser.find_function_definitions",
            mock_find_functions,
        )
        monkeypatch.setattr(
            "codespy.tools.git.patch_utils.TreeSitterParser.__init__", lambda self, path: None
        )
        monkeypatch.setattr(
            "codespy.tools.git.patch_utils.TreeSitterParser.available", True
        )

        # Should not raise
        compact_patches([scope], tmp_path)

        # Original patch should be preserved
        assert scope.changed_files[0].patch == "@@ -1,3 +1,3 @@\n-old\n+new"
