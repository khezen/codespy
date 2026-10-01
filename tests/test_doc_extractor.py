"""Tests for the documentation extractor module."""

import tempfile
from pathlib import Path

import pytest

from codespy.agents.review.doc.doc_extractor import _trim_changelog, extract_documentation


class TestTrimChangelog:
    """Tests for _trim_changelog function."""

    def test_keeps_unreleased_plus_two_releases(self):
        """Trim keeps [Unreleased] + two release sections."""
        content = """# Changelog

## [Unreleased]

### Added
- New feature

## [1.0.2] - 2024-01-15

### Fixed
- Bug fix

## [1.0.1] - 2024-01-10

### Fixed
- Another bug

## [1.0.0] - 2024-01-01

### Added
- Initial release
"""
        result = _trim_changelog(content)
        assert "[Unreleased]" in result
        assert "[1.0.2]" in result
        assert "[1.0.1]" in result
        assert "[1.0.0]" not in result
        assert "[older entries omitted]" in result

    def test_no_trim_with_two_headings(self):
        """No trim when fewer than 3 headings."""
        content = """# Changelog

## [1.0.0] - 2024-01-01

### Added
- Initial release
"""
        result = _trim_changelog(content)
        assert result == content
        assert "[older entries omitted]" not in result

    def test_no_trim_with_one_heading(self):
        """No trim with only 1 heading."""
        content = """# Changelog

## [Unreleased]

### Added
- Feature
"""
        result = _trim_changelog(content)
        assert result == content

    def test_no_trim_with_no_headings(self):
        """No trim with no headings."""
        content = """# Changelog

Some text without proper headings.
"""
        result = _trim_changelog(content)
        assert result == content

    def test_exactly_three_headings(self):
        """Trim at exactly 3 headings."""
        content = """# Changelog

## [Unreleased]

### Added
- New

## [1.0.1]

### Fixed
- Fix

## [1.0.0]

### Added
- Initial
"""
        result = _trim_changelog(content)
        assert "[Unreleased]" in result
        assert "[1.0.1]" in result
        assert "[1.0.0]" not in result
        assert result.endswith("\n[older entries omitted]")

    def test_preserves_preamble(self):
        """Preamble before first heading is preserved."""
        content = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog].

## [Unreleased]

### Added
- Feature

## [1.0.0]

### Added
- Initial

## [0.9.0]

### Added
- Beta
"""
        result = _trim_changelog(content)
        assert "# Changelog" in result
        assert "All notable changes" in result
        assert "Keep a Changelog" in result
        assert "[Unreleased]" in result
        assert "[1.0.0]" in result
        assert "[0.9.0]" not in result


class TestExtractDocumentation:
    """Tests for extract_documentation function."""

    def test_extracts_readme(self):
        """README files are extracted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# README\n\nHello world")

            result = extract_documentation(root)

            assert "=== README.md ===" in result
            assert "# README" in result
            assert "Hello world" in result

    def test_extracts_changelog_and_trims(self):
        """CHANGELOG files are extracted and trimmed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            changelog_content = """# Changelog

## [Unreleased]

### Added
- New

## [1.0.1]

### Fixed
- Fix

## [1.0.0]

### Added
- Initial

## [0.9.0]

### Added
- Old
"""
            (root / "CHANGELOG.md").write_text(changelog_content)

            result = extract_documentation(root)

            assert "=== CHANGELOG.md ===" in result
            assert "[Unreleased]" in result
            assert "[1.0.1]" in result
            assert "[1.0.0]" in result
            assert "[0.9.0]" not in result
            assert "[older entries omitted]" in result

    def test_non_changelog_not_trimmed(self):
        """Non-CHANGELOG files are not trimmed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            content = """# README

## Section 1
Text 1

## Section 2
Text 2

## Section 3
Text 3
"""
            (root / "README.md").write_text(content)

            result = extract_documentation(root)

            # README should NOT be trimmed - all sections preserved
            assert "Section 1" in result
            assert "Section 2" in result
            assert "Section 3" in result
            assert "[older entries omitted]" not in result

    def test_changelog_case_insensitive(self):
        """CHANGELOG matching is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            changelog_content = """# Changelog

## [Unreleased]
### Added
- New

## [1.0.0]
### Added
- Initial

## [0.9.0]
### Added
- Old
"""
            (root / "changelog").write_text(changelog_content)

            result = extract_documentation(root)

            # Should be trimmed because filename matches case-insensitively
            assert "[older entries omitted]" in result
            assert "[0.9.0]" not in result

    def test_header_unchanged_when_trimmed(self):
        """The `=== path ===` header is unchanged when trimming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            changelog_content = """# Changelog

## [Unreleased]

### Added
- New

## [1.0.0]

### Added
- Initial

## [0.9.0]

### Added
- Old
"""
            (root / "CHANGELOG.md").write_text(changelog_content)

            result = extract_documentation(root)

            # Header should be exactly as expected for filename detection
            assert "=== CHANGELOG.md ===" in result

    def test_returns_empty_for_no_docs(self):
        """Returns empty string when no documentation exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "main.py").write_text("print('hello')")

            result = extract_documentation(root)

            assert result == ""

    def test_returns_empty_for_nonexistent_path(self):
        """Returns empty string when scope root does not exist."""
        result = extract_documentation(Path("/nonexistent/path"))
        assert result == ""
