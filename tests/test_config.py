"""Tests for Settings configuration.

Tests for the Settings class in codespy.config, covering top-level
boolean configuration fields.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codespy.config import Settings


class TestCompactPatchesSetting:
    """Tests for the compact_patches configuration setting."""

    def test_default_is_false(self):
        """Test that compact_patches defaults to False."""
        # Clear any env var that might affect the setting
        env_backup = os.environ.pop("COMPACT_PATCHES", None)
        try:
            settings = Settings()
            assert settings.compact_patches is False
        finally:
            # Restore env var if it was set
            if env_backup is not None:
                os.environ["COMPACT_PATCHES"] = env_backup

    def test_explicit_true(self):
        """Test that compact_patches can be explicitly set to True."""
        settings = Settings(compact_patches=True)
        assert settings.compact_patches is True

    def test_explicit_false(self):
        """Test that compact_patches can be explicitly set to False."""
        settings = Settings(compact_patches=False)
        assert settings.compact_patches is False

    def test_env_var_true(self, monkeypatch):
        """Test that COMPACT_PATCHES env var sets the value to True."""
        monkeypatch.setenv("COMPACT_PATCHES", "true")
        # Settings reload is needed to pick up env var changes
        from codespy.config import reload_settings
        settings = reload_settings()
        assert settings.compact_patches is True

    def test_env_var_false(self, monkeypatch):
        """Test that COMPACT_PATCHES env var sets the value to False."""
        monkeypatch.setenv("COMPACT_PATCHES", "false")
        from codespy.config import reload_settings
        settings = reload_settings()
        assert settings.compact_patches is False
