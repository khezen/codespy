"""Tests for Settings configuration.

Tests for the Settings class in codespy.config, covering the new
restructured configuration.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codespy.config import Settings


class TestSettingsStructure:
    """Tests for the restructured configuration."""

    def test_llm_config_exists(self):
        """Test that llm config exists and has expected fields."""
        settings = Settings()
        assert hasattr(settings, "llm")
        assert hasattr(settings.llm, "default_model")
        assert hasattr(settings.llm, "retries")
        assert hasattr(settings.llm, "timeout")

    def test_review_config_exists(self):
        """Test that review config exists and has expected fields."""
        settings = Settings()
        assert hasattr(settings, "review")
        assert hasattr(settings.review, "min_confidence")
        assert hasattr(settings.review, "output_format")
        assert hasattr(settings.review, "output_stdout")
        assert hasattr(settings.review, "output_git")
        assert hasattr(settings.review, "cache_dir")

    def test_memory_config_exists(self):
        """Test that memory config exists and has expected fields."""
        settings = Settings()
        assert hasattr(settings, "memory")
        assert hasattr(settings.memory, "enabled")
        assert hasattr(settings.memory, "hippocampus")
        assert hasattr(settings.memory, "cerebral")

    def test_signature_configs_exist(self):
        """Test that signature configs exist in review section."""
        settings = Settings()
        assert hasattr(settings.review, "code_review")
        assert hasattr(settings.review, "doc")
        assert hasattr(settings.review, "scope")
        assert hasattr(settings.review, "supply_chain")
        assert hasattr(settings.review, "summary")
        assert hasattr(settings.review, "audit")

    def test_old_top_level_fields_removed(self):
        """Test that old top-level fields are removed."""
        settings = Settings()
        # These should no longer exist at top level
        old_fields = [
            "default_model",
            "extraction_model",
            "default_max_iters",
            "default_max_llm_calls",
            "default_reasoning_effort",
            "default_temperature",
            "default_max_tokens",
            "llm_retries",
            "llm_timeout",
            "rlm_fallback",
            "signatures",
            "min_confidence",
            "output_format",
            "output_stdout",
            "output_git",
            "cache_dir",
            "excluded_directories",
            "enable_prompt_caching",
            "compact_patches",
        ]
        for field in old_fields:
            assert not hasattr(settings, field), f"Old field '{field}' should not exist"

    def test_env_override_mapping(self, monkeypatch, tmp_path):
        """Test that env vars are mapped correctly to new paths."""
        # Apply isolation: change to temp dir and use monkeypatch for env
        monkeypatch.chdir(tmp_path)

        # DEFAULT_MODEL works (new name, no LLM_ prefix)
        monkeypatch.setenv("DEFAULT_MODEL", "test-model")
        settings = Settings()
        assert settings.llm.default_model == "test-model"
        monkeypatch.delenv("DEFAULT_MODEL")

        # LLM_DEFAULT_MODEL is ignored (old name no longer works)
        monkeypatch.setenv("LLM_DEFAULT_MODEL", "ignored-model")
        settings = Settings()
        # Should use default, not the ignored env var
        assert settings.llm.default_model != "ignored-model"
        monkeypatch.delenv("LLM_DEFAULT_MODEL")

        # LLM_RETRIES still works (exception that keeps LLM_ prefix)
        monkeypatch.setenv("LLM_RETRIES", "7")
        settings = Settings()
        assert settings.llm.retries == 7
        monkeypatch.delenv("LLM_RETRIES")
