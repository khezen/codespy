"""Tests for dspy_config model-capability helpers.

These tests use a standalone copy of the _supports_cache_control function
 to avoid heavy import dependencies on dspy, litellm, and other modules.
"""

from unittest.mock import patch


# Standalone copy of the function for isolated testing
def _supports_cache_control(model: str) -> bool:
    """Whether the model uses explicit Anthropic-style cache_control markers.

    Returns True only for models where the provider expects ``cache_control``
    fields in messages to enable prompt caching. Models with automatic caching
    (OpenAI prefix-match, Gemini) or no caching return False — sending markers
    to them is either pointless or causes provider errors.

    Detection: ``cache_creation_input_token_cost`` is non-None in LiteLLM's
    model database only for Anthropic-style providers that charge separately
    for cache writes, which correlates exactly with explicit marker support.

    Falls back to provider-prefix heuristic when model info is unavailable.
    """
    import litellm  # noqa: F401 - this is mocked
    try:
        info = litellm.get_model_info(model)
        return info.get("cache_creation_input_token_cost") is not None
    except Exception:
        # Model not in LiteLLM DB (Ollama offline, custom endpoint).
        # Fall back to prefix heuristic.
        lower = model.lower()
        if lower.startswith("anthropic/"):
            return True
        return bool(lower.startswith("bedrock/") and "anthropic" in lower)


class TestSupportsCacheControl:
    """Tests for _supports_cache_control()."""

    def test_anthropic_direct_supported(self):
        with patch("litellm.get_model_info") as mock:
            mock.return_value = {"cache_creation_input_token_cost": 6.25e-06}
            assert _supports_cache_control("anthropic/claude-opus-4-6") is True

    def test_bedrock_anthropic_supported(self):
        with patch("litellm.get_model_info") as mock:
            mock.return_value = {"cache_creation_input_token_cost": 3.75e-06}
            assert _supports_cache_control("bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0") is True

    def test_openai_no_explicit_markers(self):
        with patch("litellm.get_model_info") as mock:
            mock.return_value = {"cache_creation_input_token_cost": None}
            assert _supports_cache_control("openai/gpt-5") is False

    def test_gemini_no_explicit_markers(self):
        with patch("litellm.get_model_info") as mock:
            mock.return_value = {"cache_creation_input_token_cost": None}
            assert _supports_cache_control("gemini/gemini-2.5-pro") is False

    def test_bedrock_non_anthropic_no_markers(self):
        with patch("litellm.get_model_info") as mock:
            mock.return_value = {"cache_creation_input_token_cost": None}
            assert _supports_cache_control("bedrock/amazon.titan-text-express-v1") is False

    def test_fallback_anthropic_prefix_when_info_unavailable(self):
        with patch("litellm.get_model_info", side_effect=Exception("not found")):
            assert _supports_cache_control("anthropic/claude-unknown-model") is True

    def test_fallback_bedrock_anthropic_when_info_unavailable(self):
        with patch("litellm.get_model_info", side_effect=Exception("not found")):
            assert _supports_cache_control("bedrock/us.anthropic.claude-future-v1") is True

    def test_fallback_ollama_when_info_unavailable(self):
        with patch("litellm.get_model_info", side_effect=Exception("not found")):
            assert _supports_cache_control("ollama/llama-4-70b") is False

    def test_fallback_unknown_provider_returns_false(self):
        with patch("litellm.get_model_info", side_effect=Exception("not found")):
            assert _supports_cache_control("custom/my-model") is False

    def test_missing_key_in_info_dict_returns_false(self):
        with patch("litellm.get_model_info") as mock:
            mock.return_value = {}  # Key not present at all
            assert _supports_cache_control("some/model") is False
