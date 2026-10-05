"""Tests for DiagnosticTwoStepAdapter."""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import dspy  # type: ignore[import-untyped]
import pytest
from dspy.adapters.base import AdapterParseError  # type: ignore[import-untyped]

from codespy.agents.diagnostic_adapter import (
    DiagnosticTwoStepAdapter,
    _log_empty_response,
    _get_finish_reason,
    _get_finish_reason_from_lm,
    _write_refusal_dump,
)


def _make_empty_response_error():
    """Create an AdapterParseError for empty response."""
    mock_sig = MagicMock()
    mock_sig.__class__.__name__ = "TestSignature"
    return AdapterParseError(
        adapter_name="TwoStepAdapter",
        signature=mock_sig,
        lm_response="",
        message="The LM returned an empty or null response",
    )


def _make_other_error():
    """Create an AdapterParseError for other parsing error."""
    mock_sig = MagicMock()
    mock_sig.__class__.__name__ = "TestSignature"
    return AdapterParseError(
        adapter_name="TwoStepAdapter",
        signature=mock_sig,
        lm_response="{invalid json",
        message="Some other parsing error",
    )


class TestDiagnosticTwoStepAdapter:
    """Tests for DiagnosticTwoStepAdapter."""

    def test_empty_response_error_logs_diagnostics(self, caplog):
        """AdapterParseError with empty response message triggers diagnostic logging."""
        # Use actual dspy.LM or properly mock the BaseLM class
        extraction_lm = MagicMock(spec=dspy.BaseLM)
        adapter = DiagnosticTwoStepAdapter(extraction_lm)

        # Create a mock LM with history
        mock_lm = MagicMock()
        mock_lm.model = "test-model"
        mock_lm.kwargs = {"max_tokens": 4096, "reasoning_effort": "medium"}
        mock_lm.history = [
            {
                "messages": [{"role": "user", "content": "test"}],
                "response": {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "content": "",
                                "reasoning_content": "some reasoning" * 10,
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 0,
                        "reasoning_tokens": 50,
                    },
                },
            }
        ]

        with caplog.at_level(logging.WARNING, logger="codespy.agents.diagnostic_adapter"):
            with patch.object(
                DiagnosticTwoStepAdapter.__bases__[0],  # TwoStepAdapter
                "__call__",
                side_effect=_make_empty_response_error(),
            ):
                with pytest.raises(AdapterParseError) as exc_info:
                    adapter(mock_lm, {}, MagicMock(), [], {})

        assert "empty or null response" in str(exc_info.value)
        assert "Empty LM response" in caplog.text
        assert "finish_reason=length" in caplog.text

    def test_non_empty_parse_error_does_not_log(self, caplog):
        """Other AdapterParseError messages do not trigger diagnostic logging."""
        extraction_lm = MagicMock(spec=dspy.BaseLM)
        adapter = DiagnosticTwoStepAdapter(extraction_lm)

        mock_lm = MagicMock()

        with caplog.at_level(logging.WARNING, logger="codespy.agents.diagnostic_adapter"):
            with patch.object(
                DiagnosticTwoStepAdapter.__bases__[0],
                "__call__",
                side_effect=_make_other_error(),
            ):
                with pytest.raises(AdapterParseError) as exc_info:
                    adapter(mock_lm, {}, MagicMock(), [], {})

        assert "Some other parsing error" in str(exc_info.value)
        # Should not have empty response diagnostics
        assert "finish_reason=" not in caplog.text

    def test_empty_history_still_raises(self, caplog):
        """Empty LM history still propagates the original exception."""
        extraction_lm = MagicMock(spec=dspy.BaseLM)
        adapter = DiagnosticTwoStepAdapter(extraction_lm)

        mock_lm = MagicMock()
        mock_lm.history = []

        with caplog.at_level(logging.WARNING, logger="codespy.agents.diagnostic_adapter"):
            with patch.object(
                DiagnosticTwoStepAdapter.__bases__[0],
                "__call__",
                side_effect=_make_empty_response_error(),
            ):
                with pytest.raises(AdapterParseError) as exc_info:
                    adapter(mock_lm, {}, MagicMock(), [], {})

        assert "empty or null response" in str(exc_info.value)
        assert "Empty LM response" in caplog.text
        assert "no LM history available" in caplog.text

    def test_malformed_history_still_raises(self, caplog):
        """Malformed history entry still propagates the original exception."""
        extraction_lm = MagicMock(spec=dspy.BaseLM)
        adapter = DiagnosticTwoStepAdapter(extraction_lm)

        mock_lm = MagicMock()
        mock_lm.history = ["not a dict"]

        with caplog.at_level(logging.WARNING, logger="codespy.agents.diagnostic_adapter"):
            with patch.object(
                DiagnosticTwoStepAdapter.__bases__[0],
                "__call__",
                side_effect=_make_empty_response_error(),
            ):
                with pytest.raises(AdapterParseError) as exc_info:
                    adapter(mock_lm, {}, MagicMock(), [], {})

        assert "empty or null response" in str(exc_info.value)
        assert "Empty LM response" in caplog.text

    def test_lm15_response_object(self, caplog):
        """Handle lm15 Response object format."""
        extraction_lm = MagicMock(spec=dspy.BaseLM)
        adapter = DiagnosticTwoStepAdapter(extraction_lm)

        # Create lm15-style Response mock
        mock_response = MagicMock()
        mock_response.__class__.__name__ = "Response"
        mock_response.finish_reason = "stop"
        mock_response.message.parts = [MagicMock(), MagicMock()]
        mock_response.usage = {"prompt_tokens": 50}

        mock_lm = MagicMock()
        mock_lm.model = "test-model"
        mock_lm.history = [
            {
                "messages": [{"role": "user", "content": "test"}],
                "response": mock_response,
            }
        ]

        with caplog.at_level(logging.WARNING, logger="codespy.agents.diagnostic_adapter"):
            with patch.object(
                DiagnosticTwoStepAdapter.__bases__[0],
                "__call__",
                side_effect=_make_empty_response_error(),
            ):
                with pytest.raises(AdapterParseError) as exc_info:
                    adapter(mock_lm, {}, MagicMock(), [], {})

        assert "empty or null response" in str(exc_info.value)
        assert "lm15 Response" in caplog.text
        assert "finish_reason=stop" in caplog.text

    def test_no_reasoning_content_logs_no_reasoning(self, caplog):
        """When no reasoning content is present, log indicates that."""
        extraction_lm = MagicMock(spec=dspy.BaseLM)
        adapter = DiagnosticTwoStepAdapter(extraction_lm)

        mock_lm = MagicMock()
        mock_lm.model = "claude-opus"
        mock_lm.history = [
            {
                "messages": [{"role": "user", "content": "test"}],
                "response": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 0,
                    },
                },
            }
        ]

        with caplog.at_level(logging.WARNING, logger="codespy.agents.diagnostic_adapter"):
            with patch.object(
                DiagnosticTwoStepAdapter.__bases__[0],
                "__call__",
                side_effect=_make_empty_response_error(),
            ):
                with pytest.raises(AdapterParseError) as exc_info:
                    adapter(mock_lm, {}, MagicMock(), [], {})

        assert "empty or null response" in str(exc_info.value)
        assert "no reasoning_content or thinking_blocks" in caplog.text
        assert "no tool_calls" in caplog.text


class TestLogEmptyResponse:
    """Direct tests for _log_empty_response function."""

    def test_handles_no_lm(self, caplog):
        """Function handles None LM gracefully."""
        with caplog.at_level(logging.WARNING, logger="codespy.agents.diagnostic_adapter"):
            _log_empty_response(None, None)

        # Should log that no history available
        assert "Empty LM response" in caplog.text

    def test_handles_history_exception(self, caplog):
        """Function handles exceptions accessing history gracefully."""
        mock_lm = MagicMock()
        # Make history access raise
        type(mock_lm).history = property(lambda self: (_ for _ in ()).throw(ValueError("boom")))

        with caplog.at_level(logging.WARNING, logger="codespy.agents.diagnostic_adapter"):
            _log_empty_response(mock_lm, None)

        # Should log that diagnostics failed
        # (but the try/except in _log_empty_response catches it)

    def test_signature_with_output_fields(self, caplog):
        """Extract output fields from signature."""
        mock_lm = MagicMock()
        mock_lm.model = "test-model"
        mock_lm.history = []

        mock_sig = MagicMock()
        mock_sig.output_fields = ["field1", "field2"]

        with caplog.at_level(logging.WARNING, logger="codespy.agents.diagnostic_adapter"):
            _log_empty_response(mock_lm, mock_sig)

        assert "output_fields=['field1', 'field2']" in caplog.text


class TestGetFinishReason:
    """Tests for _get_finish_reason function."""

    def test_dict_style_response(self):
        """Extract finish_reason from dict-style response."""
        entry = {
            "response": {
                "choices": [
                    {"finish_reason": "content_filter", "message": {"content": ""}}
                ]
            }
        }
        assert _get_finish_reason(entry) == "content_filter"

    def test_lm15_response_object(self):
        """Extract finish_reason from lm15 Response object."""
        mock_resp = MagicMock()
        mock_resp.__class__.__name__ = "Response"
        mock_resp.finish_reason = "stop"

        entry = {"response": mock_resp}
        assert _get_finish_reason(entry) == "stop"

    def test_no_response(self):
        """Return 'unknown' when no response."""
        entry = {}
        assert _get_finish_reason(entry) == "unknown"

    def test_no_choices(self):
        """Return 'unknown' when no choices."""
        entry = {"response": {"choices": []}}
        assert _get_finish_reason(entry) == "unknown"


class TestGetFinishReasonFromLm:
    """Tests for _get_finish_reason_from_lm function."""

    def test_extracts_from_history(self):
        """Extract finish_reason from LM history."""
        mock_lm = MagicMock()
        mock_lm.history = [
            {
                "response": {
                    "choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]
                }
            }
        ]
        assert _get_finish_reason_from_lm(mock_lm) == "content_filter"

    def test_no_history(self):
        """Return None when no history."""
        mock_lm = MagicMock()
        mock_lm.history = None
        assert _get_finish_reason_from_lm(mock_lm) is None

    def test_empty_history(self):
        """Return None when empty history."""
        mock_lm = MagicMock()
        mock_lm.history = []
        assert _get_finish_reason_from_lm(mock_lm) is None


class TestWriteRefusalDump:
    """Tests for _write_refusal_dump function."""

    @patch("codespy.agents.diagnostic_adapter.get_settings")
    def test_writes_dump_file(self, mock_get_settings, tmp_path):
        """Write refusal dump to file."""
        # Setup mock settings
        mock_settings = MagicMock()
        mock_settings.review.cache_dir = str(tmp_path)
        mock_get_settings.return_value = mock_settings

        entry = {
            "messages": [{"role": "user", "content": "test"}],
            "response": {"choices": [{"finish_reason": "content_filter"}]},
        }
        lm_kwargs = {"max_tokens": 4096, "api_key": "secret123"}

        result = _write_refusal_dump(entry, "claude-3-opus", lm_kwargs, "TestSignature")

        assert result is not None
        assert result.exists()
        assert "refusals" in str(result)

        # Verify content
        with open(result) as f:
            dump = json.load(f)
        assert dump["model"] == "claude-3-opus"
        assert dump["lm_kwargs"]["max_tokens"] == 4096
        assert dump["lm_kwargs"]["api_key"] == "<redacted>"
        assert dump["messages"] == [{"role": "user", "content": "test"}]

    @patch("codespy.agents.diagnostic_adapter.get_settings")
    def test_handles_write_failure(self, mock_get_settings):
        """Gracefully handle write failure."""
        # Setup mock settings that will cause failure
        mock_settings = MagicMock()
        mock_settings.review.cache_dir = "/nonexistent/path/that/cannot/be/created"
        mock_get_settings.return_value = mock_settings

        entry = {"messages": [], "response": {}}
        result = _write_refusal_dump(entry, "model", {}, "Sig")

        assert result is None


class TestContentFilterHandling:
    """Tests for content_filter finish_reason handling."""

    @patch("codespy.agents.diagnostic_adapter.get_settings")
    def test_content_filter_raises_descriptive_error(self, mock_get_settings, caplog, tmp_path):
        """When finish_reason is content_filter, raise descriptive error."""
        # Setup mock settings to use tmp_path for cache_dir
        mock_settings = MagicMock()
        mock_settings.review.cache_dir = str(tmp_path)
        mock_get_settings.return_value = mock_settings

        extraction_lm = MagicMock(spec=dspy.BaseLM)
        adapter = DiagnosticTwoStepAdapter(extraction_lm)

        mock_lm = MagicMock()
        mock_lm.model = "claude-3-opus"
        mock_lm.kwargs = {}
        mock_lm.history = [
            {
                "messages": [{"role": "user", "content": "test"}],
                "response": {
                    "choices": [
                        {
                            "finish_reason": "content_filter",
                            "message": {"content": ""},
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 8},
                },
            }
        ]

        with caplog.at_level(logging.WARNING, logger="codespy.agents.diagnostic_adapter"):
            with patch.object(
                DiagnosticTwoStepAdapter.__bases__[0],
                "__call__",
                side_effect=AdapterParseError(
                    adapter_name="TwoStepAdapter",
                    signature=MagicMock(),
                    lm_response="",
                    message="The LM returned an empty or null response",
                ),
            ):
                with pytest.raises(AdapterParseError) as exc_info:
                    adapter(mock_lm, {}, MagicMock(), [], {})

        assert "model refused (content_filter)" in str(exc_info.value)
        assert "Empty LM response" in caplog.text

    def test_non_content_filter_re_raises_original(self):
        """When finish_reason is not content_filter, re-raise original error."""
        extraction_lm = MagicMock(spec=dspy.BaseLM)
        adapter = DiagnosticTwoStepAdapter(extraction_lm)

        mock_lm = MagicMock()
        mock_lm.model = "claude-3-opus"
        mock_lm.kwargs = {}
        mock_lm.history = [
            {
                "messages": [{"role": "user", "content": "test"}],
                "response": {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": ""},
                        }
                    ],
                },
            }
        ]

        original_error = AdapterParseError(
            adapter_name="TwoStepAdapter",
            signature=MagicMock(),
            lm_response="",
            message="The LM returned an empty or null response",
        )

        with patch.object(
            DiagnosticTwoStepAdapter.__bases__[0],
            "__call__",
            side_effect=original_error,
        ):
            with pytest.raises(AdapterParseError) as exc_info:
                adapter(mock_lm, {}, MagicMock(), [], {})

        assert "empty or null response" in str(exc_info.value)


class TestAcallHandling:
    """Tests for async acall handling."""

    @patch("codespy.agents.diagnostic_adapter.get_settings")
    def test_acall_content_filter_raises_descriptive_error(self, mock_get_settings, caplog, tmp_path):
        """Async acall raises descriptive error for content_filter."""
        import asyncio

        # Setup mock settings to use tmp_path for cache_dir
        mock_settings = MagicMock()
        mock_settings.review.cache_dir = str(tmp_path)
        mock_get_settings.return_value = mock_settings

        extraction_lm = MagicMock(spec=dspy.BaseLM)
        adapter = DiagnosticTwoStepAdapter(extraction_lm)

        mock_lm = MagicMock()
        mock_lm.model = "claude-3-opus"
        mock_lm.kwargs = {}
        mock_lm.history = [
            {
                "messages": [{"role": "user", "content": "test"}],
                "response": {
                    "choices": [
                        {
                            "finish_reason": "content_filter",
                            "message": {"content": ""},
                        }
                    ],
                },
            }
        ]

        async def run_test():
            with caplog.at_level(logging.WARNING, logger="codespy.agents.diagnostic_adapter"):
                with patch.object(
                    DiagnosticTwoStepAdapter.__bases__[0],
                    "acall",
                    side_effect=AdapterParseError(
                        adapter_name="TwoStepAdapter",
                        signature=MagicMock(),
                        lm_response="",
                        message="The LM returned an empty or null response",
                    ),
                ):
                    with pytest.raises(AdapterParseError) as exc_info:
                        await adapter.acall(mock_lm, {}, MagicMock(), [], {})
            return exc_info

        exc_info = asyncio.run(run_test())
        assert "model refused (content_filter)" in str(exc_info.value)
        assert "Empty LM response" in caplog.text

