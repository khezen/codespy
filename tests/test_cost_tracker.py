"""Tests for the CostTracker module."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from codespy.agents.cost_tracker import (
    CostTracker,
    SignatureContext,
    SignatureStats,
    _as_number,
    _calculate_costs_from_entries,
    _get_history_entries,
    _get_history_uuids,
    get_cost_tracker,
)


class TestSignatureStats:
    """Tests for SignatureStats dataclass."""

    def test_duration_seconds_returns_zero_if_no_start_time(self):
        """Duration should be 0 if start_time is None."""
        stats = SignatureStats(name="test")
        assert stats.duration_seconds == 0.0

    def test_duration_seconds_calculates_from_start_time(self):
        """Duration should calculate from start_time to now."""
        import time

        stats = SignatureStats(name="test", start_time=time.time() - 1.0)
        assert stats.duration_seconds >= 0.9

    def test_duration_seconds_uses_end_time_if_set(self):
        """Duration should use end_time if set."""
        stats = SignatureStats(name="test", start_time=0.0, end_time=5.0)
        assert stats.duration_seconds == 5.0

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all relevant fields."""
        stats = SignatureStats(
            name="test", cost=0.5, tokens=1000, call_count=5, start_time=0.0, end_time=5.0
        )
        result = stats.to_dict()

        assert result["name"] == "test"
        assert result["cost"] == 0.5
        assert result["tokens"] == 1000
        assert result["call_count"] == 5
        assert result["duration_seconds"] == 5.0


class TestAsNumber:
    """Tests for _as_number helper."""

    def test_none_returns_zero(self):
        assert _as_number(None) == 0.0

    def test_bool_returns_zero(self):
        assert _as_number(True) == 0.0
        assert _as_number(False) == 0.0

    def test_int_returns_float(self):
        assert _as_number(42) == 42.0

    def test_float_returns_float(self):
        assert _as_number(3.14) == 3.14

    def test_numeric_string_returns_float(self):
        assert _as_number("10.5") == 10.5

    def test_invalid_string_returns_zero(self):
        assert _as_number("not a number") == 0.0

    def test_list_returns_zero(self):
        assert _as_number([1, 2, 3]) == 0.0


class TestGetHistoryEntries:
    """Tests for _get_history_entries helper."""

    def test_returns_empty_list_when_no_lm(self):
        with patch("dspy.settings.lm", None):
            result = _get_history_entries()
            assert result == []

    def test_returns_empty_list_when_no_history_attr(self):
        mock_lm = MagicMock()
        del mock_lm.history
        with patch("dspy.settings.lm", mock_lm):
            result = _get_history_entries()
            assert result == []

    def test_returns_history_when_available(self):
        mock_history = [{"uuid": "test-uuid", "cost": 0.5}]
        mock_lm = MagicMock()
        mock_lm.history = mock_history
        with patch("dspy.settings.lm", mock_lm):
            result = _get_history_entries()
            assert result == mock_history


class TestGetHistoryUuids:
    """Tests for _get_history_uuids helper."""

    def test_returns_set_of_uuids(self):
        mock_history = [
            {"uuid": "uuid-1", "cost": 0.5},
            {"uuid": "uuid-2", "cost": 0.3},
            {"cost": 0.1},  # No uuid
        ]
        mock_lm = MagicMock()
        mock_lm.history = mock_history
        with patch("dspy.settings.lm", mock_lm):
            result = _get_history_uuids()
            assert result == {"uuid-1", "uuid-2"}


class TestCalculateCostsFromEntries:
    """Tests for _calculate_costs_from_entries helper."""

    def test_calculates_costs_and_tokens(self):
        entries = [
            {
                "uuid": "uuid-1",
                "cost": 0.5,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
            {
                "uuid": "uuid-2",
                "cost": 0.3,
                "usage": {"prompt_tokens": 200, "completion_tokens": 100},
            },
        ]
        exclude = set()

        cost, tokens, calls = _calculate_costs_from_entries(entries, exclude)

        assert cost == 0.8
        assert tokens == 450
        assert calls == 2

    def test_excludes_specified_uuids(self):
        entries = [
            {"uuid": "uuid-1", "cost": 0.5, "usage": {"prompt_tokens": 100}},
            {"uuid": "uuid-2", "cost": 0.3, "usage": {"prompt_tokens": 200}},
        ]
        exclude = {"uuid-1"}

        cost, tokens, calls = _calculate_costs_from_entries(entries, exclude)

        assert cost == 0.3
        assert tokens == 200
        assert calls == 1

    def test_handles_non_dict_entries(self):
        entries = ["not a dict", {"uuid": "uuid-1", "cost": 0.5, "usage": {}}]
        exclude = set()

        cost, tokens, calls = _calculate_costs_from_entries(entries, exclude)

        assert cost == 0.5
        assert calls == 1

    def test_handles_missing_usage(self):
        entries = [
            {"uuid": "uuid-1", "cost": 0.5},  # No usage
        ]
        exclude = set()

        cost, tokens, calls = _calculate_costs_from_entries(entries, exclude)

        assert cost == 0.5
        assert tokens == 0
        assert calls == 1


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_reset_clears_all_stats(self):
        tracker = CostTracker()
        tracker.add_external_call("test", 1.0, 100)
        tracker.reset()

        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 0
        assert tracker.call_count == 0

    def test_start_signature_creates_new_entry(self):
        tracker = CostTracker()
        tracker.start_signature("test_sig")

        stats = tracker.get_signature_stats("test_sig")
        assert stats is not None
        assert stats.name == "test_sig"
        assert stats.start_time is not None

    def test_end_signature_updates_stats(self):
        tracker = CostTracker()
        tracker.start_signature("test_sig")
        tracker.end_signature("test_sig", 0.5, 100, 2)

        stats = tracker.get_signature_stats("test_sig")
        assert stats.cost == 0.5
        assert stats.tokens == 100
        assert stats.call_count == 2
        assert stats.end_time is not None

    def test_end_signature_accumulates_multiple_calls(self):
        tracker = CostTracker()
        tracker.start_signature("test_sig")
        tracker.end_signature("test_sig", 0.5, 100, 2)
        tracker.end_signature("test_sig", 0.3, 50, 1)

        stats = tracker.get_signature_stats("test_sig")
        assert stats.cost == 0.8
        assert stats.tokens == 150
        assert stats.call_count == 3

    def test_total_cost_sums_all_signatures(self):
        tracker = CostTracker()
        tracker.add_external_call("sig1", 0.5, 100)
        tracker.add_external_call("sig2", 0.3, 50)

        assert tracker.total_cost == 0.8

    def test_total_tokens_sums_all_signatures(self):
        tracker = CostTracker()
        tracker.add_external_call("sig1", 0.5, 100)
        tracker.add_external_call("sig2", 0.3, 50)

        assert tracker.total_tokens == 150

    def test_call_count_sums_all_signatures(self):
        tracker = CostTracker()
        tracker.add_external_call("sig1", 0.5, 100, 2)
        tracker.add_external_call("sig2", 0.3, 50, 3)

        assert tracker.call_count == 5

    def test_get_signature_stats_returns_none_for_unknown(self):
        tracker = CostTracker()
        result = tracker.get_signature_stats("unknown")
        assert result is None

    def test_get_all_signature_stats_returns_copy(self):
        tracker = CostTracker()
        tracker.add_external_call("test", 0.5, 100)

        stats1 = tracker.get_all_signature_stats()
        tracker.add_external_call("test", 0.3, 50)  # Modify after copy
        stats2 = tracker.get_all_signature_stats()

        assert stats1["test"].cost == 0.5  # Original copy unchanged
        assert stats2["test"].cost == 0.8  # New copy reflects updates


class TestCostTrackerAddExternalCall:
    """Tests for CostTracker.add_external_call method."""

    def test_add_external_call_creates_new_entry(self):
        tracker = CostTracker()
        tracker.add_external_call("cerebral_retain", 0.5, 100, 1)

        stats = tracker.get_signature_stats("cerebral_retain")
        assert stats is not None
        assert stats.name == "cerebral_retain"
        assert stats.cost == 0.5
        assert stats.tokens == 100
        assert stats.call_count == 1

    def test_add_external_call_accumulates_existing_entry(self):
        tracker = CostTracker()
        tracker.add_external_call("cerebral_retain", 0.5, 100, 1)
        tracker.add_external_call("cerebral_retain", 0.3, 50, 2)

        stats = tracker.get_signature_stats("cerebral_retain")
        assert stats.cost == 0.8
        assert stats.tokens == 150
        assert stats.call_count == 3

    def test_add_external_call_does_not_touch_start_end_time(self):
        tracker = CostTracker()
        tracker.add_external_call("cerebral_retain", 0.5, 100)

        stats = tracker.get_signature_stats("cerebral_retain")
        assert stats.start_time is None
        assert stats.end_time is None

    def test_add_external_call_default_calls_is_one(self):
        tracker = CostTracker()
        tracker.add_external_call("cerebral_retain", 0.5, 100)

        stats = tracker.get_signature_stats("cerebral_retain")
        assert stats.call_count == 1

    def test_add_external_call_is_thread_safe(self):
        """Concurrent add_external_call calls should all be counted."""
        tracker = CostTracker()
        errors = []

        def add_call(n):
            try:
                tracker.add_external_call("cerebral_retain", 0.1, 10, 1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_call, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = tracker.get_signature_stats("cerebral_retain")
        assert stats.cost == pytest.approx(10.0, rel=0.01)
        assert stats.tokens == 1000
        assert stats.call_count == 100


class TestSignatureContext:
    """Tests for SignatureContext context manager."""

    @pytest.fixture
    def mock_lm_context(self):
        with patch("codespy.agents.dspy_config.lm_context") as mock:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=None)
            mock.return_value = ctx
            yield mock

    def test_enter_applies_lm_and_starts_tracking(self, mock_lm_context):
        tracker = CostTracker()

        with patch("dspy.settings.lm", MagicMock(history=[])):
            with SignatureContext("test_sig", tracker):
                pass

        stats = tracker.get_signature_stats("test_sig")
        assert stats is not None

    def test_exit_calculates_costs(self, mock_lm_context):
        tracker = CostTracker()
        mock_history = [
            {"uuid": "new-uuid", "cost": 0.5, "usage": {"prompt_tokens": 100}},
        ]

        with patch("dspy.settings.lm", MagicMock(history=mock_history)):
            with patch.object(tracker, "end_signature") as mock_end:
                with SignatureContext("test_sig", tracker):
                    pass

                mock_end.assert_called_once()
                args = mock_end.call_args
                assert args[0][0] == "test_sig"

    def test_exit_always_releases_lm_context(self, mock_lm_context):
        tracker = CostTracker()
        mock_lm = MagicMock()
        mock_lm.history = []

        with patch("dspy.settings.lm", mock_lm):
            try:
                with SignatureContext("test_sig", tracker):
                    raise ValueError("Test error")
            except ValueError:
                pass

        # lm_context.__exit__ should have been called
        assert mock_lm_context.return_value.__exit__.called


class TestGetCostTracker:
    """Tests for get_cost_tracker function."""

    def test_returns_same_instance(self):
        tracker1 = get_cost_tracker()
        tracker2 = get_cost_tracker()
        assert tracker1 is tracker2

    def test_returns_cost_tracker_instance(self):
        tracker = get_cost_tracker()
        assert isinstance(tracker, CostTracker)
