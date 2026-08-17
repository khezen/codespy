"""Tests for Hippocampus internal methods."""

import pytest

from codespy.agents.memory.hippocampus import (
    ContextMemory,
    Hippocampus,
    Item,
    ItemTag,
    Operation,
    OpType,
)


class TestRecordMutations:
    """Unit tests for _record_mutations back-fill logic."""

    @staticmethod
    def _make_hip(topic_ids=None, distill_step=0):
        """Create minimal Hippocampus bypassing __init__ for unit testing."""
        hip = object.__new__(Hippocampus)
        hip._distill_step = distill_step
        hip._topic_ids = topic_ids or ["t1"]
        return hip

    def test_add_backfill_with_mixed_ops(self):
        """ADD item_ids back-filled correctly when DELETEs precede them."""
        pre = ContextMemory(
            context_understanding=[Item(id="cu-existing", content="Old", topic_ids=["t1"])],
        )
        ops = [
            Operation(type=OpType.DELETE, item_id="cu-existing"),
            Operation(type=OpType.ADD, section="context_understanding", content="New 1"),
            Operation(type=OpType.ADD, section="domain_constants", content="New 2"),
        ]
        new_ids = ["cu-aaa", "dc-bbb"]

        hip = self._make_hip()
        mutations = hip._record_mutations(ops, new_ids, pre)

        assert len(mutations) == 3
        assert mutations[0].type == OpType.DELETE
        assert mutations[0].item_id == "cu-existing"
        assert mutations[1].type == OpType.ADD
        assert mutations[1].item_id == "cu-aaa"
        assert mutations[2].type == OpType.ADD
        assert mutations[2].item_id == "dc-bbb"

    def test_add_backfill_skipped_delete(self):
        """ADD correct even when DELETE target not found (no mutation emitted)."""
        pre = ContextMemory()  # empty — DELETE won't find anything
        ops = [
            Operation(type=OpType.DELETE, item_id="cu-ghost"),
            Operation(type=OpType.ADD, section="context_understanding", content="New"),
        ]
        new_ids = ["cu-xyz"]

        hip = self._make_hip()
        mutations = hip._record_mutations(ops, new_ids, pre)

        assert len(mutations) == 1
        assert mutations[0].type == OpType.ADD
        assert mutations[0].item_id == "cu-xyz"

    def test_add_backfill_all_adds(self):
        """All-ADD batch back-fills in order."""
        pre = ContextMemory()
        ops = [
            Operation(type=OpType.ADD, section="context_understanding", content="A"),
            Operation(type=OpType.ADD, section="domain_constants", content="B"),
            Operation(type=OpType.ADD, section="reusable_results", content="C"),
        ]
        new_ids = ["cu-1", "dc-2", "rr-3"]

        hip = self._make_hip()
        mutations = hip._record_mutations(ops, new_ids, pre)

        assert [m.item_id for m in mutations] == ["cu-1", "dc-2", "rr-3"]
        assert all(m.type == OpType.ADD for m in mutations)

    def test_add_backfill_length_mismatch_raises(self):
        """zip(strict=True) raises ValueError on length mismatch."""
        pre = ContextMemory()
        ops = [
            Operation(type=OpType.ADD, section="context_understanding", content="A"),
        ]
        new_ids = ["cu-1", "cu-2"]  # too many IDs

        hip = self._make_hip()
        with pytest.raises(ValueError):
            hip._record_mutations(ops, new_ids, pre)


class TestUpdateItemScores:
    """Unit tests for _update_item_scores scoring logic."""

    @staticmethod
    def _make_hip():
        """Create minimal Hippocampus with empty scores."""
        hip = object.__new__(Hippocampus)
        hip.scores = {}
        return hip

    def test_helpful_increments(self):
        hip = self._make_hip()
        hip._update_item_scores({"a": ItemTag.HELPFUL})
        assert hip.scores == {"a": 1}

    def test_helpful_accumulates(self):
        hip = self._make_hip()
        hip.scores = {"a": 3}
        hip._update_item_scores({"a": ItemTag.HELPFUL})
        assert hip.scores == {"a": 4}

    def test_harmful_decrements(self):
        hip = self._make_hip()
        hip.scores = {"a": 2}
        hip._update_item_scores({"a": ItemTag.HARMFUL})
        assert hip.scores == {"a": 1}

    def test_stale_decrements(self):
        hip = self._make_hip()
        hip._update_item_scores({"a": ItemTag.STALE})
        assert hip.scores == {"a": -1}

    def test_neutral_initializes_zero(self):
        hip = self._make_hip()
        hip._update_item_scores({"a": ItemTag.NEUTRAL})
        assert hip.scores == {"a": 0}

    def test_neutral_preserves_existing(self):
        hip = self._make_hip()
        hip.scores = {"a": 5}
        hip._update_item_scores({"a": ItemTag.NEUTRAL})
        assert hip.scores == {"a": 5}
