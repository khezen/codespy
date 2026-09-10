"""Tests for Hippocampus with PostgreSQL storage (pg0-embedded)."""

import pytest
from datetime import UTC, datetime

# Skip all tests in this file if pg0-embedded is not installed
try:
    import pg0
    PG0_AVAILABLE = True
except ImportError:
    PG0_AVAILABLE = False

from codespy.agents.memory.hippocampus import (
    ContextMemory,
    Hippocampus,
    Mutation,
    Observation,
    ObservationTag,
    Operation,
    OpType,
    Topic,
)
from codespy.agents.memory.hippocampus.episode import Episode
from codespy.agents.memory.postgres import EpisodeStore


pytestmark = pytest.mark.skipif(
    not PG0_AVAILABLE,
    reason="pg0-embedded not installed (pip install pg0-embedded)"
)


@pytest.fixture(scope="module")
def pg0_uri():
    """Create a pg0-embedded PostgreSQL instance for testing."""
    from codespy.agents.memory.pg0_manager import get_pg0_uri, stop_pg0
    
    uri = get_pg0_uri(name="codespy_test", port=None)
    yield uri
    stop_pg0()


@pytest.fixture
def episode_store(pg0_uri):
    """Create a fresh EpisodeStore for each test."""
    store = EpisodeStore(pg0_uri, bank_id="test_bank")
    yield store
    store.close()


class TestEpisodeStoreBasics:
    """Basic tests for EpisodeStore initialization and schema."""

    def test_episode_store_initializes(self, episode_store):
        """EpisodeStore should initialize and create schema."""
        assert episode_store is not None
        assert episode_store.bank_id == "test_bank"

    def test_verify_access_succeeds(self, episode_store):
        """verify_access should succeed after initialization."""
        # Should not raise
        episode_store.verify_access()


class TestEpisodeStoreSaveLoad:
    """Tests for saving and loading episodes."""

    def test_save_simple_episode(self, episode_store):
        """Should save a simple episode with context memory."""
        # Create a simple episode
        ctx = ContextMemory(
            topics=[Topic(id="test/topic", type="project_scope", description="Test topic")],
            context_understanding=[
                Observation(id="cu-1", content="Test observation", topic_ids=["test/topic"])
            ],
        )
        episode = Episode(
            id=__import__("uuid").uuid4(),
            task="test_task",
            module="TestModule",
            question="Test question",
            context_memory=ctx,
            timestamp=datetime.now(UTC),
            run_id="test-run-123",
            mutations=[],
            artifacts={"test": "artifact"},
        )

        # Save should not raise
        episode_store.save_episode(episode)

    def test_load_context_returns_none_for_no_matching_episode(self, episode_store):
        """load_context should return None when no matching episode exists."""
        result = episode_store.load_context(
            task="nonexistent_task",
            topic_ids=["nonexistent/topic"],
        )
        assert result is None

    def test_save_and_load_episode_roundtrip(self, episode_store):
        """Should save and load an episode successfully."""
        import uuid
        
        topic_id = "owner/repo/pkg"
        
        # Create and save an episode with ADD mutation so observation persists
        ctx = ContextMemory(
            topics=[Topic(id=topic_id, type="project_scope", description="Test package")],
            context_understanding=[
                Observation(id="cu-abc123", content="Test understanding", topic_ids=[topic_id])
            ],
        )
        episode = Episode(
            id=uuid.uuid4(),
            task="code_review",
            module="CodeReviewer",
            question="Review PR #123",
            context_memory=ctx,
            timestamp=datetime.now(UTC),
            run_id="run-456",
            mutations=[
                Mutation(
                    step=0,
                    type=OpType.ADD,
                    observation_id="cu-abc123",
                    section="context_understanding",
                    content="Test understanding",
                    previous_content=None,
                    topic_ids=[topic_id],
                )
            ],
            artifacts={"review": "LGTM"},
        )
        episode_store.save_episode(episode)

        # Load context for the same task and topic
        loaded_ctx = episode_store.load_context(
            task="code_review",
            topic_ids=[topic_id],
        )

        # Should have loaded the context
        assert loaded_ctx is not None
        assert len(loaded_ctx.topics) == 1
        assert loaded_ctx.topics[0].id == topic_id
        assert len(loaded_ctx.context_understanding) == 1
        assert loaded_ctx.context_understanding[0].content == "Test understanding"


class TestEpisodeStoreObservationVersioning:
    """Tests for observation versioning on REPLACE operations."""

    def test_observation_versioning_on_replace(self, episode_store):
        """Observation versions should increment on REPLACE operations."""
        import uuid
        
        topic_id = "owner/repo/versioning-test"
        
        # Episode 1: Add an observation
        ctx1 = ContextMemory(
            topics=[Topic(id=topic_id, type="project_scope", description="Test repo")],
            context_understanding=[
                Observation(id="cu-obs1", content="Original content", topic_ids=[topic_id])
            ],
        )
        episode1 = Episode(
            id=uuid.uuid4(),
            task="code_review",
            module="CodeReviewer",
            question="First review",
            context_memory=ctx1,
            timestamp=datetime.now(UTC),
            run_id="run-1",
            mutations=[
                Mutation(
                    step=0,
                    type=OpType.ADD,
                    observation_id="cu-obs1",
                    section="context_understanding",
                    content="Original content",
                    previous_content=None,
                    topic_ids=[topic_id],
                )
            ],
            artifacts={},
        )
        episode_store.save_episode(episode1)

        # Episode 2: Replace the observation
        ctx2 = ContextMemory(
            topics=[Topic(id=topic_id, type="project_scope", description="Test repo")],
            context_understanding=[
                Observation(id="cu-obs1", content="Updated content", topic_ids=[topic_id])
            ],
        )
        episode2 = Episode(
            id=uuid.uuid4(),
            task="code_review",
            module="CodeReviewer",
            question="Second review",
            context_memory=ctx2,
            timestamp=datetime.now(UTC),
            run_id="run-2",
            mutations=[
                Mutation(
                    step=0,
                    type=OpType.REPLACE,
                    observation_id="cu-obs1",
                    section="context_understanding",
                    content="Updated content",
                    previous_content="Original content",
                    topic_ids=[topic_id],
                )
            ],
            artifacts={},
        )
        episode_store.save_episode(episode2)

        # Load the latest context
        loaded_ctx = episode_store.load_context(
            task="code_review",
            topic_ids=[topic_id],
        )

        # Should have the updated content
        assert loaded_ctx is not None
        assert loaded_ctx.context_understanding[0].content == "Updated content"

    def test_inherited_observations_preserved(self, episode_store):
        """Inherited observations (no mutation) should preserve their content."""
        import uuid
        
        topic_id = "owner/repo/inherited-test"
        
        # Episode 1: Add two observations
        ctx1 = ContextMemory(
            topics=[Topic(id=topic_id, type="project_scope", description="Test repo")],
            context_understanding=[
                Observation(id="cu-obs1", content="Observation 1 content", topic_ids=[topic_id]),
                Observation(id="cu-obs2", content="Observation 2 content", topic_ids=[topic_id]),
            ],
        )
        episode1 = Episode(
            id=uuid.uuid4(),
            task="code_review",
            module="CodeReviewer",
            question="First review",
            context_memory=ctx1,
            timestamp=datetime.now(UTC),
            run_id="run-1",
            mutations=[
                Mutation(
                    step=0,
                    type=OpType.ADD,
                    observation_id="cu-obs1",
                    section="context_understanding",
                    content="Observation 1 content",
                    previous_content=None,
                    topic_ids=[topic_id],
                ),
                Mutation(
                    step=0,
                    type=OpType.ADD,
                    observation_id="cu-obs2",
                    section="context_understanding",
                    content="Observation 2 content",
                    previous_content=None,
                    topic_ids=[topic_id],
                ),
            ],
            artifacts={},
        )
        episode_store.save_episode(episode1)

        # Episode 2: Only replace obs1, obs2 is inherited
        ctx2 = ContextMemory(
            topics=[Topic(id=topic_id, type="project_scope", description="Test repo")],
            context_understanding=[
                Observation(id="cu-obs1", content="Observation 1 updated", topic_ids=[topic_id]),
                Observation(id="cu-obs2", content="Observation 2 content", topic_ids=[topic_id]),  # inherited
            ],
        )
        episode2 = Episode(
            id=uuid.uuid4(),
            task="code_review",
            module="CodeReviewer",
            question="Second review",
            context_memory=ctx2,
            timestamp=datetime.now(UTC),
            run_id="run-2",
            mutations=[
                Mutation(
                    step=0,
                    type=OpType.REPLACE,
                    observation_id="cu-obs1",
                    section="context_understanding",
                    content="Observation 1 updated",
                    previous_content="Observation 1 content",
                    topic_ids=[topic_id],
                ),
            ],
            artifacts={},
        )
        episode_store.save_episode(episode2)

        # Load the latest context
        loaded_ctx = episode_store.load_context(
            task="code_review",
            topic_ids=[topic_id],
        )

        # Both observations should be present with correct content
        assert loaded_ctx is not None
        observations_by_id = {obs.id: obs for obs in loaded_ctx.context_understanding}
        assert observations_by_id["cu-obs1"].content == "Observation 1 updated"
        assert observations_by_id["cu-obs2"].content == "Observation 2 content"


class TestEpisodeStoreWithHippocampus:
    """Integration tests for Hippocampus with EpisodeStore."""

    def test_hippocampus_end_episode_with_store(self, episode_store):
        """Hippocampus.end_episode should work with EpisodeStore."""
        import dspy
        
        # Create a simple mock agent
        class MockAgent(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(result="test")
        
        # Create Hippocampus with the mock agent
        mem = Hippocampus(
            MockAgent(),
            task_name="test_task",
            run_id="test-run",
            topics=[Topic(id="test/topic", type="project_scope", description="Test topic")],
        )
        
        # Make a call
        mem.forward()
        
        # End episode with the store
        mem.end_episode(store=episode_store, artifacts={"test": "value"})
        
        # Episode should be set
        assert mem.episode is not None
        assert mem.episode.task == "test_task"
        assert mem.episode.run_id == "test-run"

    def test_load_context_returns_latest_episode(self, episode_store):
        """load_context should return the context from the latest episode."""
        import uuid
        
        topic_id = "owner/repo/latest-episode-test"
        
        # Save multiple episodes with ADD mutations so observations persist
        for i in range(3):
            ctx = ContextMemory(
                topics=[Topic(id=topic_id, type="project_scope", description="Test repo")],
                context_understanding=[
                    Observation(id=f"cu-obs{i}", content=f"Content {i}", topic_ids=[topic_id])
                ],
            )
            episode = Episode(
                id=uuid.uuid4(),
                task="code_review",
                module="CodeReviewer",
                question=f"Review {i}",
                context_memory=ctx,
                timestamp=datetime.now(UTC),
                run_id=f"run-{i}",
                mutations=[
                    Mutation(
                        step=0,
                        type=OpType.ADD,
                        observation_id=f"cu-obs{i}",
                        section="context_understanding",
                        content=f"Content {i}",
                        previous_content=None,
                        topic_ids=[topic_id],
                    )
                ],
                artifacts={},
            )
            episode_store.save_episode(episode)

        # Load the latest context
        loaded_ctx = episode_store.load_context(
            task="code_review",
            topic_ids=[topic_id],
        )

        # Should have all 3 observations (from ADD mutations)
        assert loaded_ctx is not None
        assert len(loaded_ctx.context_understanding) == 3  # All observations accumulated


class TestEpisodeStoreDeleteTombstone:
    """Tests for DELETE operations (tombstone handling)."""

    def test_delete_creates_tombstone(self, episode_store):
        """DELETE should create a tombstone version of the observation."""
        import uuid
        
        topic_id = "owner/repo/delete-test"
        
        # Episode 1: Add an observation
        ctx1 = ContextMemory(
            topics=[Topic(id=topic_id, type="project_scope", description="Test repo")],
            context_understanding=[
                Observation(id="cu-obs1", content="To be deleted", topic_ids=[topic_id])
            ],
        )
        episode1 = Episode(
            id=uuid.uuid4(),
            task="code_review",
            module="CodeReviewer",
            question="First review",
            context_memory=ctx1,
            timestamp=datetime.now(UTC),
            run_id="run-1",
            mutations=[
                Mutation(
                    step=0,
                    type=OpType.ADD,
                    observation_id="cu-obs1",
                    section="context_understanding",
                    content="To be deleted",
                    previous_content=None,
                    topic_ids=[topic_id],
                )
            ],
            artifacts={},
        )
        episode_store.save_episode(episode1)

        # Episode 2: Delete the observation (observation not in context_memory, but in mutations)
        ctx2 = ContextMemory(
            topics=[Topic(id=topic_id, type="project_scope", description="Test repo")],
            context_understanding=[],  # Empty after delete
        )
        episode2 = Episode(
            id=uuid.uuid4(),
            task="code_review",
            module="CodeReviewer",
            question="Second review",
            context_memory=ctx2,
            timestamp=datetime.now(UTC),
            run_id="run-2",
            mutations=[
                Mutation(
                    step=0,
                    type=OpType.DELETE,
                    observation_id="cu-obs1",
                    section="context_understanding",
                    content=None,  # DELETE has no new content
                    previous_content="To be deleted",
                    topic_ids=[topic_id],
                )
            ],
            artifacts={},
        )
        episode_store.save_episode(episode2)

        # Load the latest context - deleted observation should not appear
        loaded_ctx = episode_store.load_context(
            task="code_review",
            topic_ids=[topic_id],
        )

        # Should have no observations (deleted)
        assert loaded_ctx is not None
        assert len(loaded_ctx.context_understanding) == 0


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
        """ADD observation_ids back-filled correctly when DELETEs precede them."""
        pre = ContextMemory(
            context_understanding=[Observation(id="cu-existing", content="Old", topic_ids=["t1"])],
        )
        ops = [
            Operation(type=OpType.DELETE, observation_id="cu-existing"),
            Operation(type=OpType.ADD, section="context_understanding", content="New 1"),
            Operation(type=OpType.ADD, section="domain_constants", content="New 2"),
        ]
        new_ids = ["cu-aaa", "dc-bbb"]

        hip = self._make_hip()
        mutations = hip._record_mutations(ops, new_ids, pre)

        assert len(mutations) == 3
        assert mutations[0].type == OpType.DELETE
        assert mutations[0].observation_id == "cu-existing"
        assert mutations[1].type == OpType.ADD
        assert mutations[1].observation_id == "cu-aaa"
        assert mutations[2].type == OpType.ADD
        assert mutations[2].observation_id == "dc-bbb"

    def test_add_backfill_skipped_delete(self):
        """ADD correct even when DELETE target not found (no mutation emitted)."""
        pre = ContextMemory()  # empty — DELETE won't find anything
        ops = [
            Operation(type=OpType.DELETE, observation_id="cu-ghost"),
            Operation(type=OpType.ADD, section="context_understanding", content="New"),
        ]
        new_ids = ["cu-xyz"]

        hip = self._make_hip()
        mutations = hip._record_mutations(ops, new_ids, pre)

        assert len(mutations) == 1
        assert mutations[0].type == OpType.ADD
        assert mutations[0].observation_id == "cu-xyz"

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

        assert [m.observation_id for m in mutations] == ["cu-1", "dc-2", "rr-3"]
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


class TestUpdateObservationScores:
    """Unit tests for _update_observation_scores scoring logic."""

    @staticmethod
    def _make_hip():
        """Create minimal Hippocampus with empty scores."""
        hip = object.__new__(Hippocampus)
        hip.scores = {}
        return hip

    def test_helpful_increments(self):
        hip = self._make_hip()
        hip._update_observation_scores({"a": ObservationTag.HELPFUL})
        assert hip.scores == {"a": 1}

    def test_helpful_accumulates(self):
        hip = self._make_hip()
        hip.scores = {"a": 3}
        hip._update_observation_scores({"a": ObservationTag.HELPFUL})
        assert hip.scores == {"a": 4}

    def test_harmful_decrements(self):
        hip = self._make_hip()
        hip.scores = {"a": 2}
        hip._update_observation_scores({"a": ObservationTag.HARMFUL})
        assert hip.scores == {"a": 1}

    def test_stale_decrements(self):
        hip = self._make_hip()
        hip._update_observation_scores({"a": ObservationTag.STALE})
        assert hip.scores == {"a": -1}

    def test_neutral_initializes_zero(self):
        hip = self._make_hip()
        hip._update_observation_scores({"a": ObservationTag.NEUTRAL})
        assert hip.scores == {"a": 0}

    def test_neutral_preserves_existing(self):
        hip = self._make_hip()
        hip.scores = {"a": 5}
        hip._update_observation_scores({"a": ObservationTag.NEUTRAL})
        assert hip.scores == {"a": 5}
