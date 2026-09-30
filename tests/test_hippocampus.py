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
        
        # Create agent and hippocampus (composable pattern)
        agent = MockAgent()
        hippo = Hippocampus(
            task_name="test_task",
            run_id="test-run",
            topics=[Topic(id="test/topic", type="project_scope", description="Test topic")],
        )
        
        # Agents no longer receive context_memory as input
        result = agent()
        hippo.observe(result)
        
        # End episode with the store
        hippo.end_episode(store=episode_store, artifacts={"test": "value"})
        
        # Episode should be set
        assert hippo.episode is not None
        assert hippo.episode.task == "test_task"
        assert hippo.episode.run_id == "test-run"

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

        # When all observations are deleted, load_context returns None
        assert loaded_ctx is None


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


def _stub_reflection(hippo, operations=None):
    """Replace the Distiller/Cartographer with no-LLM stubs."""
    import dspy

    hippo.distill = lambda **kw: dspy.Prediction(
        diagnosis="", observation_tags={}, cache_candidates=[]
    )
    hippo.cartograph = lambda **kw: dspy.Prediction(
        justification="", operations=list(operations or [])
    )


class TestEviction:
    """Eviction of prior observations records an EVICT mutation (not DELETE)."""

    TOPIC = "owner/repo/evict-unit"

    def _prior(self, n: int, size: int = 400) -> ContextMemory:
        return ContextMemory(
            topics=[Topic(id=self.TOPIC, type="project_scope", description="t")],
            context_understanding=[
                Observation(
                    id=f"cu-prior{i}", content=f"{i} " + "x" * size, topic_ids=[self.TOPIC]
                )
                for i in range(n)
            ],
        )

    def test_evicted_prior_records_evict_mutation(self):
        from codespy.agents.memory.hippocampus import MemoryBudget, MutationType

        prior = self._prior(4)
        hippo = Hippocampus(
            task_name="evict_test",
            budget=MemoryBudget(max_hippocampus_tokens=300),
            initial_memory=prior,
        )
        _stub_reflection(hippo)
        hippo._distill("trajectory", "question")

        survivors = hippo.cmem.ids()
        evicted_ids = {o.id for o in prior.all_observations()} - survivors
        assert evicted_ids, "budget should force eviction"

        evicts = [m for m in hippo._mutations if m.type == MutationType.EVICT]
        assert {m.observation_id for m in evicts} == evicted_ids
        assert not any(m.type == MutationType.DELETE for m in hippo._mutations)
        prior_by_id = {o.id: o for o in prior.all_observations()}
        for m in evicts:
            assert m.content is None
            assert m.section == "context_understanding"
            assert m.previous_content == prior_by_id[m.observation_id].content
            assert m.topic_ids == [self.TOPIC]
            assert m.step == 0

    def test_no_eviction_records_nothing(self):
        from codespy.agents.memory.hippocampus import MemoryBudget

        hippo = Hippocampus(
            task_name="evict_test",
            budget=MemoryBudget(max_hippocampus_tokens=100_000),
            initial_memory=self._prior(2),
        )
        _stub_reflection(hippo)
        hippo._distill("trajectory", "question")
        assert hippo._mutations == []

    def test_evicted_own_add_records_evict(self):
        from codespy.agents.memory.hippocampus import MemoryBudget, MutationType

        hippo = Hippocampus(
            task_name="evict_test",
            budget=MemoryBudget(max_hippocampus_tokens=200),
            topics=[Topic(id=self.TOPIC, type="project_scope", description="t")],
        )
        ops = [
            Operation(type=OpType.ADD, section="context_understanding", content=f"{i} " + "y" * 400)
            for i in range(3)
        ]
        _stub_reflection(hippo, ops)
        hippo._distill("trajectory", "question")

        assert len(hippo.cmem.all_observations()) < 3, "budget should force eviction"
        # Now records EVICT for own ADDs too, with ADD's content as previous_content
        evicts = [m for m in hippo._mutations if m.type == MutationType.EVICT]
        assert len(evicts) > 0, "should record EVICT for evicted own ADDs"
        for m in evicts:
            assert m.content is None
            assert m.previous_content is not None
            assert m.previous_content.startswith("0 ") or m.previous_content.startswith("1 ") or m.previous_content.startswith("2 ")

    def test_episode_keeps_mutations_after_end_episode(self):
        from codespy.agents.memory.hippocampus import MemoryBudget, MutationType

        hippo = Hippocampus(
            task_name="evict_test",
            budget=MemoryBudget(max_hippocampus_tokens=300),
            initial_memory=self._prior(4),
        )
        _stub_reflection(hippo)
        hippo.observe("trajectory")
        hippo.end_episode()
        assert hippo.episode is not None
        assert any(m.type == MutationType.EVICT for m in hippo.episode.mutations)
        assert hippo._mutations == []

    def test_cartographer_cannot_emit_evict(self):
        """EVICT is a MutationType only; Operation (Cartographer output) rejects it."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Operation(type="EVICT", observation_id="cu-x")


class TestEvictTombstone:
    """Store round-trip: EVICT writes a tombstone version and hides the observation."""

    def _row_versions(self, store, obs_id):
        from psycopg.rows import dict_row

        with store._pool.connection() as conn:
            conn.row_factory = dict_row
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT version, op_type, content, previous_content FROM observations "
                    "WHERE bank_id = %s AND id = %s ORDER BY version",
                    (store.bank_id, obs_id),
                )
                return cur.fetchall()

    def test_evict_tombstone_round_trip(self, episode_store):
        import uuid

        from codespy.agents.memory.hippocampus import MutationType

        topic_id = f"owner/repo/evict-{uuid.uuid4().hex[:8]}"
        topic = Topic(id=topic_id, type="project_scope", description="t")
        keep_id, evict_id = f"cu-keep{uuid.uuid4().hex}", f"cu-evict{uuid.uuid4().hex}"

        ctx1 = ContextMemory(
            topics=[topic],
            context_understanding=[
                Observation(id=keep_id, content="keep me", topic_ids=[topic_id]),
                Observation(id=evict_id, content="evict me", topic_ids=[topic_id]),
            ],
        )
        episode_store.save_episode(Episode(
            id=uuid.uuid4(), task="code_review", module="m", question="q1",
            context_memory=ctx1, timestamp=datetime.now(UTC), run_id="r1",
            mutations=[
                Mutation(step=0, type=MutationType.ADD, observation_id=oid,
                         section="context_understanding", content=c, topic_ids=[topic_id])
                for oid, c in ((keep_id, "keep me"), (evict_id, "evict me"))
            ],
            artifacts={},
        ))

        # Episode 2: the prior observation was evicted (not in context_memory)
        ctx2 = ContextMemory(
            topics=[topic],
            context_understanding=[Observation(id=keep_id, content="keep me", topic_ids=[topic_id])],
        )
        episode_store.save_episode(Episode(
            id=uuid.uuid4(), task="code_review", module="m", question="q2",
            context_memory=ctx2, timestamp=datetime.now(UTC), run_id="r2",
            mutations=[
                Mutation(step=0, type=MutationType.EVICT, observation_id=evict_id,
                         section="context_understanding", content=None,
                         previous_content="evict me", topic_ids=[topic_id])
            ],
            artifacts={},
        ))

        rows = self._row_versions(episode_store, evict_id)
        assert [(r["version"], r["op_type"]) for r in rows] == [(1, "ADD"), (2, "EVICT")]
        assert rows[1]["content"] is None
        assert rows[1]["previous_content"] == "evict me"

        loaded = episode_store.load_context(task="code_review", topic_ids=[topic_id])
        assert loaded is not None
        assert loaded.ids() == {keep_id}

    def test_add_then_evict_same_run_no_orphan_row(self, episode_store):
        """ADD then EVICT of a new id in same run → no row in observations."""
        import uuid

        from codespy.agents.memory.hippocampus import MutationType

        topic_id = f"owner/repo/add-evict-{uuid.uuid4().hex[:8]}"
        topic = Topic(id=topic_id, type="project_scope", description="t")
        new_id = f"cu-new{uuid.uuid4().hex}"

        # Single episode: ADD then EVICT the same observation
        ctx = ContextMemory(
            topics=[topic],
            context_understanding=[],  # Evicted, so not in context
        )
        episode_store.save_episode(Episode(
            id=uuid.uuid4(), task="code_review", module="m", question="q",
            context_memory=ctx, timestamp=datetime.now(UTC), run_id="r",
            mutations=[
                Mutation(step=0, type=MutationType.ADD, observation_id=new_id,
                         section="context_understanding", content="new content", topic_ids=[topic_id]),
                Mutation(step=1, type=MutationType.EVICT, observation_id=new_id,
                         section="context_understanding", content=None,
                         previous_content="new content", topic_ids=[topic_id]),
            ],
            artifacts={},
        ))

        # No rows for this observation (ADD was never persisted, EVICT skipped tombstone)
        rows = self._row_versions(episode_store, new_id)
        assert rows == [], f"expected no rows for ADD→EVICT in same run, got {rows}"

    def test_load_orders_oldest_first_so_eviction_drops_oldest(self, episode_store):
        """Load order follows episode time, not id; evict() then drops the oldest."""
        import uuid
        from datetime import timedelta

        from codespy.agents.memory.hippocampus import MemoryBudget, MutationType

        suffix = uuid.uuid4().hex
        topic_id = f"owner/repo/order-{suffix[:8]}"
        topic = Topic(id=topic_id, type="project_scope", description="t")
        # Id order is the reverse of time order: the oldest has the largest id.
        ids_oldest_first = [f"cu-z{suffix}", f"cu-m{suffix}", f"cu-a{suffix}"]
        now = datetime.now(UTC)
        for age, oid in zip((3, 2, 1), ids_oldest_first, strict=True):
            content = f"{oid} " + "x" * 400
            episode_store.save_episode(Episode(
                id=uuid.uuid4(), task="code_review", module="m", question="q",
                context_memory=ContextMemory(
                    topics=[topic],
                    context_understanding=[
                        Observation(id=oid, content=content, topic_ids=[topic_id])
                    ],
                ),
                timestamp=now - timedelta(days=age), run_id=f"r-{age}",
                mutations=[
                    Mutation(step=0, type=MutationType.ADD, observation_id=oid,
                             section="context_understanding", content=content,
                             topic_ids=[topic_id])
                ],
                artifacts={},
            ))

        loaded = episode_store.load_context(task="code_review", topic_ids=[topic_id])
        assert loaded is not None
        assert [o.id for o in loaded.context_understanding] == ids_oldest_first

        # Budget fits only the newest observation: the two oldest are evicted.
        hippo = Hippocampus(
            task_name="code_review",
            budget=MemoryBudget(max_hippocampus_tokens=250),
            initial_memory=loaded,
        )
        _stub_reflection(hippo)
        hippo._distill("trajectory", "question")
        assert hippo.cmem.ids() == {ids_oldest_first[-1]}
        assert [m.observation_id for m in hippo._mutations] == ids_oldest_first[:2]


class TestLoadContextTombstones:
    """load_context never fails on tombstones and uses bindings of the loaded version."""

    @staticmethod
    def _save_add(store, topic, oid, content, topic_ids=None):
        import uuid

        from codespy.agents.memory.hippocampus import MutationType

        ep_id = uuid.uuid4()
        store.save_episode(Episode(
            id=ep_id, task="code_review", module="m", question="q",
            context_memory=ContextMemory(
                topics=[topic],
                context_understanding=[
                    Observation(id=oid, content=content, topic_ids=topic_ids or [topic.id])
                ],
            ),
            timestamp=datetime.now(UTC), run_id="r",
            mutations=[
                Mutation(step=0, type=MutationType.ADD, observation_id=oid,
                         section="context_understanding", content=content,
                         topic_ids=topic_ids or [topic.id])
            ],
            artifacts={},
        ))
        return ep_id

    def test_unknown_tombstone_type_is_skipped_not_fatal(self, episode_store):
        """A NULL-content latest version of an unknown op_type hides only that observation."""
        import uuid

        suffix = uuid.uuid4().hex
        topic = Topic(id=f"owner/repo/tomb-{suffix[:8]}", type="project_scope", description="t")
        live_id, dead_id = f"cu-live{suffix}", f"cu-dead{suffix}"
        self._save_add(episode_store, topic, live_id, "live")
        ep_id = self._save_add(episode_store, topic, dead_id, "dead")

        with episode_store._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations "
                "(bank_id, id, version, type, content, episode_id, step, op_type, "
                "previous_content, ordinal) "
                "VALUES (%s, %s, 2, 'context_understanding', NULL, %s, 0, 'ARCHIVE', 'dead', 0)",
                (episode_store.bank_id, dead_id, str(ep_id)),
            )
            conn.commit()

        loaded = episode_store.load_context(task="code_review", topic_ids=[topic.id])
        assert loaded is not None
        assert loaded.ids() == {live_id}

    def test_tombstone_types_come_from_tombstone_types(self, episode_store, monkeypatch):
        """load_context reads the tombstone list from TOMBSTONE_TYPES."""
        import uuid

        from codespy.agents.memory.hippocampus import context_memory as cm

        suffix = uuid.uuid4().hex
        topic = Topic(id=f"owner/repo/types-{suffix[:8]}", type="project_scope", description="t")
        oid = f"cu-add{suffix}"
        self._save_add(episode_store, topic, oid, "content")

        # Treat ADD as a tombstone type: the ADD-only observation must disappear.
        monkeypatch.setattr(
            cm, "TOMBSTONE_TYPES", frozenset(cm.TOMBSTONE_TYPES | {cm.MutationType.ADD})
        )
        assert episode_store.load_context(task="code_review", topic_ids=[topic.id]) is None

    def test_bindings_come_from_loaded_version(self, episode_store):
        """After a REPLACE rebinds topics, only the latest version's bindings load."""
        import uuid

        from codespy.agents.memory.hippocampus import MutationType

        suffix = uuid.uuid4().hex
        t1 = Topic(id=f"owner/repo/b1-{suffix[:8]}", type="project_scope", description="t1")
        t2 = Topic(id=f"owner/repo/b2-{suffix[:8]}", type="project_scope", description="t2")
        oid = f"cu-rebind{suffix}"
        self._save_add(episode_store, t1, oid, "v1", topic_ids=[t1.id])

        episode_store.save_episode(Episode(
            id=uuid.uuid4(), task="code_review", module="m", question="q",
            context_memory=ContextMemory(
                topics=[t1, t2],
                context_understanding=[Observation(id=oid, content="v2", topic_ids=[t2.id])],
            ),
            timestamp=datetime.now(UTC), run_id="r",
            mutations=[
                Mutation(step=0, type=MutationType.REPLACE, observation_id=oid,
                         section="context_understanding", content="v2",
                         previous_content="v1", topic_ids=[t2.id])
            ],
            artifacts={},
        ))

        loaded = episode_store.load_context(task="code_review", topic_ids=[t1.id])
        assert loaded is not None
        obs = loaded.find_observation(oid)[1]
        assert obs.content == "v2"
        assert obs.topic_ids == [t2.id]


def _recall_records():
    from codespy.agents.memory.recall import RecallRecord

    return [
        RecallRecord(
            ordinal=0,
            kind="load",
            query="context query",
            reach="org",
            reflects=5,
            status="ok",
            text="## Around this work\nRECALLED-SECRET-TEXT",
            model="openai/gpt-4o-mini",
            llm_calls=3,
            input_tokens=1200,
            output_tokens=300,
            input_cost=0.012,
            output_cost=0.006,
            latency_ms=4200,
            details={"briefings": 1, "remote": 2, "facets": {"context": {"count": 1}}},
        ),
        RecallRecord(
            ordinal=1,
            kind="tool",
            query="where is login",
            reach="local",
            status="limit",
            text="recall limit reached",
            details={"requested_reach": "bank", "mode": "reflect"},
        ),
    ]


class TestEpisodeStoreRecalls:
    """Prefrontal recalls are persisted to the recalls table, never distilled."""

    def _episode(self, recalls):
        import uuid

        return Episode(
            id=uuid.uuid4(),
            task="code_review",
            module="m",
            question="q",
            context_memory=ContextMemory(
                topics=[Topic(id="owner/repo/recalls", type="project_scope", description="t")]
            ),
            timestamp=datetime.now(UTC),
            run_id="run-recalls",
            recalls=recalls,
        )

    def _rows(self, store, episode_id):
        from psycopg.rows import dict_row

        with store._pool.connection() as conn:
            conn.row_factory = dict_row
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM recalls WHERE bank_id = %s AND episode_id = %s ORDER BY ordinal",
                    (store.bank_id, str(episode_id)),
                )
                return cur.fetchall()

    def test_save_writes_recall_rows(self, episode_store):
        episode = self._episode(_recall_records())
        episode_store.save_episode(episode)

        rows = self._rows(episode_store, episode.id)
        assert [r["ordinal"] for r in rows] == [0, 1]
        load, tool = rows
        assert load["kind"] == "load" and load["status"] == "ok"
        assert load["text"].endswith("RECALLED-SECRET-TEXT")
        assert load["model"] == "openai/gpt-4o-mini"
        assert load["llm_calls"] == 3
        assert load["input_tokens"] == 1200 and load["output_tokens"] == 300
        assert load["input_cost"] == pytest.approx(0.012)
        assert load["output_cost"] == pytest.approx(0.006)
        assert load["latency_ms"] == 4200
        assert load["reflects"] == 5 and load["reach"] == "org"
        assert load["details"]["facets"]["context"]["count"] == 1
        assert tool["status"] == "limit"
        assert tool["details"] == {"requested_reach": "bank", "mode": "reflect"}

    def test_save_is_idempotent(self, episode_store):
        episode = self._episode(_recall_records())
        episode_store.save_episode(episode)
        episode_store.save_episode(episode)
        assert len(self._rows(episode_store, episode.id)) == 2

    def test_episode_without_recalls(self, episode_store):
        episode = self._episode([])
        episode_store.save_episode(episode)
        assert self._rows(episode_store, episode.id) == []


class TestHippocampusRecalls:
    def test_end_episode_attaches_recalls_without_distilling_them(self):
        hippo = Hippocampus(task_name="code_review", run_id="r", question="question")
        hippo.observe("trajectory text")
        seen: list[tuple[str, str]] = []
        hippo._distill = lambda trajectory, question: seen.append((trajectory, question))

        recalls = _recall_records()
        hippo.end_episode(recalls=recalls)

        assert hippo.episode is not None
        assert [r.ordinal for r in hippo.episode.recalls] == [0, 1]
        assert len(seen) == 1
        trajectory, question = seen[0]
        assert "RECALLED-SECRET-TEXT" not in trajectory
        assert "RECALLED-SECRET-TEXT" not in question

    def test_end_episode_defaults_to_no_recalls(self):
        hippo = Hippocampus(task_name="code_review", run_id="r")
        hippo.observe("trajectory text")
        hippo._distill = lambda trajectory, question: None
        hippo.end_episode()
        assert hippo.episode is not None and hippo.episode.recalls == []
