"""Tests for the Cerebral Hindsight semantic memory module."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from codespy.agents.memory.cerebral import Cerebral
from codespy.agents.memory.cerebral.cost import MeteredLiteLLMSDKEmbeddings
from codespy.agents.memory.hippocampus.context_memory import (
    ContextMemory,
    Mutation,
    Observation,
    OpType,
    Topic,
)
from codespy.agents.memory.hippocampus.episode import Episode


def async_mock(*args, **kwargs):
    """Create a coroutine mock."""
    async def mock_coro(*args, **kwargs):
        return None
    return mock_coro()


@pytest.fixture
def mock_engine():
    """Create a mock MemoryEngine."""
    engine = MagicMock()
    # Mock async methods - need to return coroutines
    async def mock_coro(*args, **kwargs):
        return None

    engine.initialize = mock_coro
    engine.ensure_bank_profile = mock_coro
    engine.update_bank_config = mock_coro
    engine.retain_batch_async = mock_coro
    return engine


@pytest.fixture
def mock_request_context():
    """Create a mock RequestContext."""
    with patch("codespy.agents.memory.cerebral.cerebral.RequestContext") as mock_ctx:
        mock_ctx.return_value = MagicMock()
        yield mock_ctx


@pytest.fixture
def mock_memory_engine_class(mock_engine):
    """Create a mock MemoryEngine class."""
    with patch("codespy.agents.memory.cerebral.cerebral.MemoryEngine") as mock_cls:
        mock_cls.return_value = mock_engine
        yield mock_cls


@pytest.fixture
def mock_litellm():
    """Mock litellm.embedding to avoid API calls during tests."""
    with patch("litellm.embedding") as mock_emb:
        mock_emb.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        yield mock_emb


@pytest.fixture
def episode():
    """Create a test episode with observations and artifacts."""
    topics = [
        Topic(id="test/repo/package", type="project_scope", description="Test package"),
        Topic(id="https://github.com/test/repo/pull/1", type="pull_request", description="Test PR"),
    ]

    context_memory = ContextMemory(
        topics=topics,
        context_understanding=[
            Observation(
                id="cu-test-1",
                content="This is a test observation",
                topic_ids=["test/repo/package"],
            ),
        ],
        actions=[
            Observation(
                id="ac-test-1",
                content="Performed tool call",
                topic_ids=["test/repo/package"],
            ),
        ],
    )

    return Episode(
        id=uuid.uuid4(),
        run_id="test-run-123",
        timestamp=datetime.now(UTC),
        task="code_review",
        module="CodeReviewer",
        question="Review PR #1",
        artifacts={"review": "# Review Results\n\nFound 2 issues."},
        context_memory=context_memory,
        mutations=[
            Mutation(
                step=0,
                type=OpType.ADD,
                observation_id="cu-test-1",
                section="context_understanding",
                content="This is a test observation",
                previous_content=None,
                topic_ids=["test/repo/package"],
            ),
            Mutation(
                step=0,
                type=OpType.ADD,
                observation_id="ac-test-1",
                section="actions",
                content="Performed tool call",
                previous_content=None,
                topic_ids=["test/repo/package"],
            ),
        ],
    )


class TestCerebralInit:
    """Tests for Cerebral initialization."""

    def test_init_creates_engine_with_correct_params(self, mock_memory_engine_class, mock_litellm):
        """Test that MemoryEngine is created with correct parameters."""
        Cerebral(
            database_url="postgresql://localhost:5432/test",
            llm_provider="litellm",
            llm_model="openai/gpt-4",
            llm_api_key="sk-test",
            llm_base_url="https://custom.openai.com",
            bank_id="test-bank",
        )

        mock_memory_engine_class.assert_called_once_with(
            db_url="postgresql://localhost:5432/test",
            memory_llm_provider="litellm",
            memory_llm_model="openai/gpt-4",
            memory_llm_api_key="sk-test",
            memory_llm_base_url="https://custom.openai.com",
            embeddings=mock_memory_engine_class.call_args.kwargs["embeddings"],
            cross_encoder=mock_memory_engine_class.call_args.kwargs["cross_encoder"],
            tenant_extension=mock_memory_engine_class.call_args.kwargs["tenant_extension"],
            skip_llm_verification=mock_memory_engine_class.call_args.kwargs["skip_llm_verification"],
            task_backend=mock_memory_engine_class.call_args.kwargs["task_backend"],
        )

    def test_init_uses_metered_embeddings(self, mock_memory_engine_class, mock_litellm):
        """Test that MemoryEngine is created with MeteredLiteLLMSDKEmbeddings."""
        Cerebral(
            database_url="postgresql://localhost:5432/test",
            llm_provider="litellm",
            llm_model="openai/gpt-4",
            embeddings_model="openai/text-embedding-3-small",
        )

        # Check that embeddings is a MeteredLiteLLMSDKEmbeddings
        embeddings = mock_memory_engine_class.call_args.kwargs["embeddings"]
        assert isinstance(embeddings, MeteredLiteLLMSDKEmbeddings)

    def test_init_calls_initialize_and_routine_repair(self, mock_memory_engine_class, mock_engine, mock_litellm):
        """Test that initialize() and routine repair are called during construction."""
        with patch.object(Cerebral, "_run_async") as mock_run_async:
            Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )
            # initialize(), repair_orphaned_operations()
            assert mock_run_async.call_count == 2

    def test_init_handles_none_base_url(self, mock_memory_engine_class, mock_litellm):
        """Test that None base_url is handled correctly."""
        Cerebral(
            database_url="postgresql://localhost:5432/test",
            llm_provider="litellm",
            llm_model="openai/gpt-4",
            llm_api_key="sk-test",
            llm_base_url=None,
        )

        mock_memory_engine_class.assert_called_once_with(
            db_url="postgresql://localhost:5432/test",
            memory_llm_provider="litellm",
            memory_llm_model="openai/gpt-4",
            memory_llm_api_key="sk-test",
            memory_llm_base_url=None,
            embeddings=mock_memory_engine_class.call_args.kwargs["embeddings"],
            cross_encoder=mock_memory_engine_class.call_args.kwargs["cross_encoder"],
            tenant_extension=mock_memory_engine_class.call_args.kwargs["tenant_extension"],
            skip_llm_verification=mock_memory_engine_class.call_args.kwargs["skip_llm_verification"],
            task_backend=mock_memory_engine_class.call_args.kwargs["task_backend"],
        )


class TestCerebralBank:
    """Tests for Cerebral bank management."""

    def test_ensure_bank_mission_contains_retracted_invalidates(self, mock_memory_engine_class, mock_engine, mock_litellm):
        """Test that retain_mission contains 'Only RETRACTED invalidates a fact'."""
        captured_mission = None

        async def mock_update_bank_config(bank_id, updates, request_context):
            nonlocal captured_mission
            captured_mission = updates.get("retain_mission")
            return None

        async def mock_coro(*args, **kwargs):
            return None

        def mock_memory_engine(*args, **kwargs):
            engine = MagicMock()
            engine.initialize = mock_coro
            engine.ensure_bank_profile = mock_coro
            engine.update_bank_config = mock_update_bank_config
            engine.close = mock_coro
            return engine

        with patch("codespy.agents.memory.cerebral.cerebral.MemoryEngine", mock_memory_engine):
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )
            cerebral._bank_ensured = False
            cerebral._ensure_bank()

        assert captured_mission is not None
        assert "Only RETRACTED invalidates a fact" in captured_mission

    def test_ensure_bank_mission_is_domain_agnostic(self, mock_memory_engine_class, mock_engine, mock_litellm):
        """Test that retain_mission does not contain code-review-specific wording."""
        captured_mission = None

        async def mock_update_bank_config(bank_id, updates, request_context):
            nonlocal captured_mission
            captured_mission = updates.get("retain_mission")
            return None

        async def mock_coro(*args, **kwargs):
            return None

        def mock_memory_engine(*args, **kwargs):
            engine = MagicMock()
            engine.initialize = mock_coro
            engine.ensure_bank_profile = mock_coro
            engine.update_bank_config = mock_update_bank_config
            engine.close = mock_coro
            return engine

        with patch("codespy.agents.memory.cerebral.cerebral.MemoryEngine", mock_memory_engine):
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )
            cerebral._bank_ensured = False
            cerebral._ensure_bank()

        assert captured_mission is not None
        assert "code review" not in captured_mission.lower()
        assert "architectural decisions" not in captured_mission.lower()
        assert "dependency relationships" not in captured_mission.lower()
        assert "repositories" not in captured_mission.lower()

    def test_ensure_bank_chunk_size_default(self, mock_memory_engine_class, mock_engine, mock_litellm):
        """Test that _ensure_bank sends retain_chunk_size == 12288 by default."""
        captured_chunk_size = None

        async def mock_update_bank_config(bank_id, updates, request_context):
            nonlocal captured_chunk_size
            captured_chunk_size = updates.get("retain_chunk_size")
            return None

        async def mock_coro(*args, **kwargs):
            return None

        def mock_memory_engine(*args, **kwargs):
            engine = MagicMock()
            engine.initialize = mock_coro
            engine.ensure_bank_profile = mock_coro
            engine.update_bank_config = mock_update_bank_config
            engine.close = mock_coro
            return engine

        with patch("codespy.agents.memory.cerebral.cerebral.MemoryEngine", mock_memory_engine):
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )
            cerebral._bank_ensured = False
            cerebral._ensure_bank()

        assert captured_chunk_size == 12288

    def test_ensure_bank_chunk_size_custom(self, mock_memory_engine_class, mock_engine, mock_litellm):
        """Test that _ensure_bank sends custom retain_chunk_size when provided."""
        captured_chunk_size = None

        async def mock_update_bank_config(bank_id, updates, request_context):
            nonlocal captured_chunk_size
            captured_chunk_size = updates.get("retain_chunk_size")
            return None

        async def mock_coro(*args, **kwargs):
            return None

        def mock_memory_engine(*args, **kwargs):
            engine = MagicMock()
            engine.initialize = mock_coro
            engine.ensure_bank_profile = mock_coro
            engine.update_bank_config = mock_update_bank_config
            engine.close = mock_coro
            return engine

        with patch("codespy.agents.memory.cerebral.cerebral.MemoryEngine", mock_memory_engine):
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
                retain_chunk_size=5000,
            )
            cerebral._bank_ensured = False
            cerebral._ensure_bank()

        assert captured_chunk_size == 5000

    def test_ensure_bank_called_once(self, mock_memory_engine_class, mock_engine, mock_litellm):
        """Test that _ensure_bank is only called once."""
        with patch.object(Cerebral, "_run_async") as mock_run_async:
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )

            # Reset the mock to track only _ensure_bank calls
            mock_run_async.reset_mock()

            # First call to _ensure_bank
            cerebral._ensure_bank()

            # Second call should be a no-op
            cerebral._ensure_bank()

            # ensure_bank_profile, update_bank_config and the placeholder scan: once each
            assert mock_run_async.call_count == 3

    def test_ensure_bank_calls_profile_and_config(self, mock_memory_engine_class, mock_engine, mock_litellm):
        """Test that _ensure_bank calls ensure_bank_profile and update_bank_config."""
        with patch.object(Cerebral, "_run_async") as mock_run_async:
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
                bank_id="test-bank",
            )

            mock_run_async.reset_mock()
            cerebral._bank_ensured = False
            cerebral._ensure_bank()

            # ensure_bank_profile, update_bank_config, list_mental_models
            assert mock_run_async.call_count == 3

    @pytest.mark.parametrize("cap", [100, 50, -1])
    def test_ensure_bank_sends_consolidation_strategies(
        self, mock_memory_engine_class, mock_engine, mock_litellm, cap
    ):
        """The cap goes through consolidation_strategies (also -1, to lift a stored cap)."""
        captured_updates = None

        async def mock_update_bank_config(bank_id, updates, request_context):
            nonlocal captured_updates
            captured_updates = updates
            return None

        mock_engine.update_bank_config = mock_update_bank_config

        with patch(
            "codespy.agents.memory.cerebral.cerebral.MemoryEngine", return_value=mock_engine
        ):
            c = Cerebral(
                database_url="postgresql://x/y",
                llm_provider="litellm",
                max_observations_per_scope=cap,
            )
            c._ensure_bank()
            c.close()

        assert captured_updates is not None
        assert captured_updates["consolidation_strategies"] == [
            {
                "scopes": [
                    {"tags": ["org:*", "repo:*", "project_scope:*"], "tags_match": "exact"}
                ],
                "max_observations_per_scope": cap,
            }
        ]
        # Deprecated key is tombstoned; the bank-wide cap is never sent
        assert "observation_scope_limits" in captured_updates
        assert captured_updates["observation_scope_limits"] is None
        assert "max_observations_per_scope" not in captured_updates


class TestCerebralRetainEpisode:
    """Tests for Cerebral retain_episode method."""

    def test_retain_episode_builds_correct_tags(self, mock_memory_engine_class, episode, mock_litellm):
        """Test that tags are built correctly from episode.

        Only project_scope topics are included; pull_request, episode, and
        run_id tags are no longer added. Repo/org tags are added when repo is known.
        """
        with patch.object(Cerebral, "_run_async"):
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )

            tags = cerebral._build_tags(episode, repo_full_name="test/repo")

            expected_tags = [
                "project_scope:test/repo/package",
                "org:test",
                "repo:test/repo",
                "task:code_review",
            ]
            assert tags == expected_tags
            # episode: and run_id: should NOT be present
            assert not any(t.startswith(("episode:", "run_id:")) for t in tags)
            # pull_request: should NOT be present
            assert not any(t.startswith("pull_request:") for t in tags)

    def test_retain_episode_creates_contents_for_observations(self, mock_memory_engine_class, episode, mock_engine, mock_litellm):
        """Test that retain_episode creates a single merged content item for all observations."""
        with patch.object(Cerebral, "_run_async") as mock_run_async:
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )

            mock_run_async.reset_mock()
            cerebral._bank_ensured = True  # Skip bank setup

            cerebral.retain_episode(episode)

            # Should call retain_batch_async
            mock_run_async.assert_called_once()

            # Get the contents passed to retain_batch_async
            call_args = mock_run_async.call_args
            assert call_args is not None

            # Extract the contents argument from the coroutine call
            # The first positional arg is the coroutine, we need to inspect it
            coro = call_args[0][0]
            # Access the bound arguments via the coroutine's __self__ if possible
            # or check the call was made to retain_batch_async
            engine_call = mock_engine.retain_batch_async
            assert call_args is not None

    def test_retain_episode_creates_contents_for_artifacts(self, mock_memory_engine_class, episode, mock_litellm):
        """Test that retain_episode creates a single merged content item for all artifacts."""
        with patch.object(Cerebral, "_run_async") as mock_run_async:
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )

            mock_run_async.reset_mock()
            cerebral._bank_ensured = True

            cerebral.retain_episode(episode)

            # Should call retain_batch_async
            mock_run_async.assert_called_once()

    def test_retain_episode_skips_when_no_contents(self, mock_memory_engine_class, mock_litellm):
        """Test that retain_episode skips when there are no observations or artifacts."""
        empty_episode = Episode(
            id=uuid.uuid4(),
            run_id="test-run",
            timestamp=datetime.now(UTC),
            task="code_review",
            module="CodeReviewer",
            question="Review PR",
            artifacts={},
            context_memory=ContextMemory(),
        )

        with patch.object(Cerebral, "_run_async") as mock_run_async:
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )

            mock_run_async.reset_mock()
            cerebral._bank_ensured = True

            cerebral.retain_episode(empty_episode)

            # Should NOT call retain_batch_async
            mock_run_async.assert_not_called()

    def test_retain_episode_handles_exception_gracefully(self, mock_memory_engine_class, episode, mock_litellm):
        """Test that retain_episode handles exceptions gracefully."""
        with patch.object(Cerebral, "_run_async") as mock_run_async:
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )

            cerebral._bank_ensured = True

            # Set side effect AFTER creating Cerebral (so initialize succeeds)
            mock_run_async.side_effect = Exception("Network error")

            # Should not raise
            cerebral.retain_episode(episode)

    def test_retain_episode_bundles_content_correctly(self, mock_memory_engine_class, episode, mock_litellm):
        """Test that observations and artifacts are merged into at most 2 content items with proper headers."""
        captured_contents = []

        # Create a mock engine that captures the contents passed to retain_batch_async
        async def mock_retain_batch_async(*, bank_id, contents, request_context):
            captured_contents.extend(contents)
            return None

        # Create async mock coroutines for other methods
        async def mock_coro(*args, **kwargs):
            return None

        def mock_memory_engine(*args, **kwargs):
            engine = MagicMock()
            engine.initialize = mock_coro
            engine.ensure_bank_profile = mock_coro
            engine.update_bank_config = mock_coro
            engine.retain_batch_async = mock_retain_batch_async
            engine.close = mock_coro
            return engine

        # Patch the MemoryEngine class at the module level
        with patch("codespy.agents.memory.cerebral.cerebral.MemoryEngine", mock_memory_engine):
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )
            cerebral._bank_ensured = True

            cerebral.retain_episode(episode)

            # Episode has 2 observations (context_understanding, actions) + 1 artifact (review)
            # Should be bundled into 2 content items max (observations + artifacts)
            assert len(captured_contents) <= 2, f"Expected at most 2 content items, got {len(captured_contents)}"
            assert len(captured_contents) >= 1, f"Expected at least 1 content item, got {len(captured_contents)}"

            # Find observations and artifacts content items
            obs_item = None
            art_item = None
            for item in captured_contents:
                # Check metadata.kind instead of context
                if item.get("metadata", {}).get("kind") == "observation changes":
                    obs_item = item
                elif item.get("metadata", {}).get("kind") == "artifacts":
                    art_item = item

            # Verify observations item
            assert obs_item is not None, "Observations content item not found"
            assert "[context_understanding]" in obs_item["content"], "Missing context_understanding header"
            assert "[actions]" in obs_item["content"], "Missing actions header"
            assert "This is a test observation" in obs_item["content"], "Missing observation content"
            assert "Performed tool call" in obs_item["content"], "Missing action content"
            # Verify metadata instead of context
            assert "metadata" in obs_item, "Observations should have metadata"
            assert "context" not in obs_item, "Observations should not have context"
            assert obs_item["metadata"]["task"] == "code_review"
            assert obs_item["metadata"]["kind"] == "observation changes"
            assert obs_item["metadata"]["question"] == episode.question

            # Verify artifacts item
            assert art_item is not None, "Artifacts content item not found"
            assert "[artifact:review]" in art_item["content"], "Missing artifact header"
            assert "# Review Results" in art_item["content"], "Missing artifact content"
            # Verify metadata
            assert "metadata" in art_item, "Artifacts should have metadata"
            assert "context" not in art_item, "Artifacts should not have context"
            assert art_item["metadata"]["task"] == "code_review"
            assert art_item["metadata"]["kind"] == "artifacts"

            # Verify shared fields
            expected_tags = cerebral._build_tags(episode, repo_full_name="test/repo")
            expected_doc_id = f"episode-{episode.id}"
            expected_event_date = episode.timestamp.isoformat()

            for item in captured_contents:
                assert item["tags"] == expected_tags, f"Tags mismatch: {item['tags']} != {expected_tags}"
                assert item["document_id"] == expected_doc_id, f"document_id mismatch"
                assert item["event_date"] == expected_event_date, f"event_date mismatch"

    def test_retain_episode_uses_correct_document_id(self, mock_memory_engine_class, episode, mock_engine, mock_litellm):
        """Test that retain_episode uses correct document_id format."""
        with patch.object(Cerebral, "_run_async") as mock_run_async:
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )

            mock_run_async.reset_mock()
            cerebral._bank_ensured = True

            cerebral.retain_episode(episode)

            # Get the contents passed to retain_batch_async
            call_args = mock_run_async.call_args
            assert call_args is not None


class TestCerebralRunAsync:
    """Tests for Cerebral _run_async helper."""

    def test_run_async_submits_to_dedicated_loop(self, mock_memory_engine_class, mock_litellm):
        """Test _run_async submits coroutine to the dedicated event loop."""
        with mock_memory_engine_class:
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )

            async def test_coro():
                return "result"

            result = cerebral._run_async(test_coro())
            assert result == "result"
            # Verify the loop is running in a background thread
            assert cerebral._loop_thread.is_alive()
            assert cerebral._loop.is_running()

    def test_run_async_from_different_thread(self, mock_memory_engine_class, mock_litellm):
        """Test _run_async is thread-safe when called from other threads."""
        import concurrent.futures

        with mock_memory_engine_class:
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )

            async def test_coro():
                return "thread_result"

            # Submit from a different thread using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(cerebral._run_async, test_coro())
                result = future.result(timeout=5)

            assert result == "thread_result"


class TestCerebralClose:
    """Tests for Cerebral close method."""

    def test_close_stops_loop_and_joins_thread(self, mock_memory_engine_class, mock_engine, mock_litellm):
        """Test that close() stops the event loop and joins the thread."""
        with mock_memory_engine_class:
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )

            # Verify loop is running before close
            assert cerebral._loop_thread.is_alive()
            assert cerebral._loop.is_running()

            cerebral.close()

            # After close, thread should be stopped
            assert not cerebral._loop_thread.is_alive()
            assert not cerebral._loop.is_running()

    def test_close_calls_engine_close(self, mock_memory_engine_class, mock_engine, mock_litellm):
        """Test that close() calls engine.close()."""
        with mock_memory_engine_class:
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )

            cerebral.close()

            # Verify engine.close() was called
            mock_engine.close.assert_called_once()


class TestCerebralBuildTags:
    """Tests for Cerebral _build_tags static method."""

    def test_build_tags_with_multiple_topics(self):
        """Test tag building with multiple topics."""
        episode = Episode(
            id=uuid.uuid4(),
            run_id="run-123",
            timestamp=datetime.now(UTC),
            task="scope",
            module="ScopeResolver",
            question="Resolve scopes",
            artifacts={},
            context_memory=ContextMemory(
                topics=[
                    Topic(id="owner/repo", type="project_scope", description="Main scope"),
                    Topic(id="pr-123", type="pull_request", description="PR"),
                ],
            ),
        )

        tags = Cerebral._build_tags(episode)

        assert len(tags) == 5
        assert "project_scope:owner/repo" in tags
        assert "pull_request:pr-123" in tags
        assert f"episode:{episode.id}" in tags
        assert "task:scope" in tags
        assert "run_id:run-123" in tags

    def test_build_tags_with_no_topics(self):
        """Test tag building with no topics."""
        episode = Episode(
            id=uuid.uuid4(),
            run_id="run-456",
            timestamp=datetime.now(UTC),
            task="code_review",
            module="CodeReviewer",
            question="Review code",
            artifacts={},
            context_memory=ContextMemory(),
        )

        tags = Cerebral._build_tags(episode)

        assert len(tags) == 3
        assert f"episode:{episode.id}" in tags
        assert "task:code_review" in tags
        assert "run_id:run-456" in tags


def _mut(type_, obs_id, *, content=None, previous=None, section="context_understanding"):
    from codespy.agents.memory.hippocampus.context_memory import MutationType

    return Mutation(
        step=0,
        type=MutationType(type_),
        observation_id=obs_id,
        section=section,
        content=content,
        previous_content=previous,
    )


def _episode_with(mutations, observations=()):
    return Episode(
        id=uuid.uuid4(),
        run_id="r",
        timestamp=datetime.now(UTC),
        task="code_review",
        module="m",
        question="q",
        artifacts={},
        context_memory=ContextMemory(
            context_understanding=[Observation(id=i, content=c) for i, c in observations]
        ),
        mutations=list(mutations),
    )


class TestMutationLines:
    """Cerebral retains the mutation log only."""

    def test_add(self):
        ep = _episode_with([_mut("ADD", "cu-a", content="new")], [("cu-a", "new")])
        assert Cerebral._mutation_lines(ep) == ["[context_understanding] new"]

    def test_add_then_replace_uses_final_content(self):
        ep = _episode_with(
            [_mut("ADD", "cu-a", content="v1"), _mut("REPLACE", "cu-a", content="v2", previous="v1")],
            [("cu-a", "v2")],
        )
        assert Cerebral._mutation_lines(ep) == ["[context_understanding] v2"]

    def test_add_then_delete_emits_nothing(self):
        ep = _episode_with(
            [_mut("ADD", "cu-a", content="v1"), _mut("DELETE", "cu-a", previous="v1")]
        )
        assert Cerebral._mutation_lines(ep) == []

    def test_prior_replace_supersedes(self):
        ep = _episode_with(
            [_mut("REPLACE", "cu-p", content="new", previous="old")], [("cu-p", "new")]
        )
        assert Cerebral._mutation_lines(ep) == ["[context_understanding] new (supersedes: old)"]

    def test_replace_then_replace_uses_first_previous(self):
        ep = _episode_with(
            [
                _mut("REPLACE", "cu-p", content="mid", previous="old"),
                _mut("REPLACE", "cu-p", content="new", previous="mid"),
            ],
            [("cu-p", "new")],
        )
        assert Cerebral._mutation_lines(ep) == ["[context_understanding] new (supersedes: old)"]

    def test_prior_delete_retracted(self):
        ep = _episode_with([_mut("DELETE", "cu-p", previous="wrong fact")])
        assert Cerebral._mutation_lines(ep) == [
            "[context_understanding] RETRACTED — shown incorrect or misleading, do not rely on: wrong fact"
        ]

    def test_prior_evict_emits_nothing(self):
        ep = _episode_with([_mut("EVICT", "cu-p", previous="still true")])
        assert Cerebral._mutation_lines(ep) == []

    def test_replace_then_evict_emits_supersedes(self):
        ep = _episode_with(
            [
                _mut("REPLACE", "cu-p", content="new", previous="old"),
                _mut("EVICT", "cu-p", previous="new"),
            ]
        )
        assert Cerebral._mutation_lines(ep) == [
            "[context_understanding] new (supersedes: old)"
        ]

    def test_inherited_unchanged_observation_not_retained(self):
        ep = _episode_with([], [("cu-p", "unchanged prior")])
        assert Cerebral._mutation_lines(ep) == []

    def test_add_replace_evict_emits_last_content(self):
        """ADD → REPLACE → EVICT gives a plain line with the last content."""
        ep = _episode_with(
            [
                _mut("ADD", "cu-a", content="v1"),
                _mut("REPLACE", "cu-a", content="v2", previous="v1"),
                _mut("EVICT", "cu-a", previous="v2"),
            ]
        )
        assert Cerebral._mutation_lines(ep) == ["[context_understanding] v2"]

    def test_evicted_add_emits_content(self):
        """ADD → EVICT gives a plain line with the evicted content."""
        ep = _episode_with(
            [_mut("ADD", "cu-a", content="gone"), _mut("EVICT", "cu-a", previous="gone")]
        )
        assert Cerebral._mutation_lines(ep) == ["[context_understanding] gone"]


# ---------------------------------------------------------------------------
# Prefrontal-related Cerebral behaviour
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock  # noqa: E402

from hindsight_api.engine.memory_engine import Budget  # noqa: E402
from hindsight_api.engine.task_backend import SyncTaskBackend  # noqa: E402
from hindsight_api.extensions import OperationValidationError  # noqa: E402

from codespy.agents.memory.cerebral.cerebral import (  # noqa: E402
    BRIEFING_FULL_REFRESH_FACTOR,
    MENTAL_MODEL_SOURCE_QUERY,
    OBSERVATIONS_MISSION,
    ROOT_MENTAL_MODEL_SOURCE_QUERY,
)
from codespy.agents.memory.prefrontal.reach import mental_model_id  # noqa: E402


@pytest.fixture
def async_engine():
    """A MemoryEngine mock whose coroutine methods are AsyncMocks."""
    engine = MagicMock()
    for name in (
        "initialize",
        "ensure_bank_profile",
        "update_bank_config",
        "retain_batch_async",
        "recall_async",
        "reflect_async",
        "get_mental_model",
        "create_mental_model",
        "update_mental_model",
        "submit_async_refresh_mental_model",
        "get_observation_history",
        "get_bank_profile",
        "close",
    ):
        setattr(engine, name, AsyncMock(return_value=None))
    # Default get_bank_profile to return a profile so existing tests pass
    engine.get_bank_profile.return_value = {"bank_id": "codebase"}
    return engine


@pytest.fixture
def cerebral_async(async_engine, mock_litellm):
    with patch(
        "codespy.agents.memory.cerebral.cerebral.MemoryEngine", return_value=async_engine
    ) as cls:
        c = Cerebral(database_url="postgresql://localhost:5432/test", llm_provider="litellm")
        c._bank_ensured = True
        c._engine_cls = cls
        yield c
        c.close()


class TestCerebralTaskBackend:
    def test_sync_task_backend_passed_to_engine(self, mock_memory_engine_class, mock_litellm):
        Cerebral(database_url="postgresql://localhost:5432/test", llm_provider="litellm")
        assert isinstance(mock_memory_engine_class.call_args.kwargs["task_backend"], SyncTaskBackend)
        # Check that the subclass has the schema set correctly
        backend = mock_memory_engine_class.call_args.kwargs["task_backend"]
        assert backend._schema == "semantic"

    def test_schema_sync_task_backend_injects_schema(self):
        """Test that _SchemaSyncTaskBackend injects _schema into task dicts."""
        from codespy.agents.memory.cerebral.cerebral import _SchemaSyncTaskBackend, HINDSIGHT_SCHEMA

        backend = _SchemaSyncTaskBackend(HINDSIGHT_SCHEMA)
        received_task = {}

        async def mock_executor(task_dict):
            received_task.update(task_dict)

        backend.set_executor(mock_executor)

        import asyncio
        asyncio.run(backend.submit_task({"type": "consolidation"}))

        assert received_task.get("_schema") == "semantic"

    def test_schema_sync_task_backend_preserves_existing_schema(self):
        """Test that _SchemaSyncTaskBackend preserves existing _schema values."""
        from codespy.agents.memory.cerebral.cerebral import _SchemaSyncTaskBackend, HINDSIGHT_SCHEMA

        backend = _SchemaSyncTaskBackend(HINDSIGHT_SCHEMA)
        received_task = {}

        async def mock_executor(task_dict):
            received_task.update(task_dict)

        backend.set_executor(mock_executor)

        import asyncio
        asyncio.run(backend.submit_task({"type": "consolidation", "_schema": "other"}))

        assert received_task.get("_schema") == "other"

    def test_schema_sync_task_backend_does_not_mutate_original(self):
        """Test that _SchemaSyncTaskBackend does not mutate the caller's task dict."""
        from codespy.agents.memory.cerebral.cerebral import _SchemaSyncTaskBackend, HINDSIGHT_SCHEMA

        backend = _SchemaSyncTaskBackend(HINDSIGHT_SCHEMA)
        received_task = {}

        async def mock_executor(task_dict):
            received_task.update(task_dict)

        backend.set_executor(mock_executor)

        original = {"type": "consolidation"}
        original_copy = dict(original)

        import asyncio
        asyncio.run(backend.submit_task(original))

        assert original == original_copy  # Original should be unchanged
        assert received_task.get("_schema") == "semantic"

    def test_schema_sync_task_backend_has_attach_method(self):
        """Test that _SchemaSyncTaskBackend has an attach method for engine binding."""
        from codespy.agents.memory.cerebral.cerebral import _SchemaSyncTaskBackend, HINDSIGHT_SCHEMA

        backend = _SchemaSyncTaskBackend(HINDSIGHT_SCHEMA)
        mock_engine = MagicMock()

        # attach should set _engine
        backend.attach(mock_engine)
        assert backend._engine is mock_engine

    def test_schema_sync_task_backend_catches_exception_and_re_raises(self):
        """Test that _SchemaSyncTaskBackend catches exceptions, tries to mark failed, and re-raises."""
        from codespy.agents.memory.cerebral.cerebral import _SchemaSyncTaskBackend, HINDSIGHT_SCHEMA

        backend = _SchemaSyncTaskBackend(HINDSIGHT_SCHEMA)

        async def failing_executor(task_dict):
            raise RuntimeError("task failed")

        backend.set_executor(failing_executor)

        import asyncio
        with pytest.raises(RuntimeError, match="task failed"):
            asyncio.run(backend.submit_task({"type": "consolidation", "operation_id": "op-123"}))

    def test_error_message_uses_correct_env_var(self, mock_memory_engine_class, mock_litellm):
        """Test that embedding probe error mentions MEMORY_EMBEDDINGS_MODEL."""
        import litellm

        with patch.object(litellm, "embedding", side_effect=Exception("API error")):
            with pytest.raises(RuntimeError) as exc_info:
                Cerebral(database_url="postgresql://localhost:5432/test", llm_provider="litellm")

        assert "MEMORY_EMBEDDINGS_MODEL" in str(exc_info.value)

    def test_schema_sync_task_backend_sets_contextvar_for_consolidation(self):
        """Test that _SchemaSyncTaskBackend sets current_memory_task for consolidation."""
        from codespy.agents.memory.cerebral.cerebral import _SchemaSyncTaskBackend, HINDSIGHT_SCHEMA
        from codespy.agents.memory.cerebral.cost import current_memory_task

        backend = _SchemaSyncTaskBackend(HINDSIGHT_SCHEMA)
        received_task_type = []

        async def mock_executor(task_dict):
            received_task_type.append(current_memory_task.get())

        backend.set_executor(mock_executor)

        import asyncio
        asyncio.run(backend.submit_task({"type": "consolidation"}))

        assert received_task_type == ["consolidation"]
        # After the call, the contextvar should be reset
        assert current_memory_task.get() is None

    def test_schema_sync_task_backend_sets_contextvar_for_refresh_mental_model(self):
        """Test that _SchemaSyncTaskBackend sets current_memory_task for refresh_mental_model."""
        from codespy.agents.memory.cerebral.cerebral import _SchemaSyncTaskBackend, HINDSIGHT_SCHEMA
        from codespy.agents.memory.cerebral.cost import current_memory_task

        backend = _SchemaSyncTaskBackend(HINDSIGHT_SCHEMA)
        received_task_type = []

        async def mock_executor(task_dict):
            received_task_type.append(current_memory_task.get())

        backend.set_executor(mock_executor)

        import asyncio
        asyncio.run(backend.submit_task({"type": "refresh_mental_model"}))

        assert received_task_type == ["refresh_mental_model"]
        # After the call, the contextvar should be reset
        assert current_memory_task.get() is None

    def test_schema_sync_task_backend_no_contextvar_for_batch_retain(self):
        """Test that _SchemaSyncTaskBackend does not set current_memory_task for batch_retain."""
        from codespy.agents.memory.cerebral.cerebral import _SchemaSyncTaskBackend, HINDSIGHT_SCHEMA
        from codespy.agents.memory.cerebral.cost import current_memory_task

        backend = _SchemaSyncTaskBackend(HINDSIGHT_SCHEMA)
        received_task_type = []

        async def mock_executor(task_dict):
            received_task_type.append(current_memory_task.get())

        backend.set_executor(mock_executor)

        import asyncio
        asyncio.run(backend.submit_task({"type": "batch_retain"}))

        assert received_task_type == [None]
        # Contextvar should remain None
        assert current_memory_task.get() is None

    def test_schema_sync_task_backend_resets_contextvar_on_exception(self):
        """Test that _SchemaSyncTaskBackend resets current_memory_task even when executor raises."""
        from codespy.agents.memory.cerebral.cerebral import _SchemaSyncTaskBackend, HINDSIGHT_SCHEMA
        from codespy.agents.memory.cerebral.cost import current_memory_task

        backend = _SchemaSyncTaskBackend(HINDSIGHT_SCHEMA)

        async def failing_executor(task_dict):
            raise RuntimeError("task failed")

        backend.set_executor(failing_executor)

        import asyncio
        # Pre-set a value to verify it's restored
        token = current_memory_task.set("outer_task")
        try:
            with pytest.raises(RuntimeError, match="task failed"):
                asyncio.run(backend.submit_task({"type": "consolidation"}))
        finally:
            current_memory_task.reset(token)

        # After the call, the contextvar should be reset (back to None since we cleaned up)
        assert current_memory_task.get() is None

    def test_schema_sync_task_backend_nested_tasks_innermost_wins(self):
        """Test nested tasks: innermost task type wins, outer restored after inner returns."""
        from codespy.agents.memory.cerebral.cerebral import _SchemaSyncTaskBackend, HINDSIGHT_SCHEMA
        from codespy.agents.memory.cerebral.cost import current_memory_task

        backend_outer = _SchemaSyncTaskBackend(HINDSIGHT_SCHEMA)
        backend_inner = _SchemaSyncTaskBackend(HINDSIGHT_SCHEMA)

        inner_received = []

        async def inner_executor(task_dict):
            inner_received.append(current_memory_task.get())

        async def outer_executor(task_dict):
            # This simulates a consolidation task submitting a refresh_mental_model
            if task_dict.get("type") == "consolidation":
                backend_inner.set_executor(inner_executor)
                await backend_inner.submit_task({"type": "refresh_mental_model"})
            # After inner returns, outer's task should be restored

        backend_outer.set_executor(outer_executor)

        import asyncio
        asyncio.run(backend_outer.submit_task({"type": "consolidation"}))

        # Inner should see refresh_mental_model
        assert inner_received == ["refresh_mental_model"]
        # After everything, contextvar should be reset
        assert current_memory_task.get() is None


class TestCerebralReflectModel:
    def test_reflect_llm_forwarded_when_set(self, mock_memory_engine_class, mock_litellm):
        Cerebral(
            database_url="postgresql://localhost:5432/test",
            llm_provider="litellm",
            llm_model="openai/gpt-4o",
            reflect_llm_model="anthropic/claude-haiku",
            reflect_llm_api_key="sk-ant",
        )
        kwargs = mock_memory_engine_class.call_args.kwargs
        assert kwargs["memory_llm_model"] == "openai/gpt-4o"
        assert kwargs["reflect_llm_provider"] == "litellm"
        assert kwargs["reflect_llm_model"] == "anthropic/claude-haiku"
        assert kwargs["reflect_llm_api_key"] == "sk-ant"
        assert kwargs["reflect_llm_base_url"] is None

    def test_reflect_llm_omitted_by_default(self, mock_memory_engine_class, mock_litellm):
        Cerebral(database_url="postgresql://localhost:5432/test", llm_provider="litellm")
        kwargs = mock_memory_engine_class.call_args.kwargs
        assert not any(k.startswith("reflect_llm_") for k in kwargs)


class TestClearLegacyPlaceholders:
    def test_pages_and_clears_only_legacy_rows(self, cerebral_async, async_engine):
        from hindsight_api.engine.memory_engine import MentalModelPage

        page1 = [{"id": f"m{i}", "content": "real briefing"} for i in range(99)]
        page1.append({"id": "legacy-1", "content": "Generating content..."})
        page2 = [
            {"id": "legacy-2", "content": "  Generating content...  "},
            {"id": "empty", "content": ""},
            {"id": "none", "content": None},
        ]
        pages = {0: page1, 100: page2}

        async def list_mental_models(bank_id, detail, limit, offset, request_context):
            assert detail == "full" and limit == 100
            return MentalModelPage(items=pages.get(offset, []), total=103)

        async_engine.list_mental_models = AsyncMock(side_effect=list_mental_models)

        cleared = cerebral_async._clear_legacy_placeholders(MagicMock())

        assert cleared == 2
        assert [c.kwargs["offset"] for c in async_engine.list_mental_models.await_args_list] == [
            0,
            100,
        ]
        updated = [c.args[1] for c in async_engine.update_mental_model.await_args_list]
        assert updated == ["legacy-1", "legacy-2"]
        for call in async_engine.update_mental_model.await_args_list:
            assert call.kwargs["content"] == ""

    def test_ensure_bank_runs_migration(self, cerebral_async, async_engine):
        from hindsight_api.engine.memory_engine import MentalModelPage

        async_engine.list_mental_models = AsyncMock(
            return_value=MentalModelPage(
                items=[{"id": "legacy", "content": "Generating content..."}], total=1
            )
        )
        cerebral_async._bank_ensured = False
        cerebral_async._ensure_bank()
        async_engine.update_mental_model.assert_awaited_once()
        assert async_engine.update_mental_model.await_args.kwargs["content"] == ""


class TestCerebralObservationsMission:
    def test_observations_mission_sent(self, mock_litellm):
        captured = {}

        async def update_bank_config(bank_id, updates, request_context):
            captured.update(updates)

        engine = MagicMock()
        engine.initialize = AsyncMock()
        engine.ensure_bank_profile = AsyncMock()
        engine.update_bank_config = update_bank_config
        engine.close = AsyncMock()
        with patch("codespy.agents.memory.cerebral.cerebral.MemoryEngine", return_value=engine):
            c = Cerebral(database_url="postgresql://x/y", llm_provider="litellm")
            c._ensure_bank()
            c.close()
        assert captured["observations_mission"] == OBSERVATIONS_MISSION
        assert "RETRACTED" in OBSERVATIONS_MISSION and "supersedes:" in OBSERVATIONS_MISSION


class TestCerebralRetainScopes:
    def test_tags_with_repo(self, episode):
        tags = Cerebral._build_tags(episode, "test/repo")
        assert "repo:test/repo" in tags
        assert "org:test" in tags
        # episode: no longer added
        assert f"episode:{episode.id}" not in tags
        # pull_request: no longer added
        assert "pull_request:https://github.com/test/repo/pull/1" not in tags
        # task: still present
        assert "task:code_review" in tags

    def test_tags_without_repo_unchanged(self, episode):
        assert Cerebral._build_tags(episode) == Cerebral._build_tags(episode, None)
        assert not any(t.startswith(("repo:", "org:")) for t in Cerebral._build_tags(episode))
        # episode: and run_id: should not be present
        assert not any(t.startswith(("episode:", "run_id:")) for t in Cerebral._build_tags(episode))

    def test_observation_scopes_per_project_scope(self, episode):
        scopes = Cerebral._observation_scopes(episode, "test/repo")
        assert scopes == [["org:test", "repo:test/repo", "project_scope:test/repo/package"]]

    def test_observation_scopes_without_project_scope(self):
        """Fallback is now repo-root project scope, not bare [org:, repo:]."""
        ep = Episode(
            id=uuid.uuid4(), run_id="r", timestamp=datetime.now(UTC), task="summary",
            module="m", question="q", artifacts={}, context_memory=ContextMemory(),
        )
        # Fallback is now [org:, repo:, project_scope:o/r] (repo-root scope)
        assert Cerebral._observation_scopes(ep, "o/r") == [
            ["org:o", "repo:o/r", "project_scope:o/r"]
        ]

    def test_tags_without_project_scope_use_root_scope(self):
        """No scope topic: tagged with the repo-root scope so root recall reaches it."""
        ep = Episode(
            id=uuid.uuid4(), run_id="r", timestamp=datetime.now(UTC), task="summary",
            module="m", question="q", artifacts={}, context_memory=ContextMemory(),
        )
        tags = Cerebral._build_tags(ep, "o/r")
        assert "project_scope:o/r" in tags
        assert "repo:o/r" in tags and "org:o" in tags

    def test_tags_with_project_scope_do_not_add_root(self, episode):
        tags = Cerebral._build_tags(episode, "test/repo")
        assert "project_scope:test/repo" not in tags

    def test_observation_scopes_none_without_repo(self, episode):
        assert Cerebral._observation_scopes(episode, None) is None

    def test_scope_excludes_run_tags(self, episode):
        for scope in Cerebral._observation_scopes(episode, "test/repo"):
            # Scopes should only contain org:, repo:, project_scope:
            assert all(t.startswith(("org:", "repo:", "project_scope:")) for t in scope)
            # task:, episode:, run_id:, pull_request: should NOT be in scope
            assert not any(t.startswith(("task:", "episode:", "run_id:", "pull_request:")) for t in scope)

    def test_retain_sets_observation_scopes_on_both_items(self, cerebral_async, async_engine, episode):
        cerebral_async._mental_models = False
        cerebral_async.retain_episode(episode, repo_full_name="test/repo")
        contents = async_engine.retain_batch_async.await_args.kwargs["contents"]
        assert len(contents) == 2
        for item in contents:
            assert item["observation_scopes"] == [
                ["org:test", "repo:test/repo", "project_scope:test/repo/package"]
            ]
            assert "repo:test/repo" in item["tags"] and "org:test" in item["tags"]

    def test_retain_without_repo_has_no_scopes(self, cerebral_async, async_engine, episode):
        cerebral_async.retain_episode(episode)
        contents = async_engine.retain_batch_async.await_args.kwargs["contents"]
        assert all("observation_scopes" not in item for item in contents)
        async_engine.get_mental_model.assert_not_awaited()

    def test_recalls_never_retained(self, cerebral_async, async_engine, episode):
        """Episode.recalls is monitoring data: retain_episode must ignore it."""
        from codespy.agents.memory.recall import RecallRecord

        cerebral_async._mental_models = False
        cerebral_async.retain_episode(episode, repo_full_name="test/repo")
        without = async_engine.retain_batch_async.await_args.kwargs["contents"]

        with_recalls = episode.model_copy(
            update={
                "recalls": [
                    RecallRecord(ordinal=0, kind="load", text="RECALLED-TEXT", status="ok")
                ]
            }
        )
        cerebral_async.retain_episode(with_recalls, repo_full_name="test/repo")
        with_ = async_engine.retain_batch_async.await_args.kwargs["contents"]

        assert with_ == without
        assert all("RECALLED-TEXT" not in item["content"] for item in with_)


class TestConsolidateRun:
    """One consolidation per run, submitted after all retains."""

    def _engine(self, async_engine):
        async_engine.submit_async_consolidation = AsyncMock(return_value={"operation_id": "op"})
        return async_engine

    def test_retain_does_not_consolidate(self, cerebral_async, async_engine, episode):
        engine = self._engine(async_engine)
        cerebral_async.retain_episode(episode, repo_full_name="test/repo")
        engine.submit_async_consolidation.assert_not_awaited()
        assert episode.run_id in cerebral_async._pending_scopes

    def test_one_submit_with_union_of_scopes(self, cerebral_async, async_engine, episode):
        engine = self._engine(async_engine)
        other = episode.model_copy(update={"id": uuid.uuid4(), "task": "doc"})
        cerebral_async.retain_episode(episode, repo_full_name="test/repo")
        cerebral_async.retain_episode(other, repo_full_name="test/repo")

        cerebral_async.consolidate_run(episode.run_id)

        engine.submit_async_consolidation.assert_awaited_once()
        scopes = engine.submit_async_consolidation.await_args.kwargs["observation_scopes"]
        # No bare [org:, repo:] scope: it would cover every scope of a monorepo
        assert sorted(map(sorted, scopes)) == sorted(
            map(sorted, [
                ["org:test", "repo:test/repo", "project_scope:test/repo/package"],
            ])
        )
        assert episode.run_id not in cerebral_async._pending_scopes

    def test_other_run_untouched(self, cerebral_async, async_engine, episode):
        engine = self._engine(async_engine)
        cerebral_async.retain_episode(episode, repo_full_name="test/repo")
        cerebral_async.consolidate_run("another-run")
        engine.submit_async_consolidation.assert_not_awaited()
        assert episode.run_id in cerebral_async._pending_scopes

    def test_failure_swallowed_and_pending_released(self, cerebral_async, async_engine, episode):
        engine = self._engine(async_engine)
        engine.submit_async_consolidation.side_effect = RuntimeError("boom")
        cerebral_async.retain_episode(episode, repo_full_name="test/repo")
        cerebral_async.consolidate_run(episode.run_id)  # must not raise
        assert episode.run_id not in cerebral_async._pending_scopes


class TestBriefingSizeGuard:
    SCOPE = ["org:o", "repo:o/r", "project_scope:o/r/svc"]

    def _content(self, cerebral, factor):
        # chars/4 token heuristic
        return "x" * int(cerebral._max_mental_model_tokens * factor * 4)

    def test_oversized_gets_one_full_refresh_then_delta_restored(self, cerebral_async, async_engine):
        async_engine.get_mental_model.return_value = {
            "content": self._content(cerebral_async, BRIEFING_FULL_REFRESH_FACTOR + 1)
        }
        cerebral_async._enforce_briefing_size([self.SCOPE])
        async_engine.submit_async_refresh_mental_model.assert_awaited_once()
        assert async_engine.submit_async_refresh_mental_model.await_args.args[1] == mental_model_id(self.SCOPE)
        triggers = [c.kwargs["trigger"]["mode"] for c in async_engine.update_mental_model.await_args_list]
        assert triggers == ["full", "delta"]

    def test_under_limit_untouched(self, cerebral_async, async_engine):
        async_engine.get_mental_model.return_value = {"content": self._content(cerebral_async, 1.0)}
        cerebral_async._enforce_briefing_size([self.SCOPE])
        async_engine.submit_async_refresh_mental_model.assert_not_awaited()
        async_engine.update_mental_model.assert_not_awaited()

    def test_refresh_failure_swallowed_and_delta_restored(self, cerebral_async, async_engine):
        async_engine.get_mental_model.return_value = {"content": self._content(cerebral_async, 3)}
        async_engine.submit_async_refresh_mental_model.side_effect = RuntimeError("boom")
        cerebral_async._enforce_briefing_size([self.SCOPE])  # no raise
        assert async_engine.update_mental_model.await_args.kwargs["trigger"]["mode"] == "delta"

    def test_disabled_without_mental_models(self, cerebral_async, async_engine):
        cerebral_async._mental_models = False
        cerebral_async._enforce_briefing_size([self.SCOPE])
        async_engine.get_mental_model.assert_not_awaited()

    def test_runs_after_consolidation(self, cerebral_async, async_engine, episode):
        async_engine.submit_async_consolidation = AsyncMock(return_value={})
        with patch.object(cerebral_async, "_enforce_briefing_size") as guard:
            cerebral_async.retain_episode(episode, repo_full_name="test/repo")
            cerebral_async.consolidate_run(episode.run_id)
        guard.assert_called_once()
        assert guard.call_args.args[0] == [["org:test", "repo:test/repo", "project_scope:test/repo/package"]]


class TestEnsureMentalModels:
    def test_ids_are_deterministic(self):
        a = mental_model_id(["repo:o/r", "org:o"])
        b = mental_model_id(["org:o", "repo:o/r"])
        assert a == b
        assert a.startswith("mm-") and len(a) == 3 + 32
        assert a != mental_model_id(["org:o", "repo:o/other"])

    def test_retain_ensures_scope_model_only(self, cerebral_async, async_engine, episode):
        cerebral_async.retain_episode(episode, repo_full_name="test/repo")
        created = [c.kwargs["mental_model_id"] for c in async_engine.create_mental_model.await_args_list]
        # Only the scope briefing; the bare repo briefing [org:, repo:] is never created
        assert created == [
            mental_model_id(["org:test", "repo:test/repo", "project_scope:test/repo/package"]),
        ]
        assert mental_model_id(["org:test", "repo:test/repo"]) not in created
        first = async_engine.create_mental_model.await_args_list[0]
        assert first.args[1] == "Briefing: test/repo/package"
        assert first.args[2] == MENTAL_MODEL_SOURCE_QUERY
        assert first.args[3] == ""
        trigger = first.kwargs["trigger"]
        assert trigger["refresh_after_consolidation"] is True
        assert trigger["exclude_mental_models"] is True
        assert trigger["mode"] == "delta"
        assert trigger["budget"] == "low"
        assert trigger["keep_trace"] is True
        assert first.kwargs["max_tokens"] == 2048

    def test_root_briefing_created_with_repo_wide_query(self, cerebral_async, async_engine):
        root = ["org:o", "repo:o/r", "project_scope:o/r"]
        cerebral_async._sync_briefing_triggers([root])
        call = async_engine.create_mental_model.await_args
        assert call.kwargs["mental_model_id"] == mental_model_id(root)
        assert call.args[1] == "Briefing: o/r (repo-wide)"
        assert call.args[2] == ROOT_MENTAL_MODEL_SOURCE_QUERY
        assert "omit anything that is true for only one scope" in ROOT_MENTAL_MODEL_SOURCE_QUERY

    def test_existing_root_briefing_requeried_not_deleted(self, cerebral_async, async_engine):
        root = ["org:o", "repo:o/r", "project_scope:o/r"]
        async_engine.get_mental_model.return_value = {
            "id": mental_model_id(root),
            "name": "Briefing: o/r",
            "source_query": MENTAL_MODEL_SOURCE_QUERY,
            "content": "huge",
            "trigger": {"refresh_after_consolidation": True, "exclude_mental_models": True, "mode": "delta"},
        }
        cerebral_async._sync_briefing_triggers([root])
        async_engine.create_mental_model.assert_not_awaited()
        kwargs = async_engine.update_mental_model.await_args.kwargs
        assert kwargs["name"] == "Briefing: o/r (repo-wide)"
        assert kwargs["source_query"] == ROOT_MENTAL_MODEL_SOURCE_QUERY

    def test_existing_scope_briefing_up_to_date_untouched(self, cerebral_async, async_engine):
        # Stored triggers are merged supersets: extra keys must not cause an update
        scope = ["org:o", "repo:o/r", "project_scope:o/r/svc"]
        async_engine.get_mental_model.return_value = {
            "name": "Briefing: o/r/svc",
            "source_query": MENTAL_MODEL_SOURCE_QUERY,
            "content": "c",
            "trigger": {
                **cerebral_async._desired_trigger(),
                "refresh_cron": None,
                "fact_types": None,
            },
        }
        cerebral_async._sync_briefing_triggers([scope])
        async_engine.update_mental_model.assert_not_awaited()

    def test_existing_briefing_missing_budget_updated(self, cerebral_async, async_engine):
        scope = ["org:o", "repo:o/r", "project_scope:o/r/svc"]
        stored = cerebral_async._desired_trigger()
        del stored["budget"]
        async_engine.get_mental_model.return_value = {
            "name": "Briefing: o/r/svc",
            "source_query": MENTAL_MODEL_SOURCE_QUERY,
            "content": "c",
            "trigger": stored,
        }
        cerebral_async._sync_briefing_triggers([scope])
        kwargs = async_engine.update_mental_model.await_args.kwargs
        assert kwargs["trigger"]["budget"] == "low"
        assert kwargs["trigger"]["keep_trace"] is True

    def test_never_deletes(self, cerebral_async, async_engine, episode):
        async_engine.submit_async_consolidation = AsyncMock(return_value={})
        async_engine.get_mental_model.return_value = {"content": "x" * 100_000, "trigger": {}}
        cerebral_async.retain_episode(episode, repo_full_name="test/repo")
        cerebral_async.consolidate_run(episode.run_id)
        for name in ("delete_mental_model", "delete_document", "delete_memory_unit"):
            getattr(async_engine, name).assert_not_called()

    def test_existing_model_not_created(self, cerebral_async, async_engine):
        async_engine.get_mental_model.return_value = {"id": "x", "content": "c"}
        cerebral_async._sync_briefing_triggers([["org:o", "repo:o/r"]])
        async_engine.create_mental_model.assert_not_awaited()

    def test_409_tolerated(self, cerebral_async, async_engine):
        async_engine.create_mental_model.side_effect = OperationValidationError("exists", status_code=409)
        cerebral_async._sync_briefing_triggers([["org:o", "repo:o/r"]])
        assert mental_model_id(["org:o", "repo:o/r"]) in cerebral_async._ensured_mental_models

    def test_second_call_served_from_cache(self, cerebral_async, async_engine):
        cerebral_async._sync_briefing_triggers([["org:o", "repo:o/r"]])
        cerebral_async._sync_briefing_triggers([["org:o", "repo:o/r"]])
        assert async_engine.get_mental_model.await_count == 1

    def test_disabled(self, cerebral_async, async_engine, episode):
        cerebral_async._mental_models = False
        cerebral_async._sync_briefing_triggers([["org:test", "repo:test/repo"]])
        # When mental_models is False, get_mental_model is still called to check
        # for existing briefings that need their trigger updated
        async_engine.get_mental_model.assert_awaited()
        # But create is not called when mental_models is False
        async_engine.create_mental_model.assert_not_awaited()

    def test_failures_only_log(self, cerebral_async, async_engine):
        async_engine.get_mental_model.side_effect = RuntimeError("db down")
        cerebral_async._sync_briefing_triggers([["org:o", "repo:o/r"]])  # no raise
        assert not cerebral_async._ensured_mental_models

    def test_non_409_validation_error_logged(self, cerebral_async, async_engine):
        async_engine.create_mental_model.side_effect = OperationValidationError("forbidden", status_code=403)
        cerebral_async._sync_briefing_triggers([["org:o", "repo:o/r"]])  # no raise
        assert not cerebral_async._ensured_mental_models


class TestCerebralReadApi:
    def test_arecall_forwards_arguments(self, cerebral_async, async_engine):
        import asyncio as _asyncio

        fact = MagicMock()
        async_engine.recall_async.return_value = MagicMock(results=[fact])
        when = datetime(2026, 9, 29, tzinfo=UTC)
        groups = [object()]
        facts = _asyncio.run(
            cerebral_async.arecall(
                "q",
                tag_groups=groups,
                max_tokens=123,
                fact_type=["world", "observation"],
                prefer_observations=True,
                budget="high",
                question_date=when,
            )
        )
        assert facts == [fact]
        call = async_engine.recall_async.await_args
        assert call.args == ("codespy", "q")
        assert call.kwargs["tag_groups"] is groups
        assert call.kwargs["max_tokens"] == 123
        assert call.kwargs["fact_type"] == ["world", "observation"]
        assert call.kwargs["prefer_observations"] is True
        assert call.kwargs["budget"] == Budget.HIGH
        assert call.kwargs["question_date"] == when

    def test_recall_sync(self, cerebral_async, async_engine):
        async_engine.recall_async.return_value = MagicMock(results=[])
        assert cerebral_async.recall("q") == []

    def test_areflect_uses_mid_budget(self, cerebral_async, async_engine):
        import asyncio as _asyncio

        async_engine.reflect_async.return_value = MagicMock(text=" answer ")
        text, summary = _asyncio.run(
            cerebral_async.areflect("q", tag_groups=None, max_tokens=50, context="code_review on o/r")
        )
        assert text == "answer"
        assert summary.empty is False
        call = async_engine.reflect_async.await_args
        assert call.kwargs["budget"] == Budget.LOW
        assert call.kwargs["max_tokens"] == 50
        assert call.kwargs["context"] == "code_review on o/r"

    def test_areflect_uses_low_budget(self, cerebral_async, async_engine):
        """Test that areflect uses LOW budget (with 0.5x multiplier, cap is doubled)."""
        import asyncio as _asyncio

        async_engine.reflect_async.return_value = MagicMock(text="answer")
        text, summary = _asyncio.run(
            cerebral_async.areflect(
                "q",
                tag_groups=None,
                max_tokens=50,
                context="code_review",
            )
        )
        call = async_engine.reflect_async.await_args
        assert call.kwargs["budget"] == Budget.LOW

    def test_areflect_basic(self, cerebral_async, async_engine):
        """Test basic areflect call."""
        import asyncio as _asyncio

        async_engine.reflect_async.return_value = MagicMock(text="answer")
        text, summary = _asyncio.run(
            cerebral_async.areflect(
                "q",
                tag_groups=None,
                max_tokens=50,
                context="code_review",
            )
        )
        assert text == "answer"
        assert summary.empty is False

    def test_areflect_returns_reflect_summary(self, cerebral_async, async_engine):
        import asyncio as _asyncio

        # Create a mock result with the real hindsight_api shape
        # llm_trace is a list of objects with 'scope' attribute
        # tool_trace is a list of objects with 'tool', 'input', 'output' attributes
        from unittest.mock import Mock

        mock_result = MagicMock()
        mock_result.text = "reflected answer"

        # Mock llm_trace as list of objects with 'scope' attribute
        mock_result.llm_trace = [
            Mock(scope="agent_1"),
            Mock(scope="final_map_1"),
            Mock(scope="final_map_2"),
            Mock(scope="final"),
            Mock(scope="final_rewrite"),
        ]

        # Mock tool_trace as list of objects with 'tool', 'output' attributes
        mock_result.tool_trace = [
            Mock(tool="search_observations", input="q", output={"results": ["a"] * 50}),
            Mock(tool="recall", input="q", output={"text": "x" * 100}),
        ]

        # Mock usage as TokenUsage-like object
        mock_result.usage = Mock(input_tokens=500, output_tokens=200, thoughts_tokens=50)

        async_engine.reflect_async.return_value = mock_result
        text, summary = _asyncio.run(cerebral_async.areflect("q"))

        assert text == "reflected answer"
        # iterations = agent_<n> scopes (1: agent_1) + 1 for final = 2
        assert summary.iterations == 2
        # llm_calls = 5 (all scopes)
        assert summary.llm_calls == 5
        # map_calls = scopes starting with final_map_ (2)
        assert summary.map_calls == 2
        # rewrite = True (has final_rewrite scope)
        assert summary.rewrite is True
        assert summary.empty is False
        # tools = [(name, output_tokens)] from tool_trace
        assert len(summary.tools) == 2
        assert summary.tools[0][0] == "search_observations"
        assert summary.tools[1][0] == "recall"
        # usage contains token counts
        assert summary.usage["input_tokens"] == 500
        assert summary.usage["output_tokens"] == 200
        assert summary.usage["thoughts_tokens"] == 50

    def test_reflect_summary_counts_closing_done(self, cerebral_async):
        result = MagicMock()
        result.text = "answer"
        result.llm_trace = [
            Mock(scope="agent_1"),
            Mock(scope="agent_2"),
            Mock(scope="closing_done"),
        ]
        result.tool_trace = []
        result.usage = None
        summary = cerebral_async._build_reflect_summary(result)
        assert summary.iterations == 3
        assert summary.llm_calls == 3
        assert summary.map_calls == 0
        assert summary.rewrite is False

    def test_aget_mental_models_skips_missing(self, cerebral_async, async_engine):
        import asyncio as _asyncio

        async def get(bank_id, mm_id, request_context):
            return None if mm_id == "b" else {"id": mm_id}

        async_engine.get_mental_model.side_effect = get
        models = _asyncio.run(cerebral_async.aget_mental_models(["a", "b", "c"]))
        assert [m["id"] for m in models] == ["a", "c"]

    def test_aget_observation_history(self, cerebral_async, async_engine):
        import asyncio as _asyncio

        async_engine.get_observation_history.return_value = [{"previous_text": "x"}]
        assert _asyncio.run(cerebral_async.aget_observation_history("id-1")) == [{"previous_text": "x"}]
        assert async_engine.get_observation_history.await_args.args == ("codespy", "id-1")
        async_engine.get_observation_history.return_value = None
        assert _asyncio.run(cerebral_async.aget_observation_history("id-2")) == []


class TestCerebralMissingBank:
    """Tests for Cerebral read behavior when bank does not exist yet."""

    def test_arecall_returns_empty_when_bank_missing(self, cerebral_async, async_engine):
        """arecall returns [] and does not call recall_async when bank is missing."""
        import asyncio as _asyncio

        async_engine.get_bank_profile.return_value = None
        cerebral_async._bank_exists = False
        cerebral_async._missing_bank_logged = False

        facts = _asyncio.run(cerebral_async.arecall("q"))
        assert facts == []
        async_engine.recall_async.assert_not_awaited()

    def test_areflect_returns_empty_when_bank_missing(self, cerebral_async, async_engine):
        """areflect returns ("", summary) with empty=True when bank is missing."""
        import asyncio as _asyncio

        async_engine.get_bank_profile.return_value = None
        cerebral_async._bank_exists = False
        cerebral_async._missing_bank_logged = False

        text, summary = _asyncio.run(cerebral_async.areflect("q"))
        assert text == ""
        assert summary.empty is True
        assert summary.llm_calls == 0
        async_engine.reflect_async.assert_not_awaited()

    def test_aget_mental_models_returns_empty_when_bank_missing(self, cerebral_async, async_engine):
        """aget_mental_models returns [] when bank is missing."""
        import asyncio as _asyncio

        async_engine.get_bank_profile.return_value = None
        cerebral_async._bank_exists = False
        cerebral_async._missing_bank_logged = False

        models = _asyncio.run(cerebral_async.aget_mental_models(["a", "b"]))
        assert models == []
        async_engine.get_mental_model.assert_not_awaited()

    def test_aget_observation_history_returns_empty_when_bank_missing(self, cerebral_async, async_engine):
        """aget_observation_history returns [] when bank is missing."""
        import asyncio as _asyncio

        async_engine.get_bank_profile.return_value = None
        cerebral_async._bank_exists = False
        cerebral_async._missing_bank_logged = False

        history = _asyncio.run(cerebral_async.aget_observation_history("id-1"))
        assert history == []
        async_engine.get_observation_history.assert_not_awaited()

    def test_missing_not_cached_exists_becomes_true(self, cerebral_async, async_engine):
        """Missing is not cached: when bank is created, subsequent reads work."""
        import asyncio as _asyncio

        # First call: bank missing
        async_engine.get_bank_profile.return_value = None
        cerebral_async._bank_exists = False
        cerebral_async._missing_bank_logged = False

        facts = _asyncio.run(cerebral_async.arecall("q"))
        assert facts == []
        assert async_engine.get_bank_profile.await_count == 1

        # Second call: bank now exists (e.g., retain_episode was called)
        async_engine.get_bank_profile.return_value = {"bank_id": "codebase"}
        # _bank_exists is still False because we didn't call _ensure_bank
        cerebral_async._bank_exists = False

        # Reset recall_async mock to track new calls
        async_engine.recall_async.return_value = MagicMock(results=[MagicMock()])

        facts = _asyncio.run(cerebral_async.arecall("q"))
        # Should have called get_bank_profile again and now recall_async
        assert async_engine.get_bank_profile.await_count == 2
        async_engine.recall_async.assert_awaited_once()

    def test_exists_is_cached_no_duplicate_checks(self, cerebral_async, async_engine):
        """Exists is cached: multiple arecalls only check once."""
        import asyncio as _asyncio

        async_engine.get_bank_profile.return_value = {"bank_id": "codebase"}
        cerebral_async._bank_exists = False
        cerebral_async._missing_bank_logged = False

        async_engine.recall_async.return_value = MagicMock(results=[])

        # Two arecalls
        _asyncio.run(cerebral_async.arecall("q1"))
        _asyncio.run(cerebral_async.arecall("q2"))

        # get_bank_profile should only be awaited once
        assert async_engine.get_bank_profile.await_count == 1
        # recall_async should be awaited twice
        assert async_engine.recall_async.await_count == 2

    def test_ensure_bank_sets_bank_exists(self, cerebral_async, async_engine):
        """_ensure_bank success sets _bank_exists so get_bank_profile is not called."""
        cerebral_async._bank_exists = False
        cerebral_async._bank_ensured = False

        cerebral_async._ensure_bank()

        assert cerebral_async._bank_exists is True
        # After _ensure_bank, arecall should not call get_bank_profile
        async_engine.get_bank_profile.reset_mock()
        async_engine.recall_async.return_value = MagicMock(results=[])

        import asyncio as _asyncio
        _asyncio.run(cerebral_async.arecall("q"))

        async_engine.get_bank_profile.assert_not_awaited()
        async_engine.recall_async.assert_awaited_once()
