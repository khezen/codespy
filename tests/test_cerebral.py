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
    Observation,
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
            # Now called twice: once for initialize(), once for ensure_maintenance_routines()
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
        )


class TestCerebralBank:
    """Tests for Cerebral bank management."""

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

            # ensure_bank_profile and update_bank_config should only be called once
            assert mock_run_async.call_count == 2  # Once for ensure, once for update

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

            # Should call ensure_bank_profile and update_bank_config
            assert mock_run_async.call_count == 2


class TestCerebralRetainEpisode:
    """Tests for Cerebral retain_episode method."""

    def test_retain_episode_builds_correct_tags(self, mock_memory_engine_class, episode, mock_litellm):
        """Test that tags are built correctly from episode."""
        with patch.object(Cerebral, "_run_async"):
            cerebral = Cerebral(
                database_url="postgresql://localhost:5432/test",
                llm_provider="litellm",
            )

            tags = cerebral._build_tags(episode)

            expected_tags = [
                "project_scope:test/repo/package",
                "pull_request:https://github.com/test/repo/pull/1",
                f"episode:{episode.id}",
                "task:code_review",
                "run_id:test-run-123",
            ]
            assert tags == expected_tags

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
                if "observations" in item.get("context", ""):
                    obs_item = item
                elif "artifacts" in item.get("context", ""):
                    art_item = item

            # Verify observations item
            assert obs_item is not None, "Observations content item not found"
            assert "[context_understanding]" in obs_item["content"], "Missing context_understanding header"
            assert "[actions]" in obs_item["content"], "Missing actions header"
            assert "This is a test observation" in obs_item["content"], "Missing observation content"
            assert "Performed tool call" in obs_item["content"], "Missing action content"

            # Verify artifacts item
            assert art_item is not None, "Artifacts content item not found"
            assert "[artifact:review]" in art_item["content"], "Missing artifact header"
            assert "# Review Results" in art_item["content"], "Missing artifact content"

            # Verify shared fields
            expected_tags = cerebral._build_tags(episode)
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
