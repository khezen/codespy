"""Cerebral – Hindsight semantic memory for episodes.

Uses MemoryEngine (direct Python API) sharing the same PostgreSQL
instance as the episodic store, under the ``semantic`` schema.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import TYPE_CHECKING

from hindsight_api import MemoryEngine
from hindsight_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hindsight_api.engine.embeddings import LiteLLMSDKEmbeddings
from hindsight_api.extensions.builtin.tenant import DefaultTenantExtension
from hindsight_api.models import RequestContext

from codespy.agents.memory.cerebral.cost import (
    CerebralCostRecorder,
    MeteredLiteLLMSDKEmbeddings,
    register_cerebral_cost_recorder,
)

if TYPE_CHECKING:
    from codespy.agents.memory.hippocampus.episode import Episode

logger = logging.getLogger(__name__)

HINDSIGHT_SCHEMA = "semantic"


class Cerebral:
    """Retains episode observations and artifacts in Hindsight semantic memory.

    Uses MemoryEngine directly — no HTTP server. All calls are in-process
    against the same PostgreSQL instance under the ``semantic`` schema.
    """

    def __init__(
        self,
        database_url: str,
        llm_provider: str,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        bank_id: str = "codespy",
        embeddings_model: str = "openai/text-embedding-3-small",
    ):
        # Fail fast: test the embedding model before building MemoryEngine.
        # A bad model name, missing creds, or unavailable region surfaces here
        # with a clear error instead of buried inside MemoryEngine.initialize().
        import litellm
        try:
            litellm.embedding(model=embeddings_model, input=["test"], timeout=15)
        except Exception as exc:
            raise RuntimeError(
                f"Cerebral: embedding model {embeddings_model!r} is not available. "
                f"Set MEMORY_HINDSIGHT_EMBEDDINGS_MODEL to a valid litellm embedding model. "
                f"Error: {exc}"
            ) from exc

        # Start a dedicated event loop in a background thread.
        # The MemoryEngine's asyncpg pool binds to this loop; it must
        # outlive every operation, so we never close it until close().
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="cerebral-loop"
        )
        self._loop_thread.start()

        # Register cost recorder before initializing MemoryEngine so
        # all LLM calls (including the probe) are captured.
        cost_recorder = register_cerebral_cost_recorder()

        # Use metered embeddings that capture usage costs
        base_embeddings = LiteLLMSDKEmbeddings(
            model=embeddings_model,
            api_key=None,  # litellm reads from env
        )
        metered_embeddings = MeteredLiteLLMSDKEmbeddings(
            base=base_embeddings,
            cost_recorder=cost_recorder,
        )

        self._engine = MemoryEngine(
            db_url=database_url,
            memory_llm_provider=llm_provider,
            memory_llm_model=llm_model,
            memory_llm_api_key=llm_api_key,
            memory_llm_base_url=llm_base_url or None,
            embeddings=metered_embeddings,
            cross_encoder=RRFPassthroughCrossEncoder(),
            tenant_extension=DefaultTenantExtension(config={"schema": HINDSIGHT_SCHEMA}),
            skip_llm_verification=True,
        )
        self._run_async(self._engine.initialize())

        self._bank_id = bank_id
        self._bank_ensured = False

        logger.info(
            "Cerebral MemoryEngine initialized (schema=%s, bank=%s, embeddings=%s, provider=%s)",
            HINDSIGHT_SCHEMA, bank_id, embeddings_model, llm_provider,
        )

    def _run_async(self, coro):
        """Submit a coroutine to the dedicated event loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        """Shut down the MemoryEngine and stop the event loop."""
        try:
            self._run_async(self._engine.close())
        except Exception:
            logger.debug("cerebral: error closing engine", exc_info=True)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=10)

    def _ensure_bank(self) -> None:
        """Create the Hindsight bank and configure it (idempotent)."""
        if self._bank_ensured:
            return
        ctx = RequestContext()
        try:
            self._run_async(
                self._engine.ensure_bank_profile(
                    self._bank_id,
                    request_context=ctx,
                )
            )
        except Exception:
            logger.debug("cerebral: bank %s profile ensure failed", self._bank_id)

        try:
            self._run_async(
                self._engine.update_bank_config(
                    self._bank_id,
                    updates={
                        "retain_extraction_mode": "concise",
                        "retain_mission": (
                            "Retain code review observations, analysis results, and artifacts. "
                            "Focus on patterns, architectural decisions, dependency relationships, "
                            "security findings, and technical insights across repositories."
                        ),
                    },
                    request_context=ctx,
                )
            )
        except Exception:
            logger.warning("cerebral: failed to configure bank %s", self._bank_id, exc_info=True)
        self._bank_ensured = True

    def retain_episode(self, episode: Episode) -> None:
        """Retain all observations and artifacts from an episode."""
        self._ensure_bank()

        tags = self._build_tags(episode)
        episode_doc_id = f"episode-{episode.id}"

        # Build contents list: one dict per observation + one per artifact
        # Each dict has 'content', 'context', 'tags', 'document_id', 'event_date'
        contents: list[dict] = []

        for section_name in episode.context_memory.section_names():
            for obs in getattr(episode.context_memory, section_name):
                contents.append({
                    "content": f"[{section_name}] {obs.content}",
                    "context": f"{episode.task}: {episode.question}: observation ({section_name})",
                    "tags": tags,
                    "document_id": episode_doc_id,
                    "event_date": episode.timestamp.isoformat(),
                })

        for name, content in (episode.artifacts or {}).items():
            contents.append({
                "content": content,
                "context": f"{episode.task}: {episode.question}: artifact ({name})",
                "tags": tags,
                "document_id": episode_doc_id,
                "event_date": episode.timestamp.isoformat(),
            })

        if not contents:
            logger.debug("cerebral: no items for episode %s", episode.id)
            return

        try:
            self._run_async(
                self._engine.retain_batch_async(
                    bank_id=self._bank_id,
                    contents=contents,
                    request_context=RequestContext(),
                )
            )
            logger.info(
                "cerebral: retained %d items for episode %s (bank=%s, task=%s)",
                len(contents), episode.id, self._bank_id, episode.task,
            )
        except Exception:
            logger.warning(
                "cerebral: failed to retain episode %s",
                episode.id, exc_info=True,
            )

    @staticmethod
    def _build_tags(episode: Episode) -> list[str]:
        tags: list[str] = []
        for topic in episode.context_memory.topics:
            tags.append(f"{topic.type}:{topic.id}")
        tags.append(f"episode:{episode.id}")
        tags.append(f"task:{episode.task}")
        tags.append(f"run_id:{episode.run_id}")
        return tags
