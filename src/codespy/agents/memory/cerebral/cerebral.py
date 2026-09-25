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

# Module constants (must be defined before the Hindsight import)
HINDSIGHT_SCHEMA_DEFAULT = "semantic"

# Hindsight defaults embeddings_provider / reranker_provider to "local", which
# needs sentence-transformers and emits a misleading startup warning
# (hindsight_api/config.py _validate). That config is built and cached at IMPORT
# time — the `from hindsight_api import MemoryEngine` below pulls in
# engine.llm_wrapper, whose module-level _get_raw_config() calls
# HindsightConfig.from_env(). So these env vars MUST be set before that import
# line, not in Cerebral.__init__ (too late). We inject our own LiteLLM-SDK
# embeddings and an RRF-passthrough cross-encoder in Cerebral.__init__, so the
# env-driven providers are never used; align them to non-local values so the
# warning does not fire. setdefault keeps operator overrides intact.
os.environ.setdefault("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "litellm-sdk")
os.environ.setdefault("HINDSIGHT_API_RERANKER_PROVIDER", "none")

# Upstream #2638: The maintenance routines (mental_models_with_cron, etc.) are
# installed by migrations into `database_schema` and called by name from that
# schema via fq_routine(). Cerebral sets its schema only via the tenant
# extension, so without this env var, `database_schema` defaults to "public".
# With database_schema="public" and migrations running with target_schema="semantic",
# the routines are never created. Setting this makes the routines install into
# "semantic" and be called from "semantic", matching the tenant schema.
os.environ.setdefault("HINDSIGHT_API_DATABASE_SCHEMA", HINDSIGHT_SCHEMA_DEFAULT)

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
from codespy.agents.memory.cerebral.routines import ensure_maintenance_routines

if TYPE_CHECKING:
    from codespy.agents.memory.hippocampus.episode import Episode

logger = logging.getLogger(__name__)

HINDSIGHT_SCHEMA = HINDSIGHT_SCHEMA_DEFAULT

# Warn if the operator has overridden HINDSIGHT_API_DATABASE_SCHEMA to a different
# value than our schema — the routines would then be missing again (upstream #2638).
if os.environ.get("HINDSIGHT_API_DATABASE_SCHEMA") != HINDSIGHT_SCHEMA:
    logger.warning(
        "HINDSIGHT_API_DATABASE_SCHEMA is set to %r, which differs from the "
        "expected schema %r. Maintenance routines may not be found. "
        "See upstream #2638 for details.",
        os.environ.get("HINDSIGHT_API_DATABASE_SCHEMA"),
        HINDSIGHT_SCHEMA,
    )


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

        # One-time repair: install maintenance routines if missing.
        # This is idempotent and only needed for databases that were migrated
        # before HINDSIGHT_API_DATABASE_SCHEMA was set to "semantic" (upstream #2638).
        self._run_async(ensure_maintenance_routines(self._engine, HINDSIGHT_SCHEMA))

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
                        # retain_chunk_size (chars) MUST be strictly less than
                        # retain_max_completion_tokens (default 64000): Hindsight's
                        # validate_retain_completion_token_budget compares the two values
                        # directly (chars vs tokens is NOT reconciled), and a bank-config
                        # update that violates it is rejected wholesale — silently reverting
                        # the bank to defaults (retain_chunk_size=3000). Do not raise this at
                        # or above 64000. The merged observations blob is bounded by
                        # max_context_memory_tokens (16384 tokens ≈ ~65K chars worst case), so
                        # at 12288 a large blob splits into several chunks (graceful, no error).
                        "retain_chunk_size": 12288,
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

        # Build contents list: at most TWO items per episode
        # One merged observations blob + one merged artifacts blob
        # This reduces LLM call count by bundling items that share tags/document_id/event_date
        contents: list[dict] = []

        # Merge all observations into a single content item
        obs_lines: list[str] = []
        for section_name in episode.context_memory.section_names():
            for obs in getattr(episode.context_memory, section_name):
                obs_lines.append(f"[{section_name}] {obs.content}")
        if obs_lines:
            contents.append({
                "content": "\n\n".join(obs_lines),
                "context": f"{episode.task}: {episode.question}: observations",
                "tags": tags,
                "document_id": episode_doc_id,
                "event_date": episode.timestamp.isoformat(),
            })

        # Merge all artifacts into a single content item
        artifact_lines: list[str] = []
        for name, content in (episode.artifacts or {}).items():
            artifact_lines.append(f"[artifact:{name}] {content}")
        if artifact_lines:
            contents.append({
                "content": "\n\n".join(artifact_lines),
                "context": f"{episode.task}: {episode.question}: artifacts",
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
                "cerebral: retained %d content blobs for episode %s (bank=%s, task=%s)",
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
