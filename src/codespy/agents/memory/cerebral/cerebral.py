"""Cerebral – Hindsight semantic memory for episodes.

Uses MemoryEngine (direct Python API) sharing the same PostgreSQL
instance as the episodic store, under the ``semantic`` schema.

Write side: ``retain_episode`` retains Hippocampus episodes, tagged with
``repo:``/``org:`` and consolidated per observation scope, then keeps one
mental model ("briefing") per scope. With ``SyncTaskBackend`` consolidation
and mental-model refreshes run inline, on the caller's (background save)
thread.

Read side (used by Prefrontal only): ``arecall``/``recall``,
``areflect``/``reflect``, ``aget_mental_models`` and
``aget_observation_history``. Cerebral never reads Prefrontal output.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

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
from hindsight_api.engine.memory_engine import Budget
from hindsight_api.engine.task_backend import SyncTaskBackend
from hindsight_api.extensions import OperationValidationError
from hindsight_api.extensions.builtin.tenant import DefaultTenantExtension
from hindsight_api.models import RequestContext


class _SchemaSyncTaskBackend(SyncTaskBackend):
    """SyncTaskBackend that tags inline tasks with Cerebral's schema.

    Hindsight only sets ``_schema`` in the worker poller. Without it,
    ``_execute_task`` passes ``schema=None`` to the consolidation webhook,
    which then queries bare ``webhooks`` (search_path -> public) and fails with
    'relation "webhooks" does not exist'.

    The root cause upstream is ``fq_table_explicit(table, None)`` returning
    a bare table name instead of falling back to ``database_schema`` like
    ``fq_table`` and ``fq_routine`` do. This subclass can be removed once
    Hindsight falls back to ``database_schema`` there.
    """

    def __init__(self, schema: str):
        super().__init__()
        self._schema = schema

    async def submit_task(self, task_dict):
        if "_schema" not in task_dict:
            task_dict = {**task_dict, "_schema": self._schema}
        await super().submit_task(task_dict)

from codespy.agents.memory.cerebral.cost import (
    CerebralCostRecorder,
    MeteredLiteLLMSDKEmbeddings,
    register_cerebral_cost_recorder,
)
from codespy.agents.memory.cerebral.routines import ensure_maintenance_routines
from codespy.agents.memory.prefrontal.reach import (
    TAG_PROJECT_SCOPE,
    mental_model_id,
    repo_tags,
    scope_tags,
)

if TYPE_CHECKING:
    from codespy.agents.memory.hippocampus.episode import Episode

logger = logging.getLogger(__name__)


@dataclass
class ReflectSummary:
    """Summary of a reflect call for monitoring and debugging.

    Attributes:
        iterations: Number of reflect iterations performed.
        llm_calls: Total LLM calls (map calls + reduce + rewrite).
        map_calls: Number of map calls in split synthesis (0 when history fits in one chunk).
        rewrite: Whether a final rewrite occurred.
        tools: List of (tool_name, output_tokens) tuples. Output size is estimated
            in tokens (chars/4 heuristic) from the JSON-serialized tool output.
        usage: Token usage dict with input_tokens, output_tokens, thoughts_tokens.
        empty: Whether the result text is empty.
    """

    iterations: int = 0
    llm_calls: int = 0
    map_calls: int = 0
    rewrite: bool = False
    tools: list[tuple[str, int]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    empty: bool = True

HINDSIGHT_SCHEMA = HINDSIGHT_SCHEMA_DEFAULT

# Content Hindsight's HTTP API gives a mental model before its first refresh.
# Prefrontal skips briefings that still hold it.
MENTAL_MODEL_PLACEHOLDER = "Generating content..."

# Generic (not code-review specific) question each briefing answers.
MENTAL_MODEL_SOURCE_QUERY = (
    "What should an agent starting work here know so it does not rediscover it: "
    "structure and where things live, conventions, invariants and constants, "
    "dependencies and integrations, pitfalls and recurring problems, facts shown to be wrong."
)

OBSERVATIONS_MISSION = (
    "Merge facts about the same subject into one observation. Keep details that matter only "
    "to one kind of work (e.g. documentation, security, dependencies) instead of generalizing "
    "them away. When a fact is marked 'supersedes:', update the observation; when marked "
    "'RETRACTED', mark the observation invalid."
)

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
        retain_chunk_size: int = 12288,
        mental_models: bool = True,
        max_mental_model_tokens: int = 2048,
        reflect_llm_model: str | None = None,
        reflect_llm_api_key: str | None = None,
        reflect_llm_base_url: str | None = None,
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

        # Separate reflect LLM (Prefrontal reflect + mental-model refresh), only
        # when configured; otherwise Hindsight falls back to memory_llm_*.
        reflect_kwargs: dict[str, str | None] = {}
        if reflect_llm_model:
            reflect_kwargs = {
                "reflect_llm_provider": llm_provider,
                "reflect_llm_model": reflect_llm_model,
                "reflect_llm_api_key": reflect_llm_api_key,
                "reflect_llm_base_url": reflect_llm_base_url or None,
            }

        self._engine = MemoryEngine(
            db_url=database_url,
            memory_llm_provider=llm_provider,
            memory_llm_model=llm_model,
            memory_llm_api_key=llm_api_key,
            memory_llm_base_url=llm_base_url or None,
            **reflect_kwargs,
            embeddings=metered_embeddings,
            cross_encoder=RRFPassthroughCrossEncoder(),
            tenant_extension=DefaultTenantExtension(config={"schema": HINDSIGHT_SCHEMA}),
            skip_llm_verification=True,
            # Run consolidation and mental-model refreshes inline. The default
            # BrokerTaskBackend only queues rows for a WorkerPoller, which
            # codespy never starts, so they would never run.
            task_backend=_SchemaSyncTaskBackend(HINDSIGHT_SCHEMA),
        )
        self._run_async(self._engine.initialize())

        # One-time repair: install maintenance routines if missing.
        # This is idempotent and only needed for databases that were migrated
        # before HINDSIGHT_API_DATABASE_SCHEMA was set to "semantic" (upstream #2638).
        self._run_async(ensure_maintenance_routines(self._engine, HINDSIGHT_SCHEMA))

        self._bank_id = bank_id
        self._bank_ensured = False
        self._bank_exists = False
        self._missing_bank_logged = False
        self._retain_chunk_size = retain_chunk_size
        self._mental_models = mental_models
        self._max_mental_model_tokens = max_mental_model_tokens
        # Mental-model ids already ensured by this process.
        self._ensured_mental_models: set[str] = set()
        # Lock for consolidation serialization
        self._consolidation_lock = threading.Lock()

        logger.info(
            "Cerebral MemoryEngine initialized (schema=%s, bank=%s, embeddings=%s, provider=%s, chunk_size=%s)",
            HINDSIGHT_SCHEMA, bank_id, embeddings_model, llm_provider, retain_chunk_size,
        )

    def _run_async(self, coro):
        """Submit a coroutine to the dedicated event loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _await(self, coro):
        """Await a coroutine on the dedicated loop without blocking the caller's loop.

        When already running on the dedicated loop, the coroutine is awaited
        directly (``run_coroutine_threadsafe`` + wait would deadlock there).
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            return await coro
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self._loop))

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
            self._bank_exists = True
        except Exception:
            logger.debug("cerebral: bank %s profile ensure failed", self._bank_id)

        try:
            self._run_async(
                self._engine.update_bank_config(
                    self._bank_id,
                    updates={
                        "retain_extraction_mode": "concise",
                        # See CerebralRetainConfig.chunk_size for constraint details.
                        "retain_chunk_size": self._retain_chunk_size,
                        "retain_mission": (
                            "Retain durable knowledge learned while working on a task, whatever the domain. "
                            "Focus on how the subject is structured, the entities involved and how they relate, "
                            "domain rules, constraints and constants, procedures that proved effective, "
                            "data formats, and findings or results worth reusing later. "
                            "Each line starts with a [section] label naming the kind of knowledge, "
                            "or [artifact:name] for a produced output. "
                            "State each fact about the subject itself, keep names, identifiers, quantities "
                            "and references exact, and skip transient process notes that will not hold beyond this run.\n"
                            "Observation lines use these markers:\n"
                            "- A plain line is a newly learned fact.\n"
                            "- '(supersedes: X)': the line is the current wording of a fact previously worded X. "
                            "Update that fact; do not keep both.\n"
                            "- 'RETRACTED — shown incorrect or misleading': the fact was proven wrong or misled "
                            "the agent. Mark it invalid and do not rely on it.\n"
                            "A fact missing from a batch is unchanged, not outdated. Only RETRACTED invalidates a fact."
                        ),
                        "observations_mission": OBSERVATIONS_MISSION,
                        # Disable auto-consolidation; Cerebral triggers it explicitly
                        # after retain with proper scope scoping (prevents unscoped
                        # pending rows from blocking scoped submissions).
                        "enable_auto_consolidation": False,
                    },
                    request_context=ctx,
                )
            )
        except Exception:
            logger.warning("cerebral: failed to configure bank %s", self._bank_id, exc_info=True)
        self._bank_ensured = True

    @staticmethod
    def _mutation_lines(episode: Episode) -> list[str]:
        """Build observation lines from episode mutations.

        Groups mutations by observation_id, keeping their order. `first` and
        `last` are the first and last mutation in the group, and `final` is the
        observation's content at the end of the run.

        Line format by mutation group (in order):

        | Group (in order) | Line |
        |---|---|
        | has ADD, in `context_memory` | `[s] <final>` |
        | has ADD, last is EVICT | `[s] <final>`, where final = `last.previous_content` |
        | has ADD, last is DELETE | nothing |
        | no ADD, last is REPLACE, in `context_memory` | `[s] <final> (supersedes: <first.previous_content>)` |
        | no ADD, has a REPLACE, last is EVICT | `[s] <last.previous_content> (supersedes: <first.previous_content>)` |
        | no ADD, EVICT only | nothing |
        | no ADD, last is DELETE | `[s] RETRACTED — shown incorrect or misleading, do not rely on: <first.previous_content>` |

        Section `s`: for a group whose last mutation is EVICT or DELETE, use
        `last.section`. Otherwise use the section `find_observation` returns.

        Returns list of formatted lines for the observations blob.
        """
        from codespy.agents.memory.hippocampus.context_memory import MutationType

        if not episode.mutations:
            return []

        # Group mutations by observation_id, preserving order
        groups: dict[str, list] = defaultdict(list)
        for mut in episode.mutations:
            groups[mut.observation_id].append(mut)

        lines: list[str] = []
        context_memory = episode.context_memory

        for obs_id, mutations in groups.items():
            first_mut = mutations[0]
            last_mut = mutations[-1]

            # Check if this observation has an ADD
            has_add = any(mut.type == MutationType.ADD for mut in mutations)

            if has_add:
                # ADD in the mutation group
                if last_mut.type == MutationType.DELETE:
                    # ADD → DELETE: emit nothing
                    continue
                elif last_mut.type == MutationType.EVICT:
                    # ADD → EVICT: emit line with evicted content (previous_content)
                    lines.append(f"[{last_mut.section}] {last_mut.previous_content}")
                else:
                    # ADD (possibly with REPLACEs), still in context_memory
                    found = context_memory.find_observation(obs_id)
                    if found:
                        section, obs = found
                        lines.append(f"[{section}] {obs.content}")
            else:
                # No ADD in the group: prior observation
                if last_mut.type == MutationType.DELETE:
                    # DELETE only: RETRACTED line
                    lines.append(
                        f"[{last_mut.section}] RETRACTED — shown incorrect or misleading, "
                        f"do not rely on: {first_mut.previous_content}"
                    )
                elif any(mut.type == MutationType.REPLACE for mut in mutations):
                    # Has at least one REPLACE (last could be REPLACE or EVICT)
                    if last_mut.type == MutationType.EVICT:
                        # REPLACE → EVICT: emit line with evicted content + supersedes
                        # Find the EVICT to get its previous_content
                        evict_mut = next(
                            mut for mut in reversed(mutations) if mut.type == MutationType.EVICT
                        )
                        lines.append(
                            f"[{last_mut.section}] {evict_mut.previous_content} "
                            f"(supersedes: {first_mut.previous_content})"
                        )
                    else:
                        # REPLACE only, still in context_memory
                        found = context_memory.find_observation(obs_id)
                        if found:
                            section, obs = found
                            lines.append(
                                f"[{section}] {obs.content} (supersedes: {first_mut.previous_content})"
                            )
                elif last_mut.type == MutationType.EVICT:
                    # EVICT only: emit nothing (unchanged prior fact)
                    pass

        return lines

    def retain_episode(self, episode: Episode, repo_full_name: str | None = None) -> None:
        """Retain episode mutations and artifacts in Hindsight semantic memory.

        The observations blob is built from episode.mutations instead of
        context_memory, capturing ADD/REPLACE/DELETE operations as:
        - Plain lines for ADDs still in context_memory
        - "(supersedes:)" lines for REPLACEs
        - "RETRACTED" lines for DELETEs

        With ``repo_full_name``, items also get ``repo:``/``org:`` tags and an
        explicit ``observation_scopes`` list (one scope per ``project_scope``
        topic, never ``task:``/``episode:``/``run_id:``/``pull_request:``), so
        observations merge across tasks and runs. Consolidation runs inline
        (``SyncTaskBackend``), then the mental models of the touched scopes
        are ensured. Without a repo, today's behaviour is kept.
        """
        self._ensure_bank()

        tags = self._build_tags(episode, repo_full_name)
        scopes = self._observation_scopes(episode, repo_full_name)
        episode_doc_id = f"episode-{episode.id}"

        # Build contents list: at most TWO items per episode
        contents: list[dict] = []

        # Build observations blob from mutations
        obs_lines = self._mutation_lines(episode)
        if obs_lines:
            contents.append({
                "content": "\n\n".join(obs_lines),
                "context": f"{episode.task}: {episode.question}: observation changes",
                "tags": tags,
                "document_id": episode_doc_id,
                "event_date": episode.timestamp.isoformat(),
            })

        # Merge all artifacts into a single content item (unchanged)
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

        if scopes is not None:
            for item in contents:
                item["observation_scopes"] = [list(sc) for sc in scopes]

        # Sync briefing triggers BEFORE retain+consolidation so consolidation
        # uses the correct refresh settings from the start.
        mental_model_scopes = self._mental_model_scopes(scopes or [], repo_full_name) if repo_full_name else []
        self._sync_briefing_triggers(mental_model_scopes)

        started = time.monotonic()
        logger.info(
            "cerebral: retain + consolidation started for episode %s (bank=%s, task=%s)",
            episode.id, self._bank_id, episode.task,
        )

        # Capture baseline memory_other stats for delta calculation
        from codespy.agents.memory.cerebral.cost import BUCKET_MEMORY_OTHER
        from codespy.agents.cost_tracker import get_cost_tracker
        tracker = get_cost_tracker()
        baseline_stats = tracker.get_signature_stats(BUCKET_MEMORY_OTHER)
        baseline_calls = baseline_stats.call_count if baseline_stats else 0

        try:
            self._run_async(
                self._engine.retain_batch_async(
                    bank_id=self._bank_id,
                    contents=contents,
                    request_context=RequestContext(),
                )
            )
        except Exception:
            logger.warning(
                "cerebral: failed to retain episode %s",
                episode.id, exc_info=True,
            )
            return

        # Scoped consolidation (serialized) - prevents unscoped pending rows
        # from blocking subsequent submissions.
        with self._consolidation_lock:
            for scope_tags in mental_model_scopes:
                self._submit_scoped_consolidation(scope_tags)

        # Log consolidation completion with delta
        end_stats = tracker.get_signature_stats(BUCKET_MEMORY_OTHER)
        end_calls = end_stats.call_count if end_stats else baseline_calls
        delta_calls = end_calls - baseline_calls

        logger.info(
            "cerebral: retained %d content blobs for episode %s (bank=%s, task=%s); "
            "consolidation finished in %.1fs, memory_other +%d calls",
            len(contents), episode.id, self._bank_id, episode.task,
            time.monotonic() - started,
            delta_calls,
        )

    @staticmethod
    def _observation_scopes(episode: Episode, repo_full_name: str | None) -> list[list[str]] | None:
        """Observation scopes for consolidation, or None without a repo (Hindsight default).

        One ``[org:, repo:, project_scope:]`` scope per ``project_scope`` topic,
        or ``[org:, repo:]`` when the episode has none.
        """
        if not repo_full_name:
            return None
        scopes = [
            scope_tags(repo_full_name, topic.id)
            for topic in episode.context_memory.topics
            if topic.type == "project_scope"
        ]
        return scopes or [repo_tags(repo_full_name)]

    @staticmethod
    def _mental_model_scopes(
        scopes: Sequence[Sequence[str]], repo_full_name: str
    ) -> list[list[str]]:
        """Observation scopes plus the repo scope ``[org:, repo:]`` (always added, deduped)."""
        out: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for sc in [*scopes, repo_tags(repo_full_name)]:
            key = tuple(sorted(sc))
            if key not in seen:
                seen.add(key)
                out.append(list(sc))
        return out

    @staticmethod
    def _mental_model_name(tags: Sequence[str]) -> str:
        for tag in tags:
            if tag.startswith(TAG_PROJECT_SCOPE):
                return f"Briefing: {tag[len(TAG_PROJECT_SCOPE):]}"
        for tag in tags:
            if tag.startswith("repo:"):
                return f"Briefing: {tag[len('repo:'):]}"
        return "Briefing"

    def _sync_briefing_triggers(self, scopes: list[list[str]]) -> None:
        """Sync briefing triggers to match current settings before consolidation.

        Runs before retain+consolidation so that consolidation uses the correct
        refresh settings. The desired trigger is:
        - ``refresh_after_consolidation``: True when mental_models is enabled
        - ``exclude_mental_models``: True (briefings exclude other briefings)

        For existing briefings, updates the trigger if it differs. For missing
        briefings, creates them only when mental_models is enabled.
        """
        if not scopes:
            return

        desired_trigger = {
            "refresh_after_consolidation": self._mental_models,
            "exclude_mental_models": True,
        }

        for tags in scopes:
            mm_id = mental_model_id(tags)
            if mm_id in self._ensured_mental_models:
                continue

            try:
                ctx = RequestContext()
                existing = self._run_async(
                    self._engine.get_mental_model(self._bank_id, mm_id, request_context=ctx)
                )

                if existing is None:
                    # Create missing briefing only when mental_models is enabled
                    if not self._mental_models:
                        self._ensured_mental_models.add(mm_id)
                        continue

                    try:
                        self._run_async(
                            self._engine.create_mental_model(
                                self._bank_id,
                                self._mental_model_name(tags),
                                MENTAL_MODEL_SOURCE_QUERY,
                                MENTAL_MODEL_PLACEHOLDER,
                                mental_model_id=mm_id,
                                tags=sorted(tags),
                                max_tokens=self._max_mental_model_tokens,
                                trigger=desired_trigger,
                                request_context=ctx,
                            )
                        )
                        logger.info(
                            "cerebral: created briefing %s (%s) refresh=%s",
                            mm_id,
                            self._mental_model_name(tags),
                            "on" if self._mental_models else "off",
                        )
                    except OperationValidationError as exc:
                        if getattr(exc, "status_code", None) != 409:
                            raise
                        # 409 means another process created it; fall through to update
                        existing = self._run_async(
                            self._engine.get_mental_model(self._bank_id, mm_id, request_context=ctx)
                        )
                else:
                    # Existing briefing: update trigger if it differs
                    current_trigger = existing.get("trigger") or {}
                    needs_update = (
                        current_trigger.get("refresh_after_consolidation") != self._mental_models
                        or current_trigger.get("exclude_mental_models") is not True
                    )
                    if needs_update:
                        self._run_async(
                            self._engine.update_mental_model(
                                self._bank_id,
                                mm_id,
                                trigger=desired_trigger,
                                request_context=ctx,
                            )
                        )
                        logger.info(
                            "cerebral: briefing %s refresh %s -> %s",
                            mm_id,
                            "on" if current_trigger.get("refresh_after_consolidation") else "off",
                            "on" if self._mental_models else "off",
                        )

                self._ensured_mental_models.add(mm_id)
            except Exception:
                logger.warning("cerebral: failed to sync briefing trigger for %s", mm_id, exc_info=True)

    def _submit_scoped_consolidation(self, scope_tags: list[str]) -> None:
        """Submit a scoped consolidation for one scope under the process lock.

        Scoped consolidation is never skipped, unlike unscoped submissions which
        are deduplicated when a pending unscoped row exists.
        """
        ctx = RequestContext()
        try:
            self._run_async(
                self._engine.submit_async_consolidation(
                    self._bank_id,
                    request_context=ctx,
                    observation_scopes=[scope_tags],
                )
            )
        except Exception:
            logger.warning(
                "cerebral: scoped consolidation failed for scope %s",
                scope_tags,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Read API (Prefrontal)
    # ------------------------------------------------------------------

    async def _abank_exists(self) -> bool:
        """Check whether the bank exists in Hindsight.

        Returns True immediately if already known to exist. Otherwise queries
        the engine and caches the result. "Missing" is not cached: a later
        retain creates the bank, and subsequent reads must see it.
        """
        if self._bank_exists:
            return True
        profile = await self._await(
            self._engine.get_bank_profile(self._bank_id, request_context=RequestContext())
        )
        if profile is not None:
            self._bank_exists = True
            return True
        if not self._missing_bank_logged:
            self._missing_bank_logged = True
            logger.info(
                "cerebral: bank %s not created yet (nothing retained); reads return empty",
                self._bank_id,
            )
        return False

    async def arecall(
        self,
        query: str,
        *,
        tag_groups: list[Any] | None = None,
        max_tokens: int = 4096,
        fact_type: list[str] | None = None,
        prefer_observations: bool = False,
        budget: str | Budget = Budget.MID,
        question_date: datetime | None = None,
    ) -> list[Any]:
        """Raw recall (one embedding call, no LLM). Returns ``MemoryFact`` list."""
        if not await self._abank_exists():
            return []
        result = await self._await(
            self._engine.recall_async(
                self._bank_id,
                query,
                budget=Budget(budget),
                max_tokens=max_tokens,
                fact_type=list(fact_type) if fact_type else None,
                prefer_observations=prefer_observations,
                question_date=question_date,
                tag_groups=tag_groups,
                request_context=RequestContext(),
            )
        )
        return list(getattr(result, "results", None) or [])

    def recall(self, query: str, **kwargs: Any) -> list[Any]:
        """Blocking variant of :meth:`arecall`."""
        return self._run_async(self.arecall(query, **kwargs))

    async def areflect(
        self,
        query: str,
        *,
        tag_groups: list[Any] | None = None,
        max_tokens: int = 4096,
        context: str | None = None,
        fact_types: list[str] | None = None,
    ) -> tuple[str, ReflectSummary]:
        """Hindsight reflect loop (LLM). ``budget=LOW`` so the iteration cap is exactly
        ``HINDSIGHT_API_REFLECT_MAX_ITERATIONS // 2`` (the LOW budget multiplier is 0.5).

        Args:
            query: The query to reflect on.
            tag_groups: Optional tag groups to filter by.
            max_tokens: Maximum tokens for the response.
            context: Optional context string for the reflection.
            fact_types: Optional list of fact types to consider.

        Returns:
            Tuple of (text, ReflectSummary). The summary records iterations, LLM calls,
            map-call count (for split synthesis detection), rewrite, tools used, usage,
            and whether the text is empty.
        """
        if not await self._abank_exists():
            return "", ReflectSummary()
        reflect_kwargs: dict[str, Any] = {
            "budget": Budget.LOW,
            "context": context,
            "max_tokens": max_tokens,
            "tag_groups": tag_groups,
            "fact_types": list(fact_types) if fact_types else None,
            "request_context": RequestContext(),
        }

        result = await self._await(
            self._engine.reflect_async(self._bank_id, query, **reflect_kwargs)
        )
        text = (getattr(result, "text", None) or "").strip()
        summary = self._build_reflect_summary(result)
        return text, summary

    def _build_reflect_summary(self, result: Any) -> ReflectSummary:
        """Build ReflectSummary from Hindsight ReflectResult.

        The ReflectResult shape from hindsight_api:
        - llm_trace: list[LLMCallTrace(scope, duration_ms)]
        - tool_trace: list[ToolCallTrace(tool, input, output: dict, iteration, ...)]
        - usage: TokenUsage(input_tokens, output_tokens, thoughts_tokens)
        """
        import json

        summary = ReflectSummary()
        if result is None:
            return summary

        # LLM trace is a list of LLMCallTrace objects with 'scope' attribute
        llm_trace = getattr(result, "llm_trace", None) or []
        scope_names: list[str] = []
        for call in llm_trace:
            scope = getattr(call, "scope", None)
            if scope:
                scope_names.append(str(scope))
                summary.llm_calls += 1
                if str(scope).startswith("final_map_"):
                    summary.map_calls += 1
                elif str(scope) == "final_rewrite":
                    summary.rewrite = True

        # Count iterations from agent_<n> scopes (excluding _err variants)
        # +1 if there's any final scope (map, reduce, or rewrite)
        agent_scopes = {s for s in scope_names if s.startswith("agent_") and not s.endswith("_err")}
        summary.iterations = len(agent_scopes)
        if any(s.startswith("final") for s in scope_names):
            summary.iterations += 1

        # Tools from tool_trace (list of ToolCallTrace with output dict)
        tool_trace = getattr(result, "tool_trace", None) or []
        for tc in tool_trace:
            tool_name = getattr(tc, "tool", "") or ""
            output = getattr(tc, "output", None) or {}
            # Estimate tokens from serialized output (same heuristic as prefrontal.format)
            try:
                output_text = json.dumps(output) if output else ""
            except Exception:
                output_text = str(output)
            output_tokens = (len(output_text) + 3) // 4  # chars/4 heuristic
            summary.tools.append((tool_name, output_tokens))

        # Usage from TokenUsage
        usage = getattr(result, "usage", None)
        if usage:
            summary.usage = {
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                "thoughts_tokens": getattr(usage, "thoughts_tokens", 0) or 0,
            }

        summary.empty = not (getattr(result, "text", None) or "").strip()
        return summary

    def reflect(self, query: str, **kwargs: Any) -> tuple[str, ReflectSummary]:
        """Blocking variant of :meth:`areflect`."""
        return self._run_async(self.areflect(query, **kwargs))

    async def aget_mental_models(self, ids: Sequence[str]) -> list[dict]:
        """Fetch mental models by id (no LLM). Missing ids are skipped; order is kept."""
        if not await self._abank_exists():
            return []
        async def _get_all() -> list[Any]:
            return await asyncio.gather(
                *(
                    self._engine.get_mental_model(
                        self._bank_id, mm_id, request_context=RequestContext()
                    )
                    for mm_id in ids
                ),
                return_exceptions=True,
            )

        results = await self._await(_get_all())
        models: list[dict] = []
        for mm_id, res in zip(ids, results, strict=True):
            if isinstance(res, BaseException):
                logger.debug("cerebral: get_mental_model %s failed: %s", mm_id, res)
                continue
            if res:
                models.append(res)
        return models

    async def aget_observation_history(self, memory_id: str) -> list[dict]:
        """History of an observation, most recent first (no LLM, no embedding)."""
        if not await self._abank_exists():
            return []
        result = await self._await(
            self._engine.get_observation_history(
                self._bank_id, memory_id, request_context=RequestContext()
            )
        )
        return list(result or [])

    @staticmethod
    def _build_tags(episode: Episode, repo_full_name: str | None = None) -> list[str]:
        tags: list[str] = []
        for topic in episode.context_memory.topics:
            tags.append(f"{topic.type}:{topic.id}")
        if repo_full_name:
            tags.extend(repo_tags(repo_full_name))
        tags.append(f"episode:{episode.id}")
        tags.append(f"task:{episode.task}")
        tags.append(f"run_id:{episode.run_id}")
        return tags
