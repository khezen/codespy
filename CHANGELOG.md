# Changelog

## [Unreleased]

### Added
- **Review memories section**: The pre-call Prefrontal memory (run-level and scope-level) now appears in a collapsed `memories` section **before** the Summary. Each recall is nested under its own `<details>` block, and each memory section (Briefings, Around this work, Seen before, Decisions, Recurring patterns, What changed, Other repositories) is also collapsible. This section is shown in both the GitHub/GitLab review body and the stdout/MCP markdown output.
- **Cerebral structured output support**: Native structured output registration for Bedrock models (e.g., Nvidia Nemotron Super) that support it but lack the flag in litellm's model map. Two-step structured output fallback for memory models litellm cannot schema-enforce, using `llm.extraction_model`.
- **Cerebral per-scope observation cap**: New `max_observations_per_scope` setting under `memory.cerebral.consolidation` (env `MEMORY_CONSOLIDATION_MAX_OBSERVATIONS_PER_SCOPE`, default `100`). Limits observations per `[org:*, repo:*, project_scope:*]` scope; `-1` = unlimited, `0` = no new observations. Scopes that hit the cap only allow UPDATE/DELETE.

### Changed
- **Cerebral tags**: Removed `episode:` and `run_id:` tags — they were not used for scoping or reach and bloated the tag set. `pull_request:` tags are also no longer added. Retained items now only carry `project_scope:`, `repo:`, `org:`, and `task:` tags.
- **Cerebral retain framing**: Moved from `context` to `metadata`. The `context` field is no longer set on retain items; instead `metadata` carries `task`, `repo` (when known), `kind` (`"observation changes"` / `"artifacts"`), and `question`. This reduces consolidation input size (context was repeated for every fact).
- **Cerebral fallback scope**: Episodes without a `project_scope` topic now consolidate into `[org:, repo:, project_scope:<owner/repo>]` (the repo-root project scope) instead of `[org:, repo:]`. This ensures every write scope matches the observation cap rule and creates/updates the repo-root briefing.
- **Mental-model configuration moved to `memory.cerebral.mental_models`**: Mental-model settings (`model`, `max_tokens`, `min_refresh_seconds`) now live under `memory.cerebral.mental_models` instead of `memory.prefrontal`. This better reflects that mental models are produced by Cerebral (briefing refresh after consolidation), not Prefrontal (read-only). Old env vars `MEMORY_MAX_MENTAL_MODEL_TOKENS` and `MEMORY_MIN_MENTAL_MODEL_REFRESH_SECONDS` are removed (no aliases, as they were unreleased). New env vars: `MEMORY_MENTAL_MODELS_MAX_TOKENS`, `MEMORY_MENTAL_MODELS_MIN_REFRESH_SECONDS`.

### Added
- **Cerebral consolidation and mental-model refresh model config**: Separate model configuration for consolidation (`memory.cerebral.consolidation.model`, env `MEMORY_CONSOLIDATION_MODEL`) and mental-model refresh (`memory.cerebral.mental_models.model`, env `MEMORY_MENTAL_MODELS_MODEL`). Both fall back through their respective chains: consolidation → retain → default; mental_models → prefrontal → retain → default. When unset, behavior is unchanged (consolidation uses retain model, mental-model refresh uses reflect model). Operator env vars (`HINDSIGHT_API_CONSOLIDATION_LLM_MODEL`, `HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_*`) still win when set.
- **Cerebral mental-models settings**: `memory.cerebral.mental_models.max_tokens` (env `MEMORY_MENTAL_MODELS_MAX_TOKENS`, default `2048`) for briefing size, and `memory.cerebral.mental_models.min_refresh_seconds` (env `MEMORY_MENTAL_MODELS_MIN_REFRESH_SECONDS`, default `0`) for minimum seconds between automatic briefing refreshes. Delta mode means nothing is lost — the refresh covers all new facts since `last_memory_seen_at` when the interval expires.

### Changed
- **Run-level Prefrontal recall**: One Prefrontal recall per run, shared by summary, code_review, doc, supply_chain, and audit agents. Previously each agent call ran its own `Prefrontal.aload()`, paying for the reflect loop separately. The scope agent still loads its own memory.
- **One consolidation per run**: Consolidation and mental-model refresh run once per run, after all background episode saves complete. Previously every `retain_episode` consolidated each scope plus the repo scope, refreshing the repo briefing N times per run.
- **Recalls table schema**: The `recalls` table is redesigned for run-level tracking:
  - `ordinal` (INT) replaced by `id` (UUID)
  - New columns: `run_id` (VARCHAR), `task` (VARCHAR)
  - `episode_id` is now nullable (NULL for the shared run-level load)
  - Primary key changed from `(bank_id, episode_id, ordinal)` to `(bank_id, id)`
  - New index: `idx_recalls_run (bank_id, run_id)`
- **Cerebral cost bucket split**: Consolidation and mental-model refresh LLM calls are now billed to
  `memory_consolidation` and `memory_mental_models` respectively, instead of `memory_other`.
  The `memory_other` bucket now only holds uncategorized LLM calls (e.g., standalone reflect calls).
  Embeddings remain in `memory_embeddings` / `memory_prefrontal_embeddings` (unchanged).
- **Briefing refresh trigger**: Now defaults to delta mode with reduced evidence caps:
  - `mode=delta` (was `full`): only new facts since last refresh
  - `include_chunks=false` (was `true`): observation text only, no file chunks
  - `reflect_search_observations_max_tokens=3000` (was 5000)
  - `reflect_search_observations_include_entities=false` (was `true`)
- **Doc review CHANGELOG trim**: CHANGELOG files are now trimmed to preamble + 2 newest sections (keeps `[Unreleased]` + latest release). Saves ~25KB (21%) of the ~118KB doc payload in this repo. Non-CHANGELOG docs are unchanged.
- **Cost breakdown split**: The per-signature cost table is split into a Review table and a Memory table (memory_* buckets), each with its own subtotal.

### Added
- **Cost tracking for extraction LM**: `TwoStepAdapter` extraction calls are now counted and attributed to the calling signature. Previously these were missing from cost tracking.

### Fixed
- **Retain no longer fails with "all N facts returned by the LLM were unusable" on Nemotron 3 Super (Bedrock)**. litellm lacked its native structured-output flag, so it fell back to forced tool calls with string-encoded arrays. Cerebral now registers the capability and strips JSON Schema keywords Bedrock's native structured output rejects (e.g., `minimum`/`maximum`, `maxItems`).
- **Briefing (mental model) delta refreshes no longer fail with "operations must be a list, got <class 'str'>"** (`delta_ops_failed`). Models without native structured output on Bedrock (e.g., Nvidia Nemotron) return the operations array JSON-encoded as a string; Cerebral now decodes it before Hindsight validates it.
- **`Missing required inputs: ['prefrontal_memory']` regression**: Fixed a bug where agents failed with `dspy.RLM._validate_inputs` error when the Prefrontal instance existed but the shared run-level `prefrontal_memory` text was empty (fresh DB). The signature is now chosen based on whether there is text to pass, not on the presence of the Prefrontal instance. Affects: summary, code_review, doc, supply_chain, audit.
- Cerebral consolidation and retain LLM calls now use `llm.timeout` / `llm.retries`. Previously only reflect did, so consolidation timed out at Hindsight's 120 s default.
- Cerebral consolidation no longer logs `relation "webhooks" does not exist`. Inline (`SyncTaskBackend`) tasks now carry the `semantic` schema, so consolidation completion no longer rolls back and falls back.
- Cerebral consolidation no longer fails at exit with `cannot schedule new futures after shutdown`. Background episode saves now finish before Python shuts down its thread pools, and the pipeline waits for them without the old 120 s cap. Runs can end later, because posting the review now waits for the consolidation to finish.
- Briefings no longer stay stuck on "Generating content..." after a failed refresh. The `_SchemaSyncTaskBackend` now marks failed operations as failed, and startup repair marks orphaned pending operations older than 1 hour as failed so subsequent refreshes can proceed.
- Embedding inputs are now capped for Bedrock Cohere v3 (2048 characters) to prevent `maxLength: 2048, actual: 2639` errors. Use `memory.cerebral.embeddings.max_input_chars` (env: `MEMORY_EMBEDDINGS_MAX_INPUT_CHARS`) to override: `null` = auto per model, `0` = disabled, `N` = custom cap.
- **Cost calculation**: `TwoStepAdapter` extraction calls are now included in signature cost tracking. Previously these calls were invisible because `SignatureContext` only read the main LM's history.
- **Cost billing**: DSPy cache hits (`response.cache_hit`) are now skipped in cost accounting, as they are not billed by the provider.
- **Cost split calculation**: When `entry["cost"]` is available (billed cost), input cost is now calculated as `billed_cost - output_cost` (list price). Previously rows could sum to more than Total when cache discounts applied.

### Added
- **Prefrontal reflect hard caps**: programmatic reflect configuration with new caps:
  - `memory.prefrontal.max_reflect_completion_tokens` (env `MEMORY_MAX_REFLECT_COMPLETION_TOKENS`, default `4096`): per-LLM-call output token cap. Reasoning tokens count against this first; keep it large enough for briefing rewrites (~2048 target).
  - `memory.prefrontal.max_reflect_context_tokens` (env `MEMORY_MAX_REFLECT_CONTEXT_TOKENS`, default `null`): when `null`, the cap is auto-derived from `5 × max_mental_model_tokens + max_reflect_fetch_tokens + 2048` overhead, divided by 0.8, so worst-case history fits in one chunk and no map calls occur. With defaults this is ~16000. An explicit value still wins.
  - Reflect caps are set programmatically on the Hindsight `_raw_config` after import, so they do not require a process restart. Env vars still win as operator overrides.
  - Load INFO line now includes `maps=N` (split synthesis map-call count), `rewrite` flag, and `thoughts=N` (reasoning tokens) for monitoring.
- **Prefrontal embeddings bucket**: `memory_prefrontal_embeddings` cost bucket for embeddings during Prefrontal recall (separate from `memory_prefrontal` LLM calls).
- **Reflect summary**: `areflect` returns `(text, ReflectSummary)` with iterations, LLM calls, map calls, rewrite flag, tools used, usage, and empty flag. Stored in `RecallRecord.details["reflect"]` for monitoring.
- **Prefrontal reflect cost caps**: new settings to cap the Hindsight reflect loop cost:
  - `memory.prefrontal.max_reflect_fetch_tokens` (env `MEMORY_MAX_REFLECT_FETCH_TOKENS`, default `2048`): per-call fetch limit for recall and search_observations tools. This is the default size used when the LLM omits one; the LLM can still ask for more up to the ceiling. Also disables chunk retrieval (`recall_include_chunks=false`).
- **Prefrontal (semantic recall)**: new `codespy/agents/memory/prefrontal/` package. Before each agent call it reads Cerebral and injects a read-only `prefrontal_memory` input (mental-model briefings, five question-shaped recall facets — context, seen before, decisions, recurring patterns, what changed — and facts from other repositories). The RLM agents (code_review, scope, supply_chain) also get an async `recall_memory(query, reach)` tool. Configured under `memory.prefrontal` (`MEMORY_PREFRONTAL_REACH`, `MEMORY_PREFRONTAL_REFLECTS`, `MEMORY_MAX_MENTAL_MODEL_TOKENS`, `MEMORY_MAX_PREFRONTAL_TOKENS`, `MEMORY_PREFRONTAL_REMOTE_SHARE`, `MEMORY_PREFRONTAL_RECALL_BUDGET`, `MEMORY_PREFRONTAL_TOOL`, `MEMORY_MAX_PREFRONTAL_TOOL_TOKENS`, `MEMORY_MAX_PREFRONTAL_TOOL_CALLS`). Active only when the signature's memory is enabled and Cerebral is available; otherwise agents behave exactly as before. Prefrontal never writes to Cerebral and never feeds Hippocampus.
- **Cerebral read API**: `arecall`/`recall`, `areflect`/`reflect` (`budget=mid`), `aget_mental_models`, `aget_observation_history`.
- **Prefrontal model setting**: `memory.prefrontal.model` (env `MEMORY_PREFRONTAL_MODEL`, Action input `memory-prefrontal-model`) selects the model for Hindsight's reflect loop and mental-model refresh. It falls back to `memory.cerebral.retain.model` → `llm.default_model`, so behaviour is unchanged when unset. It is checked at startup when `reflects > 0`.
- **Prefrontal recall monitoring**: every pre-call load and `recall_memory` call is logged as one INFO line (status, model, LLM calls, input/output tokens, cost, latency) and persisted to a new episodic `recalls` table (`Episode.recalls`, `RecallRecord` in `codespy/agents/memory/recall.py`). Rows hold the exact text the agent received and the measured token usage and cost. The table is additive (`CREATE TABLE IF NOT EXISTS`); recalls are never retained into Cerebral nor given to the Distiller/Cartographer.
- **`memory_prefrontal` cost bucket**: Prefrontal read calls (reflect LLM calls and query embeddings) are attributed to it through a context variable. Previously they were counted in `memory_other` / `memory_embeddings`.
- **Cerebral mental models**: one briefing per observation scope and per repo, created on first retain and refreshed after consolidation. `memory.prefrontal.reflects > 0` sets `HINDSIGHT_API_REFLECT_MAX_ITERATIONS`, which also caps mental-model refresh loops.
- **Cerebral (Semantic Memory)**:
  - New `memory.cerebral.retain.chunk_size` configuration (default: 12288, env var: `MEMORY_RETAIN_CHUNK_SIZE`). Chars per fact-extraction chunk; must be < 64000. See `CerebralRetainConfig` in `config_memory.py`.
  - New GitHub Action inputs: `memory-retain-model`, `memory-embeddings-model`, `memory-retain-chunk-size`.

### Changed
- **Cerebral consolidation now actually runs**: `MemoryEngine` uses `SyncTaskBackend`, so consolidation and mental-model refreshes run inline in `retain_episode` (background save thread). Previously `BrokerTaskBackend` only queued `pending` rows for a worker codespy never started. The first retain after upgrading consolidates the whole backlog (slow, costs LLM tokens); start and end are logged. Every retain now costs extra LLM calls (`memory_other` bucket) and background saves take longer.
- **Cerebral tags and observation scopes**: retained items get `repo:<owner/repo>` and `org:<owner>` tags and an explicit `observation_scopes` list (`[org:, repo:, project_scope:]` per scope, or `[org:, repo:]`), so observations merge across tasks and runs. `task:`/`episode:`/`run_id:`/`pull_request:` never enter a scope. The bank also gets an `observations_mission`. `retain_episode` takes `repo_full_name`; without it the old behaviour is kept. Old data is not backfilled.
- **Cerebral `retain_mission` is now domain-agnostic**: Replaced code-review-specific wording with general guidance for "observations, decisions, relationships, findings, insights" across any domain. The marker block (plain / `supersedes` / `RETRACTED`) semantics remain unchanged.
- **Agents no longer receive memory input**: `inject_context_memory` and the `context_memory=` argument have been removed from all six review agents (summary, scope, code_review, doc, supply_chain, audit). Memory is learned but not used directly; prior knowledge now reaches agents only through Prefrontal (see Added).
- **Hippocampus eviction writes tombstones**: When a prior observation is evicted from `context_memory` to stay within `max_hippocampus_tokens`, Hippocampus records a new `EVICT` mutation (`MutationType.EVICT`, separate from the Cartographer's `OpType`, so the LLM cannot emit it). `save_episode` writes an `op_type='EVICT'` tombstone version (`content=NULL`) for persisted observations, and `load_context` excludes both `DELETE` and `EVICT` tombstones.
- **Cerebral DELETE vs EVICT handling**: DELETE produces a `RETRACTED` line marking facts as wrong; EVICT of unchanged prior facts sends nothing; content new this run (ADD→EVICT or REPLACE→EVICT) is retained to avoid losing newly learned facts. The retain mission explicitly states that only `RETRACTED` invalidates a fact and missing facts are unchanged.
- **EpisodeStore.load_context() semantics changed**: Now loads the latest version of every observation from ALL prior episodes (not just the latest episode), excluding tombstones (`TOMBSTONE_TYPES`: `DELETE`, `EVICT`) and any version with no content. Observations are returned oldest-first, so eviction drops the oldest facts first. Topic prefix matching now uses proper LIKE escaping so `owner/repo` no longer matches `owner/repo-other`.
- **Rollback note**: loaders older than this release only skip `op_type='DELETE'` and fail to load a topic that has an `EVICT` row. Before rolling back, run `UPDATE observations SET op_type = 'DELETE' WHERE op_type = 'EVICT';` (both are content-NULL tombstones, so no data is lost).
- **BREAKING — Environment Variables**: Renamed env vars for consistency with YAML paths
  - `llm.*` settings drop the `LLM_` prefix (e.g., `LLM_DEFAULT_MODEL` → `DEFAULT_MODEL`)
    - Exceptions: `LLM_RETRIES` and `LLM_TIMEOUT` keep their prefixes
  - `memory.hippocampus.*` and `memory.cerebral.*` settings drop `HIPPOCAMPUS_`/`CEREBRAL_` prefixes
    - Examples: `MEMORY_HIPPOCAMPUS_DISTILLER_MODEL` → `MEMORY_DISTILLER_MODEL`, `MEMORY_CEREBRAL_RETAIN_MODEL` → `MEMORY_RETAIN_MODEL`
  - `review.*` settings keep the `REVIEW_` prefix (e.g., `CODE_REVIEW_MODEL` → `REVIEW_CODE_REVIEW_MODEL`, `MIN_CONFIDENCE` → `REVIEW_MIN_CONFIDENCE`)
  - Removed: `ENABLE_PROMPT_CACHING`, `COMPACT_PATCHES`, `MEMORY_CEREBRAL_*` (except `MEMORY_RETAIN_MODEL` and `MEMORY_EMBEDDINGS_MODEL`)
  - Migration: Update your `.env` file and GitHub Action inputs. The `llm.*`, `memory.postgres.*`, and `memory.pg0.*` names are unchanged from 1.2.4.
- **GitHub Action**:
  - The `enable-prompt-caching` and `compact-patches` inputs are removed (the underlying settings no longer exist)
  - All signature env vars now use `REVIEW_*` prefix (e.g., `SCOPE_ENABLED` → `REVIEW_SCOPE_ENABLED`, `CODE_REVIEW_MODEL` → `REVIEW_CODE_REVIEW_MODEL`)
  - Memory master switch: `MEMORY_DEFAULT_ENABLED` → `MEMORY_ENABLED`
  - Added missing passthrough for `DOC_MAX_ITERS` and `DOC_MAX_LLM_CALLS`
  - Migration note: **upgrade the action ref and `codespy-version` together**. An older action with the `latest` image, or the reverse, silently drops review settings.

### Changed
- **BREAKING — Memory Token Budgets**: Renamed token budget fields for clarity
  - `memory.hippocampus.max_context_memory_tokens` → `max_hippocampus_tokens` (env: `MEMORY_MAX_HIPPOCAMPUS_TOKENS`)
  - `memory.hippocampus.max_context_item_tokens` → `max_hippocampus_item_tokens` (env: `MEMORY_MAX_HIPPOCAMPUS_ITEM_TOKENS`)
  - Old keys and env vars are silently ignored; defaults apply. See codespy.yaml for current names.
- **Internal naming**: Cost table rows and internal unit names were renamed to match the `MEMORY_*` config/env names:
  - `distiller` → `memory_distiller`
  - `cartographer` → `memory_cartographer`
  - `cerebral_retain` → `memory_retain`
  - `cerebral_embeddings` → `memory_embeddings`
  - `cerebral_other` → `memory_other`

### Fixed
- Prefrontal reads on a bank with nothing retained yet (a new `MEMORY_BANK_ID`) now return empty results instead of failing with `Bank '<id>' not found` (hindsight 0.10.1 returns 404 for reads on missing banks). Reflect is skipped there.
- `Cerebral.__init__()` no longer receives unexpected `reflect_llm_provider` kwarg — the parameter is derived internally by Cerebral from `llm_provider`, so callers only pass `reflect_llm_model/api_key/base_url` when `memory.prefrontal.model` is set
- `get_llm_config("memory_retain")` no longer raises `AttributeError` — the cerebral branch now uses empty `ReflectionModuleConfig` and falls back to `llm.default_*` values
- `MEMORY_POSTGRES_SCHEMA` env var now works correctly (previously emitted `MEMORY_POSTGRES_SCHEMA_NAME` which was ignored)
- GitHub Action `memory-enabled` input now maps to `MEMORY_ENABLED` (was `MEMORY_DEFAULT_ENABLED`, which didn't match the code)

### Removed
- **Config**: Removed dead settings `enable_prompt_caching` and `compact_patches` (fields deleted in the config refactor)

- **Cerebral retains mutations, not observations**: Changed from full context_memory to mutation log
  - `retain_episode()` now builds observations blob from `episode.mutations`
  - ADD → plain line with final content
  - REPLACE → `supersedes:` line with previous_content from first mutation
  - DELETE → `RETRACTED` line with previous_content and optional reason
  - `_mutation_lines()` helper for building lines from mutation groups
- **Dependencies**: MCP Python SDK upgraded from v1.x to v2.x
  - `mcp` dependency: `>=1.29.0,<2.0.0` → `>=2.2,<3`
  - All 8 MCP servers migrated from `FastMCP` to `MCPServer` (v2 API)
  - `mcp_utils.py` updated to use new `Client` class with `StdioServerParameters`
  - Removed dead log suppression code for "Processing request of type" messages (no longer present in v2)

## [1.2.4] - 2026-09-18

### Changed
- **Performance**: Scope resolution fast-path for clean single-scope repos
  - When deterministic resolution finds exactly 1 scope and 0 orphans, skip expensive ReAct LLM refinement
  - New `skip_refinement_when_clean` setting on scope signature (default: `true`)
  - Env var: `SCOPE_SKIP_REFINEMENT_WHEN_CLEAN=true|false`
  - Improves scope duration from ~45s to <1s for typical single-scope repositories
  - Enhanced deterministic scope descriptions to include package name and scope type
- **Performance**: Audit episode save now runs in background
  - Moved `hippo.end_episode()` from synchronous to `submit_episode_save()` background thread
  - Audit duration drops from ~48s to ~5-15s (just the ChainOfThought LLM call + memory load)
  - Stats collection now runs after `join_episode_saves()` to ensure complete distiller/cartographer stats

### Added
- New `SignatureConfig.skip_refinement_when_clean` field (scope-specific optimization)
- New `Settings.get_scope_skip_refinement()` method (returns `True` by default)
- New `_finalize_scopes()` helper in scope resolver for skills attachment

## [1.2.3] - 2026-09-16

### Changed
- **BREAKING**: `Hippocampus` refactored from wrapper to composable component
  - `Hippocampus` no longer inherits from `dspy.Module` or accepts a `module` parameter
  - Removed `forward()`, `aforward()` — agents call their own modules and pass `hippo.context_memory`
  - Added `observe(result)` / `aobserve(result)` to feed agent results back for reflection
  - Added `context_memory` read-only property
  - Added `bind_topics(topics, topic_ids)` public method (replaces direct `_topic_ids` access)
  - Added `inject_context_memory(module)` standalone utility for signature injection
  - All 6 review agents updated to composable pattern
- **Internal**: Flattened `hippocampus/modules/` directory structure
  - Moved `distiller.py` and `cartographer.py` from `hippocampus/modules/` to `hippocampus/` package root
  - Removed `hippocampus/modules/__init__.py` and deleted `modules/` subdirectory
  - Updated imports in `hippocampus.py` and `hippocampus/__init__.py` to reference new locations
  - No external API changes — imports through `hippocampus/__init__.py` remain unchanged

## [1.2.2] - 2026-09-16

### Changed
- **BREAKING**: Codebase restructured to separate agents from workflow orchestration
  - Review agents moved from `codespy.agents.reviewer.modules.*` to `codespy.agents.review.*` subpackages (audit, code_review, doc, scope, summary, supply_chain)
  - Workflow orchestration moved from `codespy.agents.reviewer` to `codespy.workflows.review` (pipeline, reporters, server, workflow-owned models)
  - Agent-owned types (`Issue`, `PRContext`, `ReviewContext`, `ReviewMetadata`, `IssueCategory`, `IssueSeverity`) moved to `codespy.agents.review.models`
  - Scope-specific types (`ScopeResult`, `ScopeType`, `PackageManifest`) moved to `codespy.agents.review.scope.models`
  - Workflow-owned types (`ReviewResult`, `ReviewConfig`, `RemoteReviewConfig`, `LocalReviewConfig`, `SignatureStatsResult`) moved to `codespy.workflows.review.models`
  - `helpers.py` moved from `codespy.agents.reviewer.modules` to `codespy.agents.review`
  - `doc_extractor.py` moved from `codespy.agents.reviewer.modules` to `codespy.agents.review.doc`
  - `manifest_parser.py` moved from `codespy.agents.reviewer.modules` to `codespy.agents.review.scope`
  - MCP server moved from `codespy.agents.reviewer.server` to `codespy.workflows.review.server`
  - CLI modules (`cli_local`, `cli_remote`, `cli_mcp_server`) updated for new import paths
  - All tests updated for new import paths

### Removed
- `codespy.agents.reviewer` package (replaced by `codespy.agents.review` + `codespy.workflows.review`)

## [1.2.1] - 2026-09-14

### Fixed
- GitHub Action: Added `--network host` to Docker run arguments to fix connectivity to external PostgreSQL (AWS RDS) hosts
  - Resolves `PoolTimeout` errors when connecting to RDS from GitHub Actions runners
  - The Docker container now shares the host's network stack, avoiding bridge network NAT/DNS issues

## [1.2.0] - 2026-09-11

### Changed
- **BREAKING**: Memory storage backend migrated from filesystem/S3 to PostgreSQL
  - `MemoryConfig` fields removed: `backend`, `root`, `s3_bucket`, `s3_region`, `s3_endpoint_url`
  - New `PostgresConfig` sub-model (`memory.postgres.*`): `host`, `port`, `user`, `password`, `database`, `schema`
  - New `Pg0Config` sub-model (`memory.pg0.*`): `name`, `port`, `data_dir` (local dev via pg0-embedded)
  - New `bank_id` field on `MemoryConfig` scoping all memory data
  - Env vars removed: `MEMORY_BACKEND`, `MEMORY_ROOT`, `MEMORY_S3_BUCKET`, `MEMORY_S3_REGION`, `MEMORY_S3_ENDPOINT_URL`
  - Env vars added: `MEMORY_POSTGRES_HOST`, `MEMORY_POSTGRES_PORT`, `MEMORY_POSTGRES_USER`, `MEMORY_POSTGRES_PASSWORD`, `MEMORY_POSTGRES_DATABASE`, `MEMORY_POSTGRES_SCHEMA`, `MEMORY_PG0_NAME`, `MEMORY_PG0_PORT`, `MEMORY_PG0_DATA_DIR`, `MEMORY_BANK_ID`
- **BREAKING**: `Item` renamed to `Observation` throughout context memory
  - `ItemTag` → `ObservationTag`
  - `Operation.item_id` → `Operation.observation_id`
  - `Mutation.item_id` → `Mutation.observation_id`
  - `ContextMemory.all_items()` → `ContextMemory.all_observations()`
  - `ContextMemory.find_item()` → `ContextMemory.find_observation()`
  - `Hippocampus._update_item_scores()` → `Hippocampus._update_observation_scores()`
- **BREAKING**: `Topic` model gains required `type` field (e.g. `"project_scope"`, `"pull_request"`)
- **BREAKING**: `get_memory_store()` / `reset_memory_store()` renamed to `get_episode_store()` / `reset_episode_store()`
- **BREAKING**: `Hippocampus.end_episode()` signature: `store` type `Storage` → `EpisodeStore`; `dir` parameter removed
- **BREAKING**: `Hippocampus.save_episode()`, `Hippocampus.load_episode()` removed; use `EpisodeStore.save_episode()` directly
- **BREAKING**: `find_latest_episode()`, `save_episode()`, `load_episode()` removed from `episode.py`; replaced by `EpisodeStore.load_context()`
- **BREAKING**: `ContextMemory.merge()`, `.to_json()`, `.from_json()` removed; `EpisodeStore.load_context()` returns a merged view via SQL
- **BREAKING**: GitHub Action inputs replaced: `memory-backend`, `memory-root`, `memory-s3-*` → `memory-postgres-*`, `memory-bank-id`
- All review modules (code_reviewer, doc_reviewer, supply_chain_auditor, auditor, summarizer, scope_resolver) now use `EpisodeStore.load_context()` with topic-based queries instead of `find_latest_episode()` with filesystem path patterns
- `PRContext.repo_full_name` property added, stripping host prefix from `repo_slug`
- REPLACE on non-existent observation with valid section prefix now falls back to ADD instead of silently skipping
- REPLACE with topic-ID-shaped `observation_id` (no section prefix) logs warning and skips
- `Hippocampus._record_mutations()` handles REPLACE-to-ADD fallback to keep mutations aligned with `new_ids`
- `Episode.id` field added (caller-provided UUID); `Episode.timestamp` no longer auto-generated
- `reset_episode_store()` calls `store.close()` before clearing the cache
- SQL in `postgres.py` refactored from f-strings to `psycopg.sql.SQL().format()`

### Added
- `src/codespy/agents/memory/postgres.py` — `EpisodeStore`: relational episode storage with schema (banks, episodes, topics, observations, observation_topics, episode_topics, mutations)
- `src/codespy/agents/memory/pg0_manager.py` — pg0-embedded lifecycle (`get_pg0_uri()`, `stop_pg0()`) for zero-config local dev
- `psycopg` dependency (>=3.1, extras: binary + pool)
- `pg0-embedded` dependency (>=0.15)
- `libgssapi-krb5-2` in Dockerfile for Kerberos/GSSAPI PostgreSQL auth
- `_PREFIX_TO_SECTION` reverse mapping in `context_memory.py`

### Removed
- `ContextMemory.merge()`, `.to_json()`, `.from_json()`
- `find_latest_episode()`, `save_episode()`, `load_episode()` from `episode.py`
- `Hippocampus.save_episode()`, `.load_episode()`, `.episode_file_path()`, `_episode_index`
- `MemoryBackend` type alias
- Storage dependency (`codespy.tools.storage.base.Storage`, `codespy.tools.storage.models`) in hippocampus/episode modules

## [1.0.16] - 2026-09-01

### Changed
- Manifest discovery (`_discover_manifests`) now scans only ancestor directories of changed files instead of walking the entire repository tree — significant performance improvement for large repos
- `_discover_manifests` signature: added `changed_files: list[ChangedFile]` parameter
- Auditor no longer receives per-file metadata: removed `changed_files` input from `AuditSignature` and `forward()` / `_call_auditor()` parameters
- `Auditor.forward()` signature simplified: `Sequence[ChangedFile]` parameter dropped, `all_issues` type narrowed from `Sequence[Issue]` to `list[Issue]`

## [1.0.15] - 2026-09-01

### Changed
- Renamed `experiences` section to `actions` in `ContextMemory` (section prefix `"ex"` → `"ac"`)
- `max_context_item_tokens` default: `410` → `512` (capacity: ~32 items with 16384 context memory tokens)
- `Hippocampus` init parameter `topic_ids: list[str]` → `topics: list[Topic]` — topics auto-registered in context memory
- `Summarizer.forward()` signature simplified: individual PR fields replaced with `pr_context: PRContext` parameter
- Scope resolver no longer builds dependency graphs between scopes
- Per-signature iteration defaults: `doc` (max_iters=1, max_llm_calls=2), `scope` (max_iters=3, max_llm_calls=5), `code_review` (max_iters=5, max_llm_calls=8)

### Added
- `PRContext.pr_url` and `PRContext.pr_description` fields
- `PRContext.to_topic()` helper — builds a `Topic` from PR metadata for cross-review memory
- PR URL topic automatically registered across all pipeline modules (scope resolver, summarizer, code reviewer, doc reviewer, supply chain auditor, auditor)
- GitHub Action inputs: `doc-max-iters`, `doc-max-llm-calls`

### Removed
- `Topic.dependencies` field and all dependency resolution logic in scope resolver
- `ScopeResult.is_dependency` field
- Dependency extraction from `manifest_parser.py` (~578 lines): `PACKAGE_MANAGER_TO_ECOSYSTEM`, `extract_dependencies()`, `infer_repo_from_name()`, and 13 per-language `_extract_deps_from_*` functions — only `extract_package_name()` retained

## [1.0.14] - 2026-08-31

### Fixed
- Resolved leftover merge conflict markers in `.env.example` and `CHANGELOG.md`

## [1.0.13] - 2026-08-31

### Changed
- **Agent runtime: `dspy.ReAct` → `dspy.RLM`** in code reviewer, scope resolver, and supply chain auditor — RLM provides the same tool-using loop with improved context management
- `default_max_iters` raised from 3 → 5
- `default_max_llm_calls` raised from 5 → 8
- `default_max_tokens` reduced from 64000 → 32000
- `summary` signature defaults: `max_iters=1`, `max_llm_calls=2`
- `audit` signature defaults: `max_iters=1`, `max_llm_calls=2`
- Removed dead-code RLM fallback defaults (`or 4` / `or 8`) in `ContextSafe._create_rlm_fallback`

### Added
- Per-module `max_iters` and `max_llm_calls` overrides for Distiller and Cartographer reflection modules
  - YAML: `memory.distiller.max_iters`, `memory.distiller.max_llm_calls` (same for cartographer)
  - Env vars: `MEMORY_DISTILLER_MAX_ITERS`, `MEMORY_DISTILLER_MAX_LLM_CALLS`, `MEMORY_CARTOGRAPHER_MAX_ITERS`, `MEMORY_CARTOGRAPHER_MAX_LLM_CALLS`
  - GitHub Action inputs: `memory-distiller-max-iters`, `memory-distiller-max-llm-calls`, `memory-cartographer-max-iters`, `memory-cartographer-max-llm-calls`
  - Defaults: `max_iters=1`, `max_llm_calls=2` (ChainOfThought modules — no tools to iterate)
- `max_iters` now propagated to doc reviewer, summarizer, and auditor ChainOfThought modules
- `max_llm_calls` now propagated to code reviewer, scope resolver, and supply chain auditor RLM agents
- GitHub Action: `default-max-tokens` input, `summary-max-iters` input, full `audit-*` signature inputs (`audit-enabled`, `audit-model`, `audit-max-iters`, `audit-max-llm-calls`, `audit-reasoning-effort`, `audit-temperature`), `audit-memory-enabled` input

## [1.0.12] - 2026-08-30

### Changed
- Parallelized scope processing in code reviewer, doc reviewer, and supply chain auditor using `asyncio.gather` — multi-scope PRs now review scopes concurrently instead of sequentially
- Reduced `default_max_iters` from 4 to 3
- Reduced `default_max_llm_calls` from 8 to 5
- Reduced `llm_retries` from 3 to 2
- Scope resolver `max_iters` changed from 20 to `null` (inherits `default_max_iters: 3`)
- Code review `reasoning_effort` default changed from `high` to `medium` in `.env.example`
- Auditor episode save switched from background fire-and-forget to synchronous (auditor is the last pipeline module)

### Added
- `join_episode_saves()` call at end of review pipeline to ensure all background episode saves complete before returning results

### Removed
- `docker-run-json` Makefile target

## [1.0.11] - 2026-08-29

### Added
- Optional trajectory compaction via `compact_trajectory` config flag (default: `false`)
  - When `false`, head+tail bounding is skipped and full trajectory goes to the Distiller
  - ContextSafe RLM fallback handles overflow if trajectory exceeds model's context window
  - Env var: `MEMORY_COMPACT_TRAJECTORY` (default: `false`)
  - Config field: `memory.compact_trajectory`

### Changed
- Memory token budget defaults increased:
  - `max_context_memory_tokens`: 8192 → 16384 (~39 items capacity vs ~19)
  - `max_trajectory_tokens`: 8192 → 16384 (~12% of 128k context window)
  - `max_question_tokens`: 2048 → 8192
- Removed `default_` prefix from global memory token budget fields in `MemoryConfig`:
  - `default_max_context_memory_tokens` → `max_context_memory_tokens`
  - `default_max_context_item_tokens` → `max_context_item_tokens`
  - `default_max_trajectory_tokens` → `max_trajectory_tokens`
  - `default_max_question_tokens` → `max_question_tokens`
  - `default_compact_trajectory` → `compact_trajectory`
- Environment variable names updated (removed `DEFAULT_` prefix):
  - `MEMORY_DEFAULT_MAX_CONTEXT_MEMORY_TOKENS` → `MEMORY_MAX_CONTEXT_MEMORY_TOKENS`
  - `MEMORY_DEFAULT_MAX_CONTEXT_ITEM_TOKENS` → `MEMORY_MAX_CONTEXT_ITEM_TOKENS`
  - `MEMORY_DEFAULT_MAX_TRAJECTORY_TOKENS` → `MEMORY_MAX_TRAJECTORY_TOKENS`
  - `MEMORY_DEFAULT_MAX_QUESTION_TOKENS` → `MEMORY_MAX_QUESTION_TOKENS`
  - `MEMORY_DEFAULT_COMPACT_TRAJECTORY` → `MEMORY_COMPACT_TRAJECTORY`

### Fixed
- Memory budget field naming now consistent: only `default_enabled` and `default_max_reflects` retain `default_` prefix (these support per-signature overrides)

## [1.0.10] - 2026-08-29

### Added
- New `experiences` section in `ContextMemory` for tracking tool execution patterns
  - Records tool usage patterns (what tool, what purpose, what result) that transfer across runs
  - Helps agents avoid redundant tool calls in future runs
  - Added `"experiences"` to `SectionName` Literal type with `"ex"` prefix
  - Added `experiences` field to `ContextMemory` class (renders last in LLM prompts)
  - Updated `CacheCandidate.section` description to include `experiences`
  - Added `experiences` to `_SECTION_EVICT_PRIORITY` at priority 1 (evicts before `reusable_results`)
  - Updated `DistillerSig` docstring to describe experiences as "Medium value" cache candidate
  - Updated `CartographerSig` docstring to include experiences in value priority list (priority 5 of 6)
  - Updated section count references from "five" to "six" in Distiller and Cartographer prompts

## [1.0.9] - 2026-08-29

### Changed
- Memory isolation: each pipeline module (code_review, doc, supply_chain, audit, summary, scope) now loads its own prior episodes per-scope instead of inheriting context memory from upstream stages
- `ReviewContext.memory` field marked unused (kept for API compatibility); modules no longer return or merge context memories
- Scope resolver, summarizer, code reviewer, doc reviewer, supply chain auditor, and auditor all load prior episodes independently via `find_latest_episode`
- Summarizer `initial_memory` parameter removed; memory loaded internally from prior "summary" episodes
- Review modules now return `(issues, None)` instead of `(issues, ContextMemory)`; pipeline no longer merges parallel memories for auditor
- Scope resolver returns `list[ScopeResult]` instead of `tuple[list[ScopeResult], ContextMemory | None]`
- `default_max_iters` reduced from 10 to 4
- `default_max_llm_calls` reduced from 30 to 8

### Added
- Background episode persistence via `submit_episode_save()` / `join_episode_saves()` in `codespy.agents.memory.hippocampus.episode`
- Non-daemon background threads for episode save ensure persistence even on fast process exit
- Per-module episode loading with `find_latest_episode` scoped by task name (e.g. `task="audit"`, `task="code_review"`)

### Fixed
- Eliminated pipeline-blocking I/O: episode consolidation and save no longer blocks the review pipeline between stages

## [1.0.8] - 2026-08-18

### Security
- S3 StreamingBody resource leak fix: wrapped `resp["Body"].read()` in try/finally to ensure `body.close()` is called, preventing connection pool exhaustion on partial read failures
- SecretStr migration for all credential fields: tokens and API keys now use Pydantic `SecretStr` type to prevent accidental logging of sensitive values
  - Affected fields: `github_token`, `gh_token`, `gitlab_token`, `aws_access_key_id`, `aws_secret_access_key`, `openai_api_key`, `anthropic_api_key`, `gemini_api_key`
  - New `config_utils.secret_value()` helper extracts plain text at API boundaries
  - `model_dump()` masks secrets as `'**********'`
- GitLab client timeout: added `timeout=30` to `gitlab.Gitlab()` instantiation to prevent indefinite hangs (python-gitlab >=4.0.0 defaults to `None`)
- Dependency floor bumps to fix known vulnerabilities:
  - `gitpython` >=3.1.42 — RCE via malicious git repo (CVE-2024-22190)
  - `json-repair` >=0.60.1 — DoS via circular $ref (GHSA-xf7x-x43h-rpqh)
  - `markdownify` >=0.15.0 — ReDoS vulnerability (GHSA-7mpr-5m44-h73h)

## [1.0.7] - 2026-08-18

### Fixed
- Resolved all 156 ruff lint violations across the codebase
  - Fixed 79 E501 line-too-long errors via `ruff format`
  - Fixed 11 B904 errors (raise without `from` in exception handlers)
  - Fixed 9 E402 errors (imports not at top of file)
  - Fixed 1 F821 error (undefined name `Topic` in models.py)
  - Fixed 4 SIM102 errors (collapsible nested if statements)
  - Fixed 2 E741 errors (ambiguous variable name `l`)
  - Fixed 2 N806 errors (uppercase variable in function)
  - Fixed 1 N817 error (CamelCase imported as acronym)
  - Fixed 2 B017 errors (blind Exception assertion in tests)
  - Fixed 42 remaining E501 errors via string concatenation and refactoring
  - Fixed 3 I001/F401 errors (import sorting and unused imports)

## [1.0.6] - 2026-08-18

### Changed
- Removed default value for `extraction-model` GitHub Action input (was `anthropic/claude-haiku-4-5-20251001`)

## [1.0.5] - 2026-08-18

### Added
- Proactive RLM fallback with configurable context rot thresholds per module type (ReAct: 0.30, ChainOfThought: 0.40, Predict: 0.50)
- New `rlm_fallback` config section with `enabled`, `react_threshold`, `chain_of_thought_threshold`, `predict_threshold`
- GitHub Action inputs: `rlm-fallback-enabled`, `rlm-fallback-react-threshold`, `rlm-fallback-chain-of-thought-threshold`, `rlm-fallback-predict-threshold`

### Changed
- `ContextSafe` now checks proactive context rot threshold before checking hard overflow (existing overflow detection retained as safety net)
- `ContextSafe._would_overflow` renamed to `_should_use_rlm` with expanded three-layer logic

## [1.0.4] - 2026-08-18

### Fixed
- LLM output parse failure: `Issue.severity` field now defaults to `medium` when omitted by the model (was hard-required, causing 0 issues returned)
- Enum case normalization: `OpType` and `ItemTag` now accept any casing via `_missing_` hooks (e.g., `"add"` → `ADD`, `"NEUTRAL"` → `neutral`)
- `CacheCandidate` fields (`section`, `transferability`, `rationale`) now have defaults so partial LLM output still parses
- Reviewer signatures now include explicit severity guidance in OUTPUT RULES (doc: low/medium, code: critical/high/medium/low)
- Cartographer signature: removed "JSON" terminology from Operation rules (aligns with Distiller change)
- `Operation.type` field description now lists allowed values explicitly

## [1.0.3] - 2026-08-18

### Added
- Configurable `DEFAULT_MAX_LLM_CALLS` setting (default 30) controlling maximum LLM calls for RLM context-overflow fallback
- Configurable `MIN_CONFIDENCE` threshold (default 0.81) for filtering low-confidence issues
- Per-signature `max_llm_calls` override (`SCOPE_MAX_LLM_CALLS`, `CODE_REVIEW_MAX_LLM_CALLS`, `SUPPLY_CHAIN_MAX_LLM_CALLS`, `SUMMARY_MAX_LLM_CALLS`, `DOC_MAX_LLM_CALLS`, `AUDIT_MAX_LLM_CALLS`)
- `ContextSafe` now accepts `max_iters` and `max_llm_calls` parameters for per-module override

### Changed
- `default-max-iters` action input default raised from 3 to 10
- Moved hardcoded `MIN_CONFIDENCE` constant from `helpers.py` to configurable `Settings.min_confidence` field (default 0.81)

### Fixed
- RLM fallback previously hardcoded `max_iters=10, max_llm_calls=20`; now respects per-signature and global configuration

## [1.0.2] - 2026-08-18

### Fixed
- Hippocampus consolidation parse failure when LLM outputs `"op"` instead of `"type"` in Cartographer operations (weaker models like nemotron/kimi adopt docstring terminology as JSON key name). Added `validation_alias` to accept both field names.
- Cartographer signature docstring now explicitly names all `Operation` JSON fields to guide LLM structured output

## [1.0.1] - 2026-08-18

### Fixed
- Git authentication failure in GitHub Actions: changed token URL format to use `x-access-token:{token}` username scheme (was `{token}@`, causing git to prompt for password)
- Added `GIT_TERMINAL_PROMPT=0` environment variable to all git operations to prevent hanging on authentication prompts in headless environments
- Guarded `_expand_sparse_for_scopes` against unborn repos (when clone fails, leaving `.git/` skeleton without commits) to prevent "branch yet to be born" crashes

## [1.0.0] - 2026-08-18

### Added
- Cross-review memory system (Hippocampus) with S3/filesystem storage backends
- Context window overflow resilience (`ContextSafe` wrapper with automatic RLM fallback)
- Scope resolver: deterministic analysis + ReAct agent refinement (replaces `ScopeIdentifier`)
- Patch compaction: expands diff hunks to enclosing function boundaries via Tree-sitter
- Deterministic package manifest parser: extracts package identity from 25+ formats without LLM (npm, Go, pip, Cargo, Maven, Gradle, Composer, Bundler, NuGet, Swift, Pub, Hex, Helm, etc.)
- Tree-sitter extractors for Bash, C++, C#, PHP, Ruby
- Ripgrep fallback extractor for languages without Tree-sitter grammar
- Unified storage abstraction layer (`tools/storage/`) with filesystem and S3 backends
- Audit signature: dedicated module for review quality assessment and recommendation
- Reasoning effort configuration (`minimal|low|medium|high`) — maps to provider-native parameters (Anthropic thinking budget, OpenAI reasoning_effort)
- Per-signature `max_tokens` output token budget (replaces `max_reasoning_tokens`)
- TwoStepAdapter with dedicated extraction model for structured field extraction
- Memory storage access verification (S3/filesystem connectivity check at startup)
- Sparse checkout support in scope resolver for large monorepos
- Deno runtime in Docker image (required by DSPy RLM sandbox)
- Full documentation suite: architecture, configuration, development, memory, usage
- GitHub Action: reasoning effort, summary, and temperature inputs

### Security
- S3 path traversal hardening: `_resolve_path` decodes percent-encoded input before validation (catches `%2e%2e`, `%2f..%2f` — CWE-22)
- `json-repair` pinned to >=0.56.0 (GHSA-xf7x-x43h-rpqh)
- `litellm` floor raised to ^1.84.0 (excludes known-vulnerable versions)
- `gitpython` floor raised to >=3.1.41 (excludes CVE-affected versions)
- `markdownify` floor raised to >=0.14.0 (excludes known-vulnerable versions)

### Changed
- **BREAKING**: `MergeRequest` model renamed to `PullRequest` (backward-compat alias removed)
- **BREAKING**: `ReviewContext.merge_request` field → `pull_request` (compat property removed)
- **BREAKING**: CLI argument `mr_url` → `pr_url`; `fetch_merge_request()` → `fetch_pull_request()`
- **BREAKING**: `build_mr_from_diff()` → `build_pr_from_diff()`
- **BREAKING**: `ReviewResult` fields: `mr_number`→`pr_number`, `mr_title`→`pr_title`, `mr_url`→`pr_url`
- **BREAKING**: MCP tool `review_pr` parameter renamed: `mr_url` → `pr_url`
- **BREAKING**: `tools/filesystem` module moved to `tools/storage/filesystem` (import path changed)
- **BREAKING**: Per-signature `max_context_size` and `max_reasoning_tokens` env vars replaced by `reasoning_effort`, `temperature`, and `max_tokens`
- **BREAKING**: GitHub Action inputs removed: `*-max-context-size`, `*-max-reasoning-tokens` (replaced by `*-reasoning-effort`)
- Docker base image: Alpine → Debian slim (glibc required by Deno)
- `dspy` dependency: ^3.1.3 → ^3.3.0
- `mcp` dependency: >=1.0.0 → >=1.29.0,<2.0.0
- `litellm` dependency: ^1.81.6 → ^1.84.0
- `gitpython` dependency: >=3.1.0 → >=3.1.41
- `json-repair` dependency: ^0.55.1 → >=0.56.0
- `markdownify` dependency: >=0.13.0 → >=0.14.0
- ScopeIdentifierSignature → ScopeRefinementSignature (extracted to `ScopeResolver` module)
- MRSummarySignature → PRSummarySignature (extracted to `Summarizer` module with config key `summary`)
- `ReviewMetadata` model introduced to reduce parameter proliferation
- Per-module overflow detection replaced by centralized `ContextSafe` wrapper
- README rewritten: detailed sections moved to `docs/`, simplified TOC
- `codespy.yaml` expanded with memory, reasoning, and per-signature configuration (194 → 326 lines)

### Removed
- `ScopeIdentifier` module (replaced by `ScopeResolver`)
- `tools/filesystem/__init__.py` (replaced by `tools/storage/` abstraction)
- `default_max_context_size` and `default_max_reasoning_tokens` settings
- Per-signature `MAX_CONTEXT_SIZE` and `MAX_REASONING_TOKENS` env vars (replaced by `REASONING_EFFORT` and `MAX_TOKENS`)
- `MergeRequest` backward-compat alias
- Per-module `_would_overflow_context()` methods
