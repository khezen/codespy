# Changelog

## [Unreleased]

### Fixed

- **code_review, supply_chain and scope refinement never found issues.** Every tool call (`read_file`, `search_literal`, `find_*`, `recall_memory`) failed inside the RLM sandbox with `This event loop is already running`. RLM agents now run off the event loop using `asyncio.to_thread()`, and tool calls are bridged back to the event loop via `asyncio.run_coroutine_threadsafe()`. New module `agents/rlm_tools.py` provides `build_rlm_agent()`, `run_rlm()`, and `log_rlm_outcome()`.
- **Reviewers silently discarded issues without an explicit confidence**, because the `Issue.confidence` default (0.8) was below the `review.min_confidence` default (0.81). Prompts now request a confidence ("≥0.9 verified with tools; 0.7–0.9 strong evidence; <0.7 weak"), and `filter_by_confidence()` logs raw/kept counts.
- **Opus 5.5 on Bedrock refused every RLM call with `content_filter`, so code_review returned nothing (`Empty LM response`).** The trigger was `dspy.RLM`'s stock `reasoning` field description ("Think step-by-step: what do you know? What remains? Plan your next action."). New `CodespyRLM` (`agents/context_safe.py`) replaces it with "Short plan for the next step." and is used by `build_rlm_agent()` and the `ContextSafe` RLM fallback. `scripts/replay_refusal.py` replays refusal dumps under bisection variants.
- **RLM REPL steps all failed with an empty `Invalid Python syntax`.** dspy's `PythonInterpreter` injects inputs as `name = repr(value)`, and an Enum repr (`<IssueCategory.BUG: 'bug'>`) is not valid Python, so every execution of code_review (`categories`) and supply_chain (`category`) failed before the model's code ran. `CodespyRLM` now passes Enum inputs as their plain values.
- **RLM tool calls with arguments failed in the sandbox** (`get_tree() got an unexpected keyword argument 'max_depth'`, `Tool 'read_file' rejected arguments: ['path']`). dspy.RLM copies each tool's Python signature into the sandbox, and the async-to-sync bridge exposed `sync_wrapper(**kwargs)`, so the sandbox wrapper accepted only a literal `kwargs` argument. `bridge_tools_to_loop()` now sets the bridged function's signature from the tool's args.

### Changed

- **Default `review.min_confidence` (`REVIEW_MIN_CONFIDENCE`) is now 0.5 (was 0.81).** This allows issues to be reported even when the LLM does not explicitly set a confidence score (the `Issue.confidence` default is 0.8, which now passes the threshold). Update your codespy.yaml or environment variable to override if needed.

## [2.0.0] - 2026-10-03

### Changed (Breaking)

- **Configuration structure** — `Settings` is now `llm` / `github` / `gitlab` / `memory` / `review` with `extra="ignore"`. A 1.2.4 `codespy.yaml` is silently ignored for every moved key. Migration:
  - Top-level `default_model`, `extraction_model`, `default_max_iters`, `default_max_llm_calls`, `default_reasoning_effort`, `default_temperature`, `default_max_tokens` → `llm.*`. `llm_retries`/`llm_timeout` → `llm.retries`/`llm.timeout`. `rlm_fallback` → `llm.rlm_fallback`.
  - `signatures.<sig>` → `review.<sig>`. `min_confidence`, `output_format`, `output_stdout`, `output_git`, `cache_dir`, `excluded_directories` → `review.*`.
  - `memory.default_enabled` → `memory.enabled`. `memory.{compact_trajectory,max_trajectory_tokens,max_question_tokens,distiller,cartographer}` → `memory.hippocampus.*`. `memory.max_context_memory_tokens`/`max_context_item_tokens` → `memory.hippocampus.max_hippocampus_tokens`/`max_hippocampus_item_tokens`.
  - Env vars: `<SIG>_*` → `REVIEW_<SIG>_*`. `MIN_CONFIDENCE`, `OUTPUT_FORMAT`, `OUTPUT_STDOUT`, `OUTPUT_GIT`, `CACHE_DIR`, `EXCLUDED_DIRECTORIES` → `REVIEW_*`. `MEMORY_DEFAULT_ENABLED` → `MEMORY_ENABLED`. `MEMORY_MAX_CONTEXT_MEMORY_TOKENS`/`MEMORY_MAX_CONTEXT_ITEM_TOKENS` → `MEMORY_MAX_HIPPOCAMPUS_TOKENS`/`MEMORY_MAX_HIPPOCAMPUS_ITEM_TOKENS`.
  - Unchanged: `DEFAULT_*`, `EXTRACTION_MODEL`, `LLM_RETRIES`, `LLM_TIMEOUT`, `RLM_FALLBACK_*`, `MEMORY_DISTILLER_*`, `MEMORY_CARTOGRAPHER_*`, `MEMORY_POSTGRES_*` (except `SCHEMA`), `MEMORY_PG0_*`, `MEMORY_BANK_ID`, `MEMORY_COMPACT_TRAJECTORY`, `MEMORY_MAX_TRAJECTORY_TOKENS`, `MEMORY_MAX_QUESTION_TOKENS`, and provider keys.
- **New default models and temperature** — out-of-the-box codespy now requires AWS Bedrock access to Opus 5.5 and both Nemotron models:
  - `DEFAULT_MODEL`: `bedrock/converse/nvidia.nemotron-nano-3-30b`
  - `EXTRACTION_MODEL`: `bedrock/converse/nvidia.nemotron-nano-3-30b`
  - `REVIEW_SCOPE_MODEL`, `REVIEW_CODE_REVIEW_MODEL`, `REVIEW_DOC_MODEL`, `REVIEW_AUDIT_MODEL`: `bedrock/converse/global.anthropic.claude-opus-5-5`
  - `REVIEW_SUPPLY_CHAIN_MODEL`, `REVIEW_SUMMARY_MODEL`: `bedrock/converse/nvidia.nemotron-super-3-120b`
  - `MEMORY_*_MODEL` (distiller, cartographer, retain, consolidation, mental_models, recall): `bedrock/converse/nvidia.nemotron-super-3-120b`
  - `MEMORY_EMBEDDINGS_MODEL`: `bedrock/cohere.embed-v4:0`
  - `DEFAULT_TEMPERATURE`: `1.0`
- **Embeddings model v4** changes the vector dimension — requires a semantic DB reset (`DROP SCHEMA semantic CASCADE`)
- **Default `MEMORY_BANK_ID` changed from `codespy` to `codebase`** — set `MEMORY_BANK_ID=codespy` to keep using existing memory data
- **Memory disabled by default** — `memory.enabled` now defaults to `false`. Previously `review.summary.memory.enabled: true` in main `codespy.yaml` caused summary episodes to save even with global memory off. Now only the global switch gates memory, so nothing is saved by default.
- **Removed `memory.postgres.schema`** — episodic schema is now hard-coded to `episodic` (like `semantic`). The `MEMORY_POSTGRES_SCHEMA` env var is removed.
- **Action `model` input is now optional** (was `required: true`); when unset, codespy.yaml default applies. Action inputs no longer carry `default:` values; codespy.yaml applies instead. Setting a pinned unit to `null` (YAML `model: null` or env `VAR=null`) re-enables the fallback chain. Migration note: upgrade the action ref and `codespy-version` together. An older action with the `latest` image, or the reverse, silently drops review settings.
- **EVICT tombstones** — Hippocampus eviction writes `EVICT` mutation tombstones (`MutationType.EVICT`, separate from `DELETE`). `load_context` excludes both `DELETE` and `EVICT`. ROLLBACK NOTE: loaders older than this release only skip `DELETE` and fail to load a topic that has an `EVICT` row. Before rolling back, run `UPDATE observations SET op_type = 'DELETE' WHERE op_type = 'EVICT';` (both are content-NULL tombstones, so no data is lost).
- **Dependencies**: `hindsight-api-slim >=0.10.1` (new); `mcp` upgraded from `>=1.29,<2` to `>=2.2,<3`

### Added

- **Cerebral (Semantic Memory)**: New `codespy/agents/memory/cerebral/` package with mutation-based retain (ADD/`supersedes`/`RETRACTED`), domain-agnostic missions, tags/scopes (`project_scope:`, `repo:`, `org:`, `task:`), once-per-run consolidation via `SyncTaskBackend`, `max_observations_per_scope`, mental-model briefings in delta mode with `memory.cerebral.mental_models.*`, embeddings + `max_input_chars`, structured-output support, read API (`arecall`/`recall`, `areflect`/`reflect`, `aget_mental_models`, `aget_observation_history`), and fixed `semantic` schema.
- **Prefrontal (Semantic Recall)**: New `codespy/agents/memory/prefrontal/` package with run-level recall shared by summary, code_review, doc, supply_chain, and audit agents. Before each agent call it reads Cerebral and injects a read-only `prefrontal_memory` input (mental-model briefings, five question-shaped recall facets — context, seen before, decisions, recurring patterns, what changed — and facts from other repositories). Configured under `memory.prefrontal` (`MEMORY_PREFRONTAL_REFLECTS`, `MEMORY_RECALL_*`). The `recall_memory` tool is available to RLM agents (code_review, scope, supply_chain) but is off by default (`max_tool_calls: 0`). Prefrontal never writes to Cerebral and never feeds Hippocampus.
- **Review memories section**: The pre-call Prefrontal memory (run-level and scope-level) now appears in a collapsed `memories` section **before** the Summary. Run-level sections (Briefings, Around this work, Seen before, Decisions, Recurring patterns, What changed, Other repositories) render directly under `memories`. Scope recall, when present, is nested in its own `scope` block. All levels are indented with `<blockquote>` so nesting is visible in GitHub/GitLab. This section is shown in both the GitHub/GitLab review body and the stdout/MCP markdown output.
- **New action inputs** (no defaults, values taken from codespy.yaml when unset):
  - `scope-max-tokens`, `code-review-max-tokens`, `doc-max-tokens`, `supply-chain-max-tokens`, `summary-max-tokens`, `audit-max-tokens`
  - `memory-distiller-max-tokens`, `memory-cartographer-max-tokens`
  - `memory-compact-trajectory`, `memory-max-hippocampus-tokens`, `memory-max-hippocampus-item-tokens`, `memory-max-trajectory-tokens`, `memory-max-question-tokens`
  - `memory-retain-model`, `memory-retain-chunk-size`
  - `memory-consolidation-model`, `memory-consolidation-max-observations-per-scope`
  - `memory-mental-models-model`, `memory-mental-models-max-tokens`, `memory-mental-models-min-refresh-seconds`
  - `memory-recall-model`, `memory-recall-reach`, `memory-recall-max-tokens`, `memory-recall-max-tool-tokens`, `memory-recall-max-tool-calls`
  - `memory-prefrontal-reflects`, `memory-embeddings-model`, `memory-embeddings-max-input-chars`
  - `scope-skip-refinement-when-clean`
  - `openai-api-base`, `azure-api-key`, `azure-api-base`, `azure-api-version`
- **AWS region forwarding**: AWS region is now forwarded independently of access key (was bundled with credentials)
- **Cost buckets**: `memory_recall`, `memory_recall_embeddings`, `memory_consolidation`, `memory_mental_models`
- **ReflectSummary monitoring**: `areflect` returns `(text, ReflectSummary)` with iterations, LLM calls, map calls, rewrite flag, tools used, usage, and empty flag. Stored in `RecallRecord.details["reflect"]` for monitoring.

### Changed

- **Review statistics matrix**: the GitHub/GitLab review Statistics table is now a severity × category matrix (Security, Bugs, Documentation, Smells + totals); Smells are now counted.
- **Review publish order**: Episode saves, retain, consolidation and mental-model refresh run after audit. For remote reviews, the GitHub/GitLab review is posted first with "Memory: pending", then edited with full costs after the memory phase completes. Local CLI and MCP reviews run the memory phase before returning/printing.
- **Cost breakdown split**: The per-signature cost table is split into a Review table and a Memory table (memory_* buckets), each with its own subtotal.
- **Cost table rows renamed** (1.2.4 rows `distiller`/`cartographer` → `memory_distiller`/`memory_cartographer`)
- **Cerebral tags**: Removed `episode:` and `run_id:` tags — they were not used for scoping or reach and bloated the tag set. `pull_request:` tags are also no longer added. Retained items now only carry `project_scope:`, `repo:`, `org:`, and `task:` tags.
- **Cerebral fallback scope**: Episodes without a `project_scope` topic now consolidate into `[org:, repo:, project_scope:<owner/repo>]` (the repo-root project scope) instead of `[org:, repo:]`. This ensures every write scope matches the observation cap rule and creates/updates the repo-root briefing.
- **Cerebral retain framing**: Moved from `context` to `metadata`. The `context` field is no longer set on retain items; instead `metadata` carries `task`, `repo` (when known), `kind` (`"observation changes"` / `"artifacts"`), and `question`. This reduces consolidation input size.
- **Doc review CHANGELOG trim**: CHANGELOG files are now trimmed to preamble + 2 newest sections (keeps `[Unreleased]` + latest release). Saves ~25KB (21%) of the ~118KB doc payload in this repo.
- **load_context semantics**: Now loads the latest version of every observation from ALL prior episodes (not just the latest episode), excluding tombstones (`DELETE`, `EVICT`) and any version with no content. Observations are returned oldest-first, so eviction drops the oldest facts first. Topic prefix matching now uses proper LIKE escaping so `owner/repo` no longer matches `owner/repo-other`.
- **Cerebral DELETE vs EVICT handling**: DELETE produces a `RETRACTED` line marking facts as wrong; EVICT of unchanged prior facts sends nothing; content new this run (ADD→EVICT or REPLACE→EVICT) is retained to avoid losing newly learned facts.

### Removed

- **Per-signature memory overrides** — `review.<sig>.memory.enabled` (YAML), `REVIEW_<SIG>_MEMORY_ENABLED` (env), and action inputs `scope-memory-enabled`, `code-review-memory-enabled`, `doc-memory-enabled`, `supply-chain-memory-enabled`, `summary-memory-enabled`, `audit-memory-enabled` are removed. Memory is now controlled only by the global `memory.enabled` (`MEMORY_ENABLED`, `memory-enabled`).
- Action input `memory-prefrontal-model` (was deprecated, never wired). The real input is `memory-recall-model`.
- Action env vars: `SCOPE_MEMORY_ENABLED`, `CODE_REVIEW_MEMORY_ENABLED`, `DOC_MEMORY_ENABLED`, `SUPPLY_CHAIN_MEMORY_ENABLED`, `SUMMARY_MEMORY_ENABLED`, `AUDIT_MEMORY_ENABLED`, `OUTPUT_FORMAT`
- Config settings: `enable_prompt_caching`/`ENABLE_PROMPT_CACHING`/`enable-prompt-caching`, `compact_patches`/`COMPACT_PATCHES`/`compact-patches`
- Tree-sitter patch compaction (`tools/git/patch_utils.py` deleted)
- Dead code: `Settings.sync_llm_settings()`, `LLMConfig.sync_from_flat()`, `Settings.get_memory_enabled()`, `get_memory_budget()` no longer takes an argument, `MemorySignatureConfig`, `SIGNATURE_PREFIXES`, `SIGNATURE_SETTINGS`, `MEMORY_SIGNATURE_SETTINGS`, `RLM_FALLBACK_ENV_SETTINGS`
- Agents no longer receive memory input: `inject_context_memory` and the `context_memory=` argument have been removed from all six review agents (summary, scope, code_review, doc, supply_chain, audit). Memory is learned but not used directly; prior knowledge now reaches agents only through Prefrontal.

### Fixed

- `recall_memory` with `reach=local` returned nothing for code_review and supply_chain; the tool now searches the reviewed project scope
- "too many values to unpack" error in review statistics matrix (line 345 was iterating `severities` as tuples instead of enums)
- `GITLAB_URL` from the environment was ignored — now properly takes precedence over YAML `gitlab.url`
- Token precedence now matches documentation: env/`.env` beats YAML (`github.token` / `gitlab.token`)
- Removed action `llm-timeout` default (was `'120'`, now uses codespy.yaml's `240`)
- Doc reviewer defaults now match codespy.yaml (`max_iters: 2`, `max_llm_calls: 4`)
- Fixed workflow example input: `summarization-model` → `summary-model`
- Prefrontal reads on a bank with nothing retained yet (a new `MEMORY_BANK_ID`) now return empty results instead of failing with `Bank '<id>' not found`
- Retain no longer fails with "all N facts returned by the LLM were unusable" on Nemotron 3 Super (Bedrock) — Cerebral now registers the structured-output capability and strips JSON Schema keywords Bedrock's native structured output rejects
- Briefing (mental model) delta refreshes no longer fail with "operations must be a list, got <class 'str'>" — Cerebral now decodes string-encoded operations before Hindsight validates them
- `Missing required inputs: ['prefrontal_memory']` regression: Fixed a bug where agents failed when the Prefrontal instance existed but the shared run-level `prefrontal_memory` text was empty (fresh DB). The signature is now chosen based on whether there is text to pass
- Cerebral consolidation and retain LLM calls now use `llm.timeout` / `llm.retries`. Previously only reflect did, so consolidation timed out at Hindsight's 120 s default.
- Cerebral consolidation no longer logs `relation "webhooks" does not exist` — inline (`SyncTaskBackend`) tasks now carry the `semantic` schema
- Cerebral consolidation no longer fails at exit with `cannot schedule new futures after shutdown` — background episode saves now finish before Python shuts down its thread pools
- Briefings no longer stay stuck on "Generating content..." after a failed refresh
- Embedding inputs are now capped for Bedrock Cohere v3 (2048 characters) to prevent `maxLength` errors
- Cost calculation: `TwoStepAdapter` extraction calls are now included in signature cost tracking
- Cost billing: DSPy cache hits (`response.cache_hit`) are now skipped in cost accounting
- Cost split calculation: When `entry["cost"]` is available (billed cost), input cost is now calculated as `billed_cost - output_cost`
- `DOC_MAX_ITERS`/`DOC_MAX_LLM_CALLS` env passthrough now works
- `memory-enabled`→`MEMORY_ENABLED` mapping in action

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
