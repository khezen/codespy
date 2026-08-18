# Changelog

## [Unreleased]

## [1.0.8] - 2026-08-18

### Security
- S3 StreamingBody resource leak fix: wrapped `resp["Body"].read()` in try/finally to ensure `body.close()` is called, preventing connection pool exhaustion on partial read failures
- SecretStr migration for all credential fields: tokens and API keys now use Pydantic `SecretStr` type to prevent accidental logging of sensitive values
  - Affected fields: `github_token`, `gh_token`, `gitlab_token`, `aws_access_key_id`, `aws_secret_access_key`, `openai_api_key`, `anthropic_api_key`, `gemini_api_key`
  - New `config_utils.secret_value()` helper extracts plain text at API boundaries
  - `model_dump()` masks secrets as `'**********'`
- GitLab client timeout: added `timeout=30` to `gitlab.Gitlab()` instantiation to prevent indefinite hangs (python-gitlab >=4.0.0 defaults to `None`)

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
