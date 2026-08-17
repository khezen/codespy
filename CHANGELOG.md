# Changelog

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
