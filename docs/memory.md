[← Back to README](../README.md#documentation)

# Hippocampus Memory System

## Overview

Episode-based memory that wraps DSPy agents with persistent context across reviews.
Agents accumulate knowledge about a codebase scope over time — patterns, constants,
parsing schemas — and reuse it in subsequent reviews of the same code area.

## Concepts

### Topics

- Every scope gets a `topic_id` derived from `make_topic_id(repo_slug, subroot)`
- Topics organize memory by code area so knowledge doesn't bleed between scopes
- `compute_common_ancestor_topic_id()` finds shared parent for cross-scope queries

### Episodes

- An Episode captures one agent's run: task, context_memory, mutations, timestamp
- Stored in PostgreSQL (auto-created tables). pg0-embedded auto-starts
  a local instance when no `MEMORY_POSTGRES_URI` is set.
- `find_latest_episode()` loads the most recent episode by `modified_at` for a given path prefix

### Context Memory

Six sections (from general to specific):

1. **`context_roadmap`** — High-level codebase structure and navigation hints
2. **`context_understanding`** — Domain knowledge and design patterns observed
3. **`domain_constants`** — Exact values, URLs, identifiers that repeat across reviews
4. **`actions`** — Tool execution patterns and action history
5. **`parsing_schema`** — File format conventions, naming patterns, structural rules
6. **`reusable_results`** — Computed facts reusable in future reviews

Each section contains Items with tags (general, scope-specific) and text content.

## Reflection Pipeline

After each agent run (at `end_episode()`):

1. **Distiller** — Analyzes the agent's trajectory (head 60% + tail 40%, capped at `max_trajectory_tokens`) and proposes `CacheCandidate` items for context memory
2. **Cartographer** — Takes candidates + current context memory, decides operations:
   - `ADD` — Insert new item
   - `REPLACE` — Update existing item with new knowledge
   - `DELETE` — Remove outdated/irrelevant item
3. **Eviction** — If memory exceeds `max_context_memory_tokens`, oldest general items are evicted first

Reflection iterates `max_reflects` times (0 = reflect once at end_episode).

## Token Budgets

| Budget | Env Var | Default | Purpose |
|--------|---------|---------|---------|
| Context memory | `MEMORY_DEFAULT_MAX_CONTEXT_MEMORY_TOKENS` | 16384 | Ceiling on persisted ContextMemory (re-sent every iteration) |
| Item | `MEMORY_DEFAULT_MAX_CONTEXT_ITEM_TOKENS` | 512 | Soft per-item limit (expressed to LLM, not truncated) |
| Trajectory | `MEMORY_DEFAULT_MAX_TRAJECTORY_TOKENS` | 16384 | Head+tail cap on trajectory fed to Distiller |
| Question | `MEMORY_DEFAULT_MAX_QUESTION_TOKENS` | 8192 | Cap on serialized inputs as reflection question |

Item capacity ≈ context_memory_tokens / item_tokens (16384/512 = 32 items)

## Configuration

### Global Settings

| Env Var | YAML Path | Default | Description |
|---------|-----------|---------|-------------|
| `MEMORY_POSTGRES_URI` | `memory.postgres_uri` | — | External PostgreSQL connection URI |
| `MEMORY_BANK_ID` | `memory.bank_id` | `codespy` | Scopes all memory data |
| `MEMORY_PG0_NAME` | `memory.pg0_name` | `codespy` | pg0-embedded database name |
| `MEMORY_PG0_PORT` | `memory.pg0_port` | auto | pg0-embedded port |
| `MEMORY_DEFAULT_ENABLED` | `memory.default_enabled` | `false` | Enable memory globally |
| `MEMORY_DEFAULT_MAX_REFLECTS` | `memory.default_max_reflects` | `0` | Reflection iterations |

### Reflection Module LLM Overrides

| Module | Env Var Pattern | YAML Path |
|--------|----------------|-----------|
| Distiller | `MEMORY_DISTILLER_{MODEL,REASONING_EFFORT,TEMPERATURE,MAX_TOKENS,MAX_ITERS,MAX_LLM_CALLS}` | `memory.distiller.*` |
| Cartographer | `MEMORY_CARTOGRAPHER_{MODEL,REASONING_EFFORT,TEMPERATURE,MAX_TOKENS,MAX_ITERS,MAX_LLM_CALLS}` | `memory.cartographer.*` |

### Per-Signature Memory Overrides

Each signature's `memory:` block in YAML (or `<SIGNATURE>_MEMORY_*` env vars):

| Setting | Env Var Suffix | Description |
|---------|---------------|-------------|
| enabled | `_MEMORY_ENABLED` | Enable/disable memory for this signature |
| max_reflects | `_MEMORY_MAX_REFLECTS` | Override reflection count |

Example: `CODE_REVIEW_MEMORY_ENABLED=true` or `SUMMARY_MEMORY_MAX_REFLECTS=2`

See [Configuration](configuration.md#recommended-model-strategy) for recommended reflection models.

## Quick Start

Enable memory for code review:
```bash
MEMORY_DEFAULT_ENABLED=true
# Or per-signature:
CODE_REVIEW_MEMORY_ENABLED=true
SUMMARY_MEMORY_ENABLED=true
```

Recommended mid-tier reflection model:
```bash
MEMORY_DISTILLER_MODEL=anthropic/claude-sonnet-4-5-20250929
MEMORY_CARTOGRAPHER_MODEL=anthropic/claude-sonnet-4-5-20250929
```

# Storage: pg0-embedded auto-starts when pg0-embedded is installed (default).
# For production, set MEMORY_POSTGRES_URI:
# MEMORY_POSTGRES_URI=postgresql://user:pass@host:5432/codespy

### GitHub Action

```yaml
- name: Run CodeSpy Review
  uses: khezen/codespy@v1
  with:
    model: 'anthropic/claude-opus-4-6'
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    # Memory with external PostgreSQL
    memory-enabled: 'true'
    memory-postgres-uri: ${{ secrets.MEMORY_POSTGRES_URI }}
    memory-distiller-model: 'anthropic/claude-haiku-4-5-20251001'
    memory-cartographer-model: 'anthropic/claude-haiku-4-5-20251001'
```

> **Note:** pg0-embedded is included in the Docker image. For persistent memory
> across CI runs, use an external PostgreSQL instance via `memory-postgres-uri`.

---

[← Back to README](../README.md#documentation)
