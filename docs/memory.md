[← Back to README](../README.md#documentation)

# Hippocampus Memory System

## Overview

Episode-based memory that wraps DSPy agents with persistent context across reviews.
Agents accumulate knowledge about a codebase scope over time — patterns, constants,
parsing schemas — and reuse it in subsequent reviews of the same code area.

## Concepts

### Banks

A bank is the top-level data partition. All episodes, topics, observations,
and artifacts cascade-delete from a bank.

- Configured via `MEMORY_BANK_ID` (default: `codespy`)
- Typical values: service name, team name, org identifier
- Changing the bank ID starts a fresh memory silo — no data carries over

### Topics

- Every scope gets a `topic_id` derived from `make_topic_id(repo_slug, subroot)`
- Topics organize memory by code area so knowledge doesn't bleed between scopes
- `compute_common_ancestor_topic_id()` finds shared parent for cross-scope queries

### Episodes

- An Episode captures one agent's run: task, context_memory, mutations, artifacts, timestamp
- Stored in PostgreSQL (auto-created tables). pg0-embedded auto-starts
  a local instance when no `MEMORY_POSTGRES_URI` is set.
- `EpisodeStore.load_context(task, topic_ids, topic_prefix)` retrieves
  the latest context by `timestamp DESC`, filtering on bank + task + topic

### Context Memory

Six sections (from general to specific):

1. **`context_roadmap`** — Index of what the context contains and where to find it
2. **`context_understanding`** — High-level understanding of the context
3. **`domain_constants`** — Exact parameters, formulas, thresholds, reference values, enum sets
4. **`actions`** — Tool execution patterns: what tool was used, for what purpose, and the result
5. **`parsing_schema`** — How to parse the context's format: delimiters, boundary patterns, field structure
6. **`reusable_results`** — Agent-derived aggregated outputs (counts, distributions, classifications) reusable across questions

Each section contains `Item` objects with `id`, `content`, and `topic_ids` linking
the item to its relevant scopes.

### Observations

Observations are the versioned audit trail of context memory items in the database.
Each time the Cartographer ADDs, REPLACEs, or DELETEs an item, a new observation
row is inserted with an incremented `version` number.

- `content` holds the new value (`NULL` for DELETE)
- `previous_content` preserves the prior state (for REPLACE / DELETE)
- `op_type` records the operation: `ADD`, `REPLACE`, or `DELETE`
- Linked to the episode that produced the mutation and the topics it belongs to

Items in `ContextMemory` map 1:1 to the latest non-deleted observation version.

### Artifacts

Named text outputs attached to an episode — for example, the final review
markdown or the PR summary text. Stored as `(episode_id, name) → content`.

## Database Schema

Auto-created by `EpisodeStore.ensure_schema()` on first connect.
Source: `src/codespy/agents/memory/postgres.py:61-211`.

### Entity Relationship Diagram

```mermaid
erDiagram
    schema_version {
        int version PK
        timestamptz applied_at "DEFAULT now()"
    }

    banks {
        varchar(64) id PK
        text description
    }

    topics {
        varchar(64) bank_id PK, FK
        varchar(256) id PK
        varchar(64) type
        text description
    }

    episodes {
        varchar(64) bank_id PK, FK
        uuid id PK
        varchar(48) run_id
        varchar(64) task
        varchar(64) module
        text question "DEFAULT ''"
        timestamptz timestamp
    }

    episode_topics {
        varchar(64) bank_id PK, FK
        uuid episode_id PK, FK
        varchar(256) topic_id PK, FK
    }

    observations {
        varchar(64) bank_id PK, FK
        varchar(48) id PK
        int version PK "DEFAULT 1"
        varchar(32) type
        text content
        uuid episode_id FK
        int step "DEFAULT 0"
        varchar(8) op_type "ADD | REPLACE | DELETE"
        text previous_content
        int ordinal "DEFAULT 0"
    }

    observation_topics {
        varchar(64) bank_id PK, FK
        varchar(48) observation_id PK, FK
        int observation_version PK, FK
        varchar(256) topic_id PK, FK
        int observation_occurrence "DEFAULT 0"
        int version_occurrence "DEFAULT 0"
    }

    episode_observations {
        varchar(64) bank_id PK, FK
        uuid episode_id PK, FK
        varchar(48) observation_id PK, FK
        int observation_version FK
    }

    artifacts {
        varchar(64) bank_id PK, FK
        uuid episode_id PK, FK
        text name PK
        text content
    }

    banks ||--o{ topics : "has"
    banks ||--o{ episodes : "has"
    banks ||--o{ observations : "has"
    episodes ||--o{ episode_topics : "tagged with"
    topics ||--o{ episode_topics : "tags"
    episodes ||--o{ observations : "creates"
    episodes ||--o{ artifacts : "produces"
    episodes ||--o{ episode_observations : "references"
    observations ||--o{ episode_observations : "referenced by"
    observations ||--o{ observation_topics : "scoped to"
    topics ||--o{ observation_topics : "scopes"
```

### Indexes

| Index | Table | Definition |
|-------|-------|------------|
| `idx_topics_id_trgm` | topics | `GIN (id gin_trgm_ops)` — trigram fuzzy search |
| `idx_episodes_task_time` | episodes | `(bank_id, task, timestamp DESC)` |
| `idx_episode_topics_reverse` | episode_topics | `(bank_id, topic_id)` |
| `idx_observations_episode` | observations | `(bank_id, episode_id)` |
| `idx_observation_topics_reverse` | observation_topics | `(bank_id, topic_id)` |
| `idx_episode_observations_reverse` | episode_observations | `(bank_id, observation_id)` |

### Notes

- All tables cascade-delete from `banks`
- `observations` is versioned: PK `(bank_id, id, version)` tracks ADD/REPLACE/DELETE history per item
- `episode_topics` and `observation_topics` are M:N junction tables
- `episode_observations` links episodes to the specific observation versions they reference
- `observation_topics.observation_occurrence` counts cumulative topic associations across all versions; `version_occurrence` counts within one version
- Requires PostgreSQL extension: `pg_trgm`

## Reflection Pipeline

After each agent run (at `end_episode()`):

1. **Distiller** — Analyzes the agent's trajectory (head 60% + tail 40%, capped at `max_trajectory_tokens`) and proposes `CacheCandidate` items for context memory
2. **Cartographer** — Takes candidates + current context memory, decides operations:
   - `ADD` — Insert new item
   - `REPLACE` — Update existing item with new knowledge
   - `DELETE` — Remove outdated/irrelevant item
3. **Eviction** — If memory exceeds `max_context_memory_tokens`, oldest general items are evicted first

The Distiller also tags each existing context memory item with an `ItemTag`:

- **`helpful`** — directly aided the agent; keep
- **`harmful`** — misled the agent or contradicted observations; remove
- **`neutral`** — present but unused this round; keep
- **`stale`** — no longer reflects the external context; remove

These tags inform the Cartographer's edit decisions.

Reflection iterates `max_reflects` times (0 = reflect once at end_episode).

## Token Budgets

| Budget | Env Var | Default | Purpose |
|--------|---------|---------|---------|
| Context memory | `MEMORY_MAX_CONTEXT_MEMORY_TOKENS` | 16384 | Ceiling on persisted ContextMemory (re-sent every iteration) |
| Item | `MEMORY_MAX_CONTEXT_ITEM_TOKENS` | 512 | Soft per-item limit (expressed to LLM, not truncated) |
| Trajectory | `MEMORY_MAX_TRAJECTORY_TOKENS` | 16384 | Head+tail cap on trajectory fed to Distiller |
| Question | `MEMORY_MAX_QUESTION_TOKENS` | 8192 | Cap on serialized inputs as reflection question |

Item capacity ≈ context_memory_tokens / item_tokens (16384/512 = 32 items)

## Configuration

### Global Settings

| Env Var | YAML Path | Default | Description |
|---------|-----------|---------|-------------|
| `MEMORY_POSTGRES_URI` | `memory.postgres_uri` | — | External PostgreSQL connection URI |
| `MEMORY_BANK_ID` | `memory.bank_id` | `codespy` | Scopes all memory data |
| `MEMORY_PG0_NAME` | `memory.pg0_name` | `codespy` | pg0-embedded database name |
| `MEMORY_PG0_PORT` | `memory.pg0_port` | auto | pg0-embedded port |
| `MEMORY_PG0_DATA_DIR` | `memory.pg0_data_dir` | — | Custom data directory for pg0-embedded |
| `MEMORY_DEFAULT_ENABLED` | `memory.default_enabled` | `false` | Enable memory globally |
| `MEMORY_DEFAULT_MAX_REFLECTS` | `memory.default_max_reflects` | `0` | Reflection iterations |
| `MEMORY_COMPACT_TRAJECTORY` | `memory.compact_trajectory` | `true` | Apply head+tail trajectory bounding before distillation |

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

> **Note:** pg0-embedded auto-starts when installed (default for local dev).
> For production, set `MEMORY_POSTGRES_URI`:
> ```
> MEMORY_POSTGRES_URI=postgresql://user:pass@host:5432/codespy
> ```

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
