[← Back to README](../README.md#documentation)

# Configuration

Priority: CLI options > Environment Variables > YAML Config > Defaults

## Setup

```bash
cp .env.example .env
```

## Git Platform Tokens

### GitHub Token

Auto-discovered from:
- `GITHUB_TOKEN` or `GH_TOKEN` environment variables
- GitHub CLI (`gh auth token`)
- Git credential helper
- `~/.netrc` file

Or create a token at https://github.com/settings/tokens with `repo` scope:
```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

To disable auto-discovery:
```bash
GITHUB_AUTO_DISCOVER_TOKEN=false
```

### GitLab Token

Auto-discovered from:
- `GITLAB_TOKEN` or `GITLAB_PRIVATE_TOKEN` environment variables
- GitLab CLI (`glab auth token`)
- Git credential helper
- `~/.netrc` file
- python-gitlab config files (`~/.python-gitlab.cfg`, `/etc/python-gitlab.cfg`)

Or create a token at https://gitlab.com/-/user_settings/personal_access_tokens with `api` scope:
```bash
GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx
```

For self-hosted GitLab:
```bash
GITLAB_URL=https://gitlab.mycompany.com
GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx
```

To disable auto-discovery:
```bash
GITLAB_AUTO_DISCOVER_TOKEN=false
```

## LLM Provider

codespy auto-discovers credentials for all providers:

**Anthropic** (auto-discovers from `$ANTHROPIC_API_KEY`, `~/.config/anthropic/`, `~/.anthropic/`):
```bash
DEFAULT_MODEL=anthropic/claude-opus-4-6
# Optional - set explicitly or let codespy auto-discover:
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
```

**AWS Bedrock** (auto-discovers from `~/.aws/credentials`, AWS CLI, env vars):
```bash
DEFAULT_MODEL=bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
AWS_REGION=us-east-1
# Optional - uses ~/.aws/credentials by default, or set explicitly:
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
```

**OpenAI** (auto-discovers from `$OPENAI_API_KEY`, `~/.config/openai/`, `~/.openai/`):
```bash
DEFAULT_MODEL=openai/gpt-5
# Optional - set explicitly or let codespy auto-discover:
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

**Google Gemini** (auto-discovers from `$GEMINI_API_KEY`, `$GOOGLE_API_KEY`, gcloud ADC):
```bash
DEFAULT_MODEL=gemini/gemini-2.5-pro
# Optional - set explicitly or let codespy auto-discover:
# GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxx
```

**Local Ollama:**
```bash
DEFAULT_MODEL=ollama/llama3
```

To disable auto-discovery for specific providers:
```bash
AUTO_DISCOVER_AWS=false
AUTO_DISCOVER_OPENAI=false
AUTO_DISCOVER_ANTHROPIC=false
AUTO_DISCOVER_GEMINI=false
```

## Model Settings

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Model | `DEFAULT_MODEL` | `bedrock/converse/nvidia.nemotron-nano-3-30b` | Primary fallback model (most signatures have dedicated pins) |
| Reasoning effort | `DEFAULT_REASONING_EFFORT` | `medium` | Provider reasoning budget: `minimal`, `low`, `medium`, `high` |
| Max tokens | `DEFAULT_MAX_TOKENS` | `32000` | Output token budget per completion (reasoning tokens included) |
| Temperature | `DEFAULT_TEMPERATURE` | `1.0` | Default temperature for LLM calls |
| Max iterations | `DEFAULT_MAX_ITERS` | `5` | Maximum ReAct iterations for tool-using agents |
| RLM fallback | `RLM_FALLBACK_ENABLED` | `true` | Proactive RLM fallback for context rot prevention |
| RLM react threshold | `RLM_FALLBACK_REACT_THRESHOLD` | `0.30` | Context ratio triggering RLM for ReAct modules |
| RLM CoT threshold | `RLM_FALLBACK_CHAIN_OF_THOUGHT_THRESHOLD` | `0.40` | Context ratio triggering RLM for ChainOfThought modules |
| RLM predict threshold | `RLM_FALLBACK_PREDICT_THRESHOLD` | `0.50` | Context ratio triggering RLM for Predict modules |

## Recommended Model Strategy

All models are now pinned in `codespy.yaml`. Setting a unit to `null` (YAML `model: null` or env `VAR=null`) re-enables the fallback chain.

| Tier | Role | Env Var | Default | Fallback chain |
|------|------|---------|---------|----------------|
| Smart | Core analysis & reasoning | `REVIEW_*_MODEL` | `bedrock/converse/global.anthropic.claude-opus-5-5` | `model` → `DEFAULT_MODEL` |
| Mid-tier | Field extraction | `EXTRACTION_MODEL` | `bedrock/converse/nvidia.nemotron-nano-3-30b` | `DEFAULT_MODEL` |
| Cheap | PR summary | `REVIEW_SUMMARY_MODEL` | `bedrock/converse/nvidia.nemotron-super-3-120b` | `model` → `DEFAULT_MODEL` |
| Mid-tier | Memory reflection | `MEMORY_DISTILLER_MODEL` / `MEMORY_CARTOGRAPHER_MODEL` | `bedrock/converse/nvidia.nemotron-super-3-120b` | `model` → `DEFAULT_MODEL` |
| Mid-tier | Semantic memory (fact extraction) | `MEMORY_RETAIN_MODEL` | `bedrock/converse/nvidia.nemotron-super-3-120b` | `DEFAULT_MODEL` |
| Mid-tier | Semantic consolidation | `MEMORY_CONSOLIDATION_MODEL` | `bedrock/converse/nvidia.nemotron-super-3-120b` | `MEMORY_RETAIN_MODEL` → `DEFAULT_MODEL` |
| Mid-tier | Mental-model refresh (briefings) | `MEMORY_MENTAL_MODELS_MODEL` | `bedrock/converse/nvidia.nemotron-super-3-120b` | `MEMORY_RECALL_MODEL` → `MEMORY_RETAIN_MODEL` → `DEFAULT_MODEL` |
| Mid-tier | Semantic recall (reflect) | `MEMORY_RECALL_MODEL` | `bedrock/converse/nvidia.nemotron-super-3-120b` | `MEMORY_RETAIN_MODEL` → `DEFAULT_MODEL` |

**Precedence:** Environment variables (`.env` or shell) > `codespy.yaml` (`github.token` / `gitlab.token`) > auto-discovery.

## Per-Signature Configuration

Each signature supports env var overrides: `REVIEW_<SIGNATURE>_<SETTING>`

| Signature | Config Key | Available Settings |
|-----------|------------|-------------------|
| Scope Identifier | `scope` | ENABLED, MAX_ITERS, MAX_LLM_CALLS, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS |
| PR Summary | `summary` | ENABLED, MAX_ITERS, MAX_LLM_CALLS, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS |
| Code Reviewer | `code_review` | ENABLED, MAX_ITERS, MAX_LLM_CALLS, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS |
| Doc Reviewer | `doc` | ENABLED, MAX_ITERS, MAX_LLM_CALLS, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS |
| Supply Chain | `supply_chain` | ENABLED, MAX_ITERS, MAX_LLM_CALLS, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS, SCAN_UNCHANGED |
| Auditor | `audit` | ENABLED, MAX_ITERS, MAX_LLM_CALLS, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS |

Example: `REVIEW_CODE_REVIEW_MODEL=anthropic/claude-sonnet-4-5-20250929`

## Advanced Configuration (YAML)

For per-signature settings, use `codespy.yaml`. See [`codespy.yaml`](../codespy.yaml) for all available options including:
- LLM provider settings and auto-discovery
- Git platform configuration (GitHub/GitLab)
- Per-signature model and iteration overrides
- Output format and destination settings
- Directory exclusions

Override YAML settings via environment variables using `_` separator:

```bash
# Default settings
export DEFAULT_MODEL=anthropic/claude-opus-4-6
export DEFAULT_MAX_ITERS=20

# Per-signature settings (use signature name, not module name)
export REVIEW_CODE_REVIEW_MODEL=anthropic/claude_sonnet-4-5-20250929

# Output settings
export REVIEW_OUTPUT_STDOUT=false
export REVIEW_OUTPUT_GIT=true
```

## Memory Configuration

The memory system has three components:

- **Hippocampus**: Episodic memory (learns from the current review)
- **Cerebral**: Semantic memory (retains episodes, consolidates, refreshes mental models)
- **Prefrontal**: Reads Cerebral and injects prior knowledge into agents

Master switch and connection:

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Enabled | `MEMORY_ENABLED` | `false` | Enable memory globally |
| Bank ID | `MEMORY_BANK_ID` | `codespy` | Scopes all memory data |
| PostgreSQL host | `MEMORY_POSTGRES_HOST` | — | External PostgreSQL host |
| PostgreSQL port | `MEMORY_POSTGRES_PORT` | `5432` | PostgreSQL port |
| PostgreSQL user | `MEMORY_POSTGRES_USER` | `postgres` | PostgreSQL user |
| PostgreSQL password | `MEMORY_POSTGRES_PASSWORD` | — | PostgreSQL password |
| PostgreSQL database | `MEMORY_POSTGRES_DATABASE` | `codespy` | PostgreSQL database |
| PostgreSQL schema | `MEMORY_POSTGRES_SCHEMA` | `episodic` | Schema for episodic store |
| pg0 name | `MEMORY_PG0_NAME` | `codespy` | pg0-embedded database name |
| pg0 port | `MEMORY_PG0_PORT` | auto | pg0-embedded port |
| pg0 data dir | `MEMORY_PG0_DATA_DIR` | — | pg0-embedded data directory |

See [Memory System](memory.md) for:
- Full configuration tables (token budgets, models, per-signature overrides)
- Database schema and SQL examples
- Lifecycle, data flow, and monitoring
- Prefrontal recall settings (reach, reflects, facets, `recall_memory` tool)

## Output Settings

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Format | `REVIEW_OUTPUT_FORMAT` | `markdown` | `markdown` or `json` |
| Stdout | `REVIEW_OUTPUT_STDOUT` | `true` | Enable stdout output |
| Git | `REVIEW_OUTPUT_GIT` | `true` | Post review to GitHub/GitLab |
| Cache dir | `REVIEW_CACHE_DIR` | `~/.cache/codespy` | Cache directory path |

## File Exclusions

`REVIEW_EXCLUDED_DIRECTORIES` (JSON array in env) — Directories to skip during code review. Binary files, lock files, and minified files are always excluded automatically.

Default excluded directories:
- Vendor/dependency: `vendor`, `node_modules`, `third_party`, `external`, `deps`, `_vendor`, `vendored`
- Build output: `dist`, `build`, `out`, `target`
- Package manager: `.bundle`, `Pods`, `Carthage`, `bower_components`, `jspm_packages`
- Version control: `.git`, `.svn`, `.hg`
- Cache: `__pycache__`, `.cache`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`

---

[← Back to README](../README.md#documentation)
