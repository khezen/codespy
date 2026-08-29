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
| Model | `DEFAULT_MODEL` | `anthropic/claude-opus-4-6` | Primary model for all signatures |
| Reasoning effort | `DEFAULT_REASONING_EFFORT` | `medium` | Provider reasoning budget: `minimal`, `low`, `medium`, `high` |
| Max tokens | `DEFAULT_MAX_TOKENS` | `64000` | Output token budget per completion (reasoning tokens included) |
| Temperature | `DEFAULT_TEMPERATURE` | `0.2` | Default temperature for LLM calls |
| Max iterations | `DEFAULT_MAX_ITERS` | `4` | Maximum ReAct iterations for tool-using agents |
| Prompt caching | `ENABLE_PROMPT_CACHING` | `true` | Provider-side prompt caching (Anthropic, OpenAI, Bedrock) |
| RLM fallback | `RLM_FALLBACK_ENABLED` | `true` | Proactive RLM fallback for context rot prevention |
| RLM react threshold | `RLM_FALLBACK_REACT_THRESHOLD` | `0.30` | Context ratio triggering RLM for ReAct modules |
| RLM CoT threshold | `RLM_FALLBACK_CHAIN_OF_THOUGHT_THRESHOLD` | `0.40` | Context ratio triggering RLM for ChainOfThought modules |
| RLM predict threshold | `RLM_FALLBACK_PREDICT_THRESHOLD` | `0.50` | Context ratio triggering RLM for Predict modules |

## Recommended Model Strategy

| Tier | Role | Env Var | Default | Recommended |
|------|------|---------|---------|-------------|
| Smart | Core analysis & reasoning | `DEFAULT_MODEL` | `anthropic/claude-opus-4-6` | Claude Opus / GPT-5 |
| Mid-tier | Field extraction | `EXTRACTION_MODEL` | Falls back to DEFAULT_MODEL | Claude Sonnet |
| Cheap | PR summary | `SUMMARY_MODEL` | Falls back to DEFAULT_MODEL | Claude Haiku |
| Mid-tier | Memory reflection | `MEMORY_DISTILLER_MODEL` / `MEMORY_CARTOGRAPHER_MODEL` | Falls back to DEFAULT_MODEL | Claude Sonnet |

## Per-Signature Configuration

Each signature supports env var overrides: `<SIGNATURE>_<SETTING>`

| Signature | Config Key | Available Settings |
|-----------|------------|-------------------|
| Scope Identifier | `scope` | ENABLED, MAX_ITERS, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS |
| PR Summary | `summary` | ENABLED, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS |
| Code Reviewer | `code_review` | ENABLED, MAX_ITERS, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS |
| Doc Reviewer | `doc` | ENABLED, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS |
| Supply Chain | `supply_chain` | ENABLED, MAX_ITERS, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS, SCAN_UNCHANGED |
| Auditor | `audit` | ENABLED, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS |

Example: `CODE_REVIEW_MODEL=anthropic/claude-sonnet-4-5-20250929`

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
export CODE_REVIEW_MODEL=anthropic/claude-sonnet-4-5-20250929

# Output settings
export OUTPUT_STDOUT=false
export OUTPUT_GIT=true
```

## Memory Configuration

Brief overview:

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Backend | `MEMORY_BACKEND` | `filesystem` | Storage: `filesystem` or `s3` |
| Root path | `MEMORY_ROOT` | `~/.cache/codespy/memory` | Filesystem storage location |
| Default enabled | `MEMORY_DEFAULT_ENABLED` | `false` | Enable memory globally |
| Max reflects | `MEMORY_DEFAULT_MAX_REFLECTS` | `0` | Reflection iterations (0 = once at end) |
| Context memory tokens | `MEMORY_DEFAULT_MAX_CONTEXT_MEMORY_TOKENS` | `8192` | Max tokens for persisted context memory |
| Item tokens | `MEMORY_DEFAULT_MAX_CONTEXT_ITEM_TOKENS` | `410` | Soft per-item token limit |
| Trajectory tokens | `MEMORY_DEFAULT_MAX_TRAJECTORY_TOKENS` | `8192` | Cap on trajectory fed to Distiller |
| Question tokens | `MEMORY_DEFAULT_MAX_QUESTION_TOKENS` | `2048` | Cap on serialized reflection inputs |

See [Memory System](memory.md) for full memory configuration details.

## Output Settings

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Format | `OUTPUT_FORMAT` | `markdown` | `markdown` or `json` |
| Stdout | `OUTPUT_STDOUT` | `true` | Enable stdout output |
| Git | `OUTPUT_GIT` | `true` | Post review to GitHub/GitLab |
| Cache dir | `CACHE_DIR` | `~/.cache/codespy` | Cache directory path |

## File Exclusions

`EXCLUDED_DIRECTORIES` (JSON array in env) — Directories to skip during code review. Binary files, lock files, and minified files are always excluded automatically.

Default excluded directories:
- Vendor/dependency: `vendor`, `node_modules`, `third_party`, `external`, `deps`, `_vendor`, `vendored`
- Build output: `dist`, `build`, `out`, `target`
- Package manager: `.bundle`, `Pods`, `Carthage`, `bower_components`, `jspm_packages`
- Version control: `.git`, `.svn`, `.hg`
- Cache: `__pycache__`, `.cache`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`

---

[← Back to README](../README.md#documentation)
