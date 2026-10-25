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

`codespy review-path` writes to the repository (it opens the report PR). A fine-grained token needs `contents: write` and `pull-requests: write`. See [Full-Path Review](usage.md#token-permissions).

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
| SEO & AI-visibility | `seo` | ENABLED (default `false`), SCOPES, MAX_ITERS, MAX_LLM_CALLS, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS. See [SEO Review](#seo-review) |
| Auditor | `audit` | ENABLED, MAX_ITERS, MAX_LLM_CALLS, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS |
| Implementation Planner | `plan` | ENABLED (default `false`), MAX_ITERS, MAX_LLM_CALLS, MODEL, REASONING_EFFORT, TEMPERATURE, MAX_TOKENS. See [Implementation Plan](#implementation-plan) |

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
| Bank ID | `MEMORY_BANK_ID` | `codebase` | Scopes all memory data |
| PostgreSQL host | `MEMORY_POSTGRES_HOST` | — | External PostgreSQL host |
| PostgreSQL port | `MEMORY_POSTGRES_PORT` | `5432` | PostgreSQL port |
| PostgreSQL user | `MEMORY_POSTGRES_USER` | `postgres` | PostgreSQL user |
| PostgreSQL password | `MEMORY_POSTGRES_PASSWORD` | — | PostgreSQL password |
| PostgreSQL database | `MEMORY_POSTGRES_DATABASE` | `codespy` | PostgreSQL database |
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
| Git | `REVIEW_OUTPUT_GIT` | `true` | Post review to GitHub/GitLab (for `review-path`: open/update the report PR) |
| Cache dir | `REVIEW_CACHE_DIR` | `~/.cache/codespy` | Cache directory path |

## Concurrency

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `review.max_concurrent_scopes` | `REVIEW_MAX_CONCURRENT_SCOPES` | `8` | Scopes reviewed at once by each of code_review, doc, seo and supply_chain (so up to 4× this value run in total), and by the planner after the audit. Modular repos (Swift packages, Gradle modules) can yield many scopes per PR; this bounds provider throttling and MCP subprocesses. Action input: `max-concurrent-scopes` |

## Full-Path Review

Caps for `codespy review-path` (see [Usage](usage.md#full-path-review)). When either is exceeded, the run stops before any LLM call and prints the file and byte counts per top-level directory.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `review.full_path.max_files` | `REVIEW_FULL_PATH_MAX_FILES` | `300` | Most reviewable files under the path. CLI: `--max-files` |
| `review.full_path.max_bytes` | `REVIEW_FULL_PATH_MAX_BYTES` | `2000000` | Most total bytes of reviewable files (about 500k tokens) |

```yaml
review:
  full_path:
    max_files: 500
    max_bytes: 4000000
```

## Implementation Plan

The `plan` signature turns the review's findings into an ordered implementation plan. It is **off by default**.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `review.plan.enabled` | `REVIEW_PLAN_ENABLED` | `false` | Enable the planner. Action input: `plan-enabled` |
| `review.plan.model` | `REVIEW_PLAN_MODEL` | `bedrock/converse/global.anthropic.claude-opus-5-5` | Model. Action input: `plan-model` |
| `review.plan.max_iters` | `REVIEW_PLAN_MAX_ITERS` | `DEFAULT_MAX_ITERS` | RLM iterations per scope. Action input: `plan-max-iters` |
| `review.plan.max_llm_calls` | `REVIEW_PLAN_MAX_LLM_CALLS` | `DEFAULT_MAX_LLM_CALLS` | RLM LLM-call budget per scope. Action input: `plan-max-llm-calls` |

`reasoning_effort`, `temperature` and `max_tokens` work like they do for every other signature (Action inputs `plan-reasoning-effort`, `plan-temperature`, `plan-max-tokens`).

How it runs:

- After the audit, with the audit's quality assessment and recommendation as input. It adds one planner run to the time before the review is posted.
- One run per scope that has issues, concurrently (bounded by `review.max_concurrent_scopes`). Each issue goes to the scope with the longest matching subroot; issues outside every scope are planned against the repo root.
- Each run gets filesystem, ripgrep and tree-sitter tools rooted at its scope, to find the call sites and tests a fix touches.
- Every issue left after confidence filtering and rejected-feedback suppression is planned. With no issues the planner is skipped (no LLM call).
- Memory is read-only: the planner gets the run-level recall and the `recall_memory` tool, but writes no episode.
- Soft-fail: a failed scope is left out of the plan; a planner failure posts the review without a plan.

Each step lists its title, files, the change, how to verify it, and the issues it addresses. Where it appears:

| Output | Rendering |
|--------|-----------|
| PR/MR review body | Collapsible `🛠️ Implementation Plan` section after the recommendation, capped at 20,000 chars (trailing steps dropped, with a note) |
| `review-path` report PR description | Same section, same cap, with a link to the full plan in the report file |
| `review-path` report file, stdout markdown, MCP | Same collapsible section, uncapped |
| JSON | `implementation_plan` field: a list of `{subroot, steps: [{title, files, change, verification, addresses}]}` |

With several scopes, each scope's steps are nested in their own collapsible section. The planner's cost appears as `plan` in the Review cost table.

## SEO Review

The `seo` signature reviews changes to public websites for problems with crawling, indexing, ranking and AI citation. It covers search engines (Googlebot, Bingbot) and AI crawlers and assistants (GPTBot, OAI-SearchBot, ChatGPT-User, ClaudeBot, Claude-SearchBot, PerplexityBot, Google AI Overviews, Copilot). It is **off by default**. Turn it on only for repos that serve a public site.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `review.seo.enabled` | `REVIEW_SEO_ENABLED` | `false` | Enable the SEO reviewer. Action input: `seo-enabled` |
| `review.seo.scopes` | `REVIEW_SEO_SCOPES` | `[]` | `fnmatch` globs matched against the scope subroot (JSON array in env, for example `["apps/web", "sites/*"]`). Empty means any scope detected as a website. Action input: `seo-scopes` |
| `review.seo.model` | `REVIEW_SEO_MODEL` | `bedrock/converse/global.anthropic.claude-opus-5-5` | Model. Action input: `seo-model` |
| `review.seo.max_iters` | `REVIEW_SEO_MAX_ITERS` | `5` | RLM iterations per scope. Action input: `seo-max-iters` |
| `review.seo.max_llm_calls` | `REVIEW_SEO_MAX_LLM_CALLS` | `16` | RLM LLM-call budget per scope. Action input: `seo-max-llm-calls` |

`reasoning_effort`, `temperature` and `max_tokens` work like they do for every other signature (Action inputs `seo-reasoning-effort`, `seo-temperature`, `seo-max-tokens`).

### Which scopes are reviewed

A changed scope is reviewed only if it passes every check below, in this order. No LLM is called for scopes that are skipped, and each skip is logged with its reason.

1. **Whitelist.** If `scopes` is set, the scope subroot must match one of the globs.
2. **Website detection.** This check is skipped for whitelisted scopes. Otherwise the scope must contain at least one of:
   - a web framework in `package.json`: Next.js, Nuxt, SvelteKit, Astro, Gatsby, Remix, Angular SSR, React DOM, Vue, Svelte, Qwik or SolidStart;
   - a static site generator: Hugo, Jekyll or Eleventy;
   - server templates under `templates/`, `views/`, `app/views/` or `resources/views/`, such as ERB, Blade, Twig, Jinja, Liquid, Nunjucks, Handlebars, HEEx, Razor or Go templates;
   - plain `.html` files;
   - a changed crawl or hosting file.
3. **Web-relevant changes.** This check always applies. At least one changed file must be markup, a component, a template, a stylesheet, content, a crawl or hosting file, or framework config. Changes that touch only tests, CI, docs or scripts are skipped.

### What it checks

The reviewer checks the following for every changed page:

- **Rendering:** content, links or facts moved to client-only rendering, or removed from the DOM by UI state.
- **Navigation:** links implemented as click handlers instead of `<a href>`.
- **Metadata:** title, description, canonical, Open Graph and accidental `noindex`.
- **Indexation:** sitemap entries, missing 301/308 redirects, soft 404s, and content that differs by user agent, geo or cookie.
- **AI crawler access:** `robots.txt` and bot rules, plus `llms.txt`.
- **Structured data:** JSON-LD that is invalid or doesn't match the visible page.
- **Content:** hardcoded volatile facts (prices, rates) and removed YMYL disclaimers.
- **Semantics:** heading order, alt text, link text.
- **Core Web Vitals:** LCP, CLS and INP risks.
- **i18n:** reciprocal hreflang alternates.

It is static only: it reads the repository and never calls the live site. The canonical host comes from site config in the repo (Astro `site`, Hugo `baseURL`, Nuxt `site.url`, next-sitemap `siteUrl`, `.env.example`). Anything decided at runtime, such as CDN/WAF rules outside the repo, CMS content or real Core Web Vitals, is reported at most once, as an `info` issue, and only when a finding depends on it.

**Project rules** (canonical host, locales, retired terms, required disclosures, AI training-bot policy) go in the `AGENTS.md` / `CLAUDE.md` files the scope already inherits. They override the generic checks, except where following them would break search-engine guidelines (cloaking, hidden text, doorway pages, fake reviews).

### Site config outside the scope

Crawl and hosting config often lives outside the web scope, for example a root `vercel.json` or an `infra/nginx/` directory. Besides its scope-restricted filesystem, ripgrep and tree-sitter tools, the reviewer gets two read-only tools, `list_site_config` and `read_site_config`. They see only tracked files that match the site-config patterns: `robots.*`, `sitemap*`, `llms.txt`, `llms-full.txt`, `vercel.json`, `netlify.toml`, `_redirects`, `_headers`, `.htaccess`, nginx/apache `*.conf`, `firebase.json`, `wrangler.toml`, `middleware.*`, and Next.js, Nuxt, Astro, SvelteKit, Gatsby, Hugo, Jekyll and Eleventy config.

- The list is built once per run with `git ls-files`.
- It skips `review.excluded_directories` and is capped at 200 paths.
- Each read returns at most 60,000 characters.

Issues in these files use repo-root paths.

> **Known limitation:** remote reviews (`review`, `review-path`) use a sparse checkout that holds the reviewed scopes plus the files directly in their parent directories. Site config files elsewhere (for example `infra/nginx/site.conf` for a `web/` scope) are not added to it yet. `list_site_config` can show such a file while `read_site_config` reports it as unreadable. Local reviews (`review-local`, `review-uncommitted`) and `review-path` of the whole repo are not affected.

### Output

Findings are regular issues with category `seo`. They are counted in the **SEO** column of the statistics table and follow the same rejected-feedback suppression as other issues. Severity:

| Severity | Used for |
|----------|----------|
| `critical` | Blockers. Examples: indexable content missing from the initial HTML, accidental `noindex`, a removed URL with no redirect, robots rules blocking search bots, a wrong canonical or hreflang, soft 404s, wrong prices or facts, removed legal disclosures, invalid structured data |
| `high` | Major problems. Examples: a missing or duplicate title or description, missing structured data for a new page type, hardcoded volatile facts, non-reciprocal hreflang, a new route missing from the sitemap, metadata set only on the client |
| `low` | Tuning: alt text, title and description length, OG images, small performance wins |
| `info` | Questions the code alone cannot answer |

Titles start with the area, for example `Rendering: FAQ moved to client-only`. Decisions that belong to people, such as AI training-bot policy, YMYL disclaimers or pricing policy, are marked `Human review:` and come with no suggested fix.

## PR Conversation Context

These settings control the commit messages, PR comments and feedback on earlier reviews that are given to the agents (see [Usage](usage.md#pr-conversation-and-human-feedback)). Commits and comments are always on for remote reviews.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `review.pr_context.feedback` | `REVIEW_PR_CONTEXT_FEEDBACK` | `true` | Learn from human reactions and replies to codespy's findings. Set to `false` on open source projects where anyone can react: nothing is suppressed, no feedback reaches the prompt or memory, and `codespy learn` does nothing |
| `review.pr_context.trusted_authors` | `REVIEW_PR_CONTEXT_TRUSTED_AUTHORS` | `["github-actions[bot]"]` | Accounts trusted as codespy when verifying who posted a finding, besides the token's own account (JSON array in env). Lets `learn` run with a personal token on reviews that CI posted. Add your GitHub App bot here (e.g. `my-app[bot]`) |
| `review.pr_context.max_commits` | `REVIEW_PR_CONTEXT_MAX_COMMITS` | `50` | Most recent commits fetched |
| `review.pr_context.max_comments` | `REVIEW_PR_CONTEXT_MAX_COMMENTS` | `100` | Human/bot comments fetched, newest first. codespy's own comments, and the replies to them, come first |
| `review.pr_context.max_comment_chars` | `REVIEW_PR_CONTEXT_MAX_COMMENT_CHARS` | `1000` | Size limit for each comment, commit message or reply |
| `review.pr_context.max_block_chars` | `REVIEW_PR_CONTEXT_MAX_BLOCK_CHARS` | `12000` | Size limit for each rendered block. The oldest items are dropped first |

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
