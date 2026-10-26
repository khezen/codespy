[← Back to README](../README.md#documentation)

# Usage Guide

## Command Line

```bash
# Review GitHub Pull Request
codespy review https://github.com/owner/repo/pull/123

# Review GitLab Merge Request
codespy review https://gitlab.com/group/project/-/merge_requests/123

# GitLab with nested groups
codespy review https://gitlab.com/group/subgroup/project/-/merge_requests/123

# Self-hosted GitLab
codespy review https://gitlab.mycompany.com/team/project/-/merge_requests/123

# Output as JSON
codespy review https://github.com/owner/repo/pull/123 --output json

# Use a specific model
codespy review https://github.com/owner/repo/pull/123 --model anthropic/claude-opus-4-6

# Use a custom config file
codespy review https://github.com/owner/repo/pull/123 --config path/to/config.yaml
codespy review https://github.com/owner/repo/pull/123 -f staging.yaml

# Disable stdout output (useful with --git-comment)
codespy review https://github.com/owner/repo/pull/123 --no-stdout

# Post review as GitHub/GitLab comment
codespy review https://github.com/owner/repo/pull/123 --git-comment

# Combine: only post to Git platform, no stdout
codespy review https://github.com/owner/repo/pull/123 --no-stdout --git-comment

# Show current configuration
codespy config

# Show configuration from a specific file
codespy config --config path/to/config.yaml

# Show version
codespy --version

# Review local git changes (no GitHub/GitLab needed)
codespy review-local                    # Review current dir vs main
codespy review-local /path/to/repo      # Review specific repo
codespy review-local --base develop     # Compare against develop
codespy review-local --base origin/main # Compare against origin/main
codespy review-local --base HEAD~5      # Compare against 5 commits back
codespy review-local --model anthropic/claude-sonnet-4-5-20250929
codespy review-local --output json --config staging.yaml

# Review uncommitted changes (staged + unstaged)
codespy review-uncommitted              # Review current dir
codespy review-uncommitted /path/to/repo
codespy review-uncommitted --output json
codespy review-uncommitted --model anthropic/claude-sonnet-4-5-20250929
codespy review-uncommitted --config staging.yaml

# Review a whole repo path (no PR needed); the report is delivered as a new PR/MR
codespy review-path https://github.com/owner/repo                       # whole repo
codespy review-path https://github.com/owner/repo --path services/api   # one path
codespy review-path https://gitlab.com/group/project --path src --ref develop
codespy review-path https://github.com/owner/repo --path lib --no-open-pr  # dry run: stdout only

# Learn from human reactions to codespy's reviews of a PR (posts nothing; needs memory)
codespy learn https://github.com/owner/repo/pull/123
```

## Full-Path Review

`codespy review-path` reviews existing code instead of a change. Use it to audit a service or module, take over an unfamiliar codebase, or check code that was never reviewed.

```bash
codespy review-path <repo-url> [--path P] [--ref BRANCH] [--max-files N] [--open-pr/--no-open-pr]
```

How it works:

1. `--ref` (default: the repo's default branch) is resolved to a commit through the API. It must be a branch, because it is also the base of the report PR. Tags and SHAs are rejected.
2. codespy sparse-clones `--path` (default `.`, the whole repo) at that commit. The files directly in each parent directory are also checked out (manifests, lock files, `AGENTS.md`), but nothing else.
3. There is no diff: every reviewable file under the path is presented as **newly added**, and the code is reviewed as-is. Binary, lock, minified and excluded-directory files are skipped, as in PR reviews, and so are symlinks, files larger than 512 KB, non-UTF-8 files and `.codespy/`. The skip counts are shown in the report.
4. All scopes under the path are identified (every manifest or module-marker package is its own scope), and the full pipeline runs: summary, code, doc, SEO and supply-chain review, audit, the implementation plan (opt-in), and memory.

### The report PR

There is no PR to comment on, so codespy opens one itself:

- **Branch** `codespy/review/<path-slug>` (`root` for the whole repo, `services-api` for `services/api`), based on the reviewed commit and targeting `--ref`.
- **File** `.codespy/reviews/<path-slug>.md` holds the full report. Every issue links to its file and lines at the reviewed commit.
- **Description** holds the same report. On very large reports, the lowest-severity issues are left out of the description (to stay under the platform's size limit), with a link to the file.

The PR is there to deliver the report. It can be closed without merging.

**Re-runs** (e.g. a weekly schedule) update the same PR. When an open report PR exists for that path and branch, codespy commits the new report file to it, replaces the description and posts one `Updated by codespy run …` comment, so earlier reports stay in the branch history. When the branch exists but its PR was closed or merged, the branch is reset to the reviewed commit and a new PR is opened. Don't push your own work to `codespy/review/*` branches.

The description is first posted before the memory phase, then rewritten with the final costs. `--no-open-pr` (or `review.output_git: false`) is a dry run: the report goes to stdout only and nothing is written to the repository.

### Token permissions

Opening the report PR needs write access:

| Platform | Token permissions |
|---|---|
| GitHub | `contents: write` and `pull-requests: write` (classic token: `repo`) |
| GitLab | `api` |

On GitHub Actions with the built-in `GITHUB_TOKEN`, also turn on **Settings → Actions → General → Allow GitHub Actions to create and approve pull requests**. For self-hosted GitLab, the repo URL host must match `GITLAB_URL`.

### Cost caps

A full-path review sends every file to the reviewers, so it can cost much more than a PR review. Both caps count only the files that will actually be reviewed, after every skip rule above. Untracked and `.gitignore`d files are never listed. Two caps stop the run **before any LLM call** and print the file and byte counts per top-level directory, so you can narrow `--path` or raise the cap:

| Setting | Env Var | Default |
|---|---|---|
| `review.full_path.max_files` (or `--max-files`) | `REVIEW_FULL_PATH_MAX_FILES` | `300` |
| `review.full_path.max_bytes` | `REVIEW_FULL_PATH_MAX_BYTES` | `2000000` (about 500k tokens) |

### Notes

- If your repo also runs the codespy Action on `pull_request`, the report PR is reviewed like any other PR. Skip it with `if: ${{ !startsWith(github.head_ref, 'codespy/review/') }}` on the review job.
- Feedback (👍/👎) on a report PR is not learned: the findings are in the description, not in review comments.
- Repository URLs must be plain repo URLs (`https://github.com/owner/repo`, `https://gitlab.example.com/group/sub/project`). PR, tree and SSH URLs are rejected.

## PR Conversation and Human Feedback

For remote reviews, codespy also reads the PR conversation and gives it to the agents:

- **Commit messages** go to summary, scope refinement, code review, doc review, SEO review and audit. Local `review-local` runs include them too.
- **Human PR comments** (general, review and inline), with their emoji reaction counts (👍2 👎1 …), go to summary, code review, doc review, SEO review and audit.
- **Human feedback on codespy's earlier comments on the same PR** goes to code review, doc review, SEO review, supply chain and audit. Each finding gets a verdict:
  - `rejected`: more 👎/😕 than 👍/❤️/🎉/🚀
  - `accepted`: more positive reactions than negative ones
  - `discussed`: a human replied
  - `resolved`: the GitLab thread was resolved with no reactions or replies

  A finding with the same fingerprint (file, category and title) as a `rejected` one is not reported again on that PR.

codespy never reads its own output back as human input. Every comment it posts carries a hidden `<!-- codespy:... -->` marker, and older reviews are still recognized by their header. Other bots' comments and reactions are ignored, and quoted codespy text in human replies is removed. For its own findings, codespy keeps only the identity (file, title, category, severity), never the description or suggestion.

With memory enabled, the feedback is retained as a `feedback` episode for each scope. Later reviews of the same scope learn what reviewers accept or reject from the Prefrontal briefings. Feedback that hasn't changed is not retained again.

A comment counts as one of codespy's findings only if it carries a codespy marker **and** the platform shows it was posted by the account codespy runs as (the token's login, or `github-actions[bot]` on GitHub Actions) or by an account listed in `review.pr_context.trusted_authors` (default `["github-actions[bot]"]`, so `codespy learn` with a personal token accepts reviews posted from CI). Markers can be copied, so a marker alone is never trusted as a finding. If that account can't be resolved (for example with a GitHub App token outside Actions), feedback is ignored. Feedback is also lost when earlier reviews were posted by a different account than the current run's. On GitHub Actions, `github-actions[bot]` is shared by every workflow in the repo, so anyone who can edit workflows can post as it.

Feedback left after the last review would otherwise be lost, so run `codespy learn` when the PR closes:

```yaml
# .github/workflows/codespy-learn.yml
name: CodeSpy Learn
on:
  pull_request:
    types: [closed]
jobs:
  learn:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: read
    steps:
      - uses: khezen/codespy@v2
        with:
          command: 'learn'   # posts nothing
          model: 'anthropic/claude-haiku-4-5-20251001'
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          memory-enabled: 'true'
          # plus the same memory storage settings as your review job
```

GitHub Action inputs for these settings: `pr-context-feedback` (default `true`), `pr-context-trusted-authors` (`["github-actions[bot]"]`), `pr-context-max-commits` (`50`), `pr-context-max-comments` (`100`), `pr-context-max-comment-chars` (`1000`), `pr-context-max-block-chars` (`12000`) and `command` (`review` or `learn`, default `review`).

Sizes are limited by `review.pr_context` (see [Configuration](configuration.md#pr-conversation-context)). Comment text is untrusted input: it is fenced and marked as informational only. On public repositories anyone can 👎 a finding, which suppresses it on that PR and feeds the scope's memory. Set `review.pr_context.feedback: false` (`REVIEW_PR_CONTEXT_FEEDBACK=false`) to turn feedback off there.

## Modular Repositories

Each module with its own manifest is reviewed as a separate scope: Swift packages (`Package.swift`), Gradle modules (`build.gradle(.kts)`), npm/Deno/Nx projects, Go modules and others. Files outside every module (for example an iOS app target next to `Example.xcodeproj`) go to the repository root scope. Scopes are reviewed in parallel, capped per module by the `max-concurrent-scopes` Action input (`review.max_concurrent_scopes`, default `8`; see [Configuration](configuration.md#concurrency)).

A repository with a single root manifest (a modular monolith) is split only where modules are declared explicitly. Directory names alone never create a scope there. Recognised module markers:

| Tool | Detected from | Scope |
|---|---|---|
| Packwerk | root `packwerk.yml` | each dir with `package.yml` |
| tach | root `tach.toml` | each `[[modules]] path` under `source_roots` |
| import-linter | `.importlinter`, `setup.cfg` or `pyproject.toml` `[tool.importlinter]` | contract `layers`/`modules`/`containers`/`source_modules` (at `.` or `src/`) |
| go-arch-lint | root `.go-arch-lint.yml` | `workdir` + literal prefix of each component `in` |
| Spring Modulith | `*Application.java`/`.kt` with `@SpringBootApplication` | each direct sub-package |
| JPMS | `module-info.java` | its module dir (without `src/main/java`) |
| Go | `cmd/<name>/main.go` with a single root `go.mod` | `cmd/<name>` |
| TypeScript | root `tsconfig.json` `references` | each referenced project |
| Laravel modules | `Modules/<name>/module.json` (or root `modules_statuses.json`) | the module dir |
| Deptrac | root `deptrac.yaml`/`deptrac.yml`/`depfile.yaml` | literal prefix of each `directory` collector |
| sbt | root `build.sbt` | `project.in(file("x"))` / `project in file("x")` |
| Mill | root `build.mill`/`build.sc`, `package.mill` | top-level `object x extends ...Module` → `x`; dirs with `package.mill` |

Marker scopes keep their dependencies on the enclosing manifest, so supply-chain review still runs on that manifest's scope. A marker dir that also has its own manifest is a manifest scope.

## Websites: SEO & AI-Visibility Review

For repositories that serve a public website, enable the opt-in SEO reviewer. It flags changes that hurt crawling, indexing, ranking, or citation by search engines and AI assistants, for example:

- content moved to client-only rendering;
- an accidental `noindex`;
- removed URLs with no redirect;
- `robots.txt` rules that block search or retrieval bots;
- structured data that doesn't match the visible page.

```bash
# Local
REVIEW_SEO_ENABLED=true codespy review-local

# Only the web app of a monorepo
REVIEW_SEO_ENABLED=true REVIEW_SEO_SCOPES='["apps/web"]' codespy review https://github.com/owner/repo/pull/123
```

```yaml
# GitHub Action
- uses: khezen/codespy@v2
  with:
    model: 'anthropic/claude-opus-4-6'
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    seo-enabled: 'true'
    seo-scopes: '["apps/web"]'   # optional; empty = any scope detected as a website
```

Scopes that don't look like a website, and diffs with no web-relevant changes (backend-only, tests, CI), are skipped without any LLM call. Before the model runs, a regex pre-scan flags likely hidden content (modals and accordions that start closed, selectors that render one dataset, hover-only menus). The model verifies those candidates and runs its own search on top. With `review-path`, the reviewer audits the whole site instead of a diff, and must cover every check area (rendering, metadata, indexation, AI crawlers, structured data, content, semantics, performance, i18n) before submitting, so it uses much more of `max_iters` / `max_llm_calls`: raise `REVIEW_SEO_MAX_ITERS` / `REVIEW_SEO_MAX_LLM_CALLS` on large sites. Put project-specific SEO rules (canonical host, locales, retired terms, required disclosures, AI training-bot policy) in `AGENTS.md`. See [Configuration → SEO Review](configuration.md#seo-review) for the gating rules, the checks, and the severity mapping.

## Implementation Plan

Enable the opt-in planner to get an ordered fix plan for the review's findings, in a collapsible section of the PR review (or of the `review-path` report PR and file):

```bash
REVIEW_PLAN_ENABLED=true codespy review https://github.com/owner/repo/pull/123
REVIEW_PLAN_ENABLED=true codespy review-path https://github.com/owner/repo --path services/api
```

```yaml
# GitHub Action
- uses: khezen/codespy@v2
  with:
    plan-enabled: 'true'
```

Steps are grouped per scope and ordered by dependency, then severity. Each one names the files, the change, how to verify it, and the issues it fixes. See [Configuration → Implementation Plan](configuration.md#implementation-plan).

## IDE Integration (MCP Server)

CodeSpy can run as an MCP (Model Context Protocol) server for integration with AI coding assistants like Cline, enabling code reviews directly from your editor without leaving your workflow.

```bash
# Start the MCP server
codespy serve

# Use a custom config file
codespy serve --config path/to/config.yaml
```

**Configure your IDE** (example for Cline in VS Code):

Add to `cline_mcp_settings.json`:
```json
{
  "mcpServers": {
    "codespy-reviewer": {
      "command": "codespy",
      "args": ["serve"],
      "env": {
        "DEFAULT_MODEL": "anthropic/claude-opus-4-6",
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

Or for AWS Bedrock:
```json
{
  "mcpServers": {
    "codespy-reviewer": {
      "command": "codespy",
      "args": ["serve"],
      "env": {
        "DEFAULT_MODEL": "bedrock/us.anthropic.claude-opus-4-6-v1",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "your-access-key",
        "AWS_SECRET_ACCESS_KEY": "your-secret-key"
      }
    }
  }
}
```

**Available MCP Tools:**
- `review_local_changes(repo_path, base_ref)` — Review branch changes vs base (e.g., vs `main`)
- `review_uncommitted(repo_path)` — Review staged + unstaged working tree changes
- `review_pr(pr_url)` — Review a GitHub PR or GitLab MR by URL

Then ask your AI assistant: *"Review my local changes"* or *"Review uncommitted work in /path/to/repo"*

## Docker

```bash
# With docker run (using GHCR image)
docker run --rm \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  -e DEFAULT_MODEL=anthropic/claude-opus-4-6 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  ghcr.io/khezen/codespy:latest review https://github.com/owner/repo/pull/123

# Or use a specific version
docker run --rm \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  -e DEFAULT_MODEL=anthropic/claude-opus-4-6 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  ghcr.io/khezen/codespy:0.2.1 review https://github.com/owner/repo/pull/123
```

## GitHub Action

Add CodeSpy to your repository for automatic PR reviews:

**Trigger on `/codespy review` comment:**

```yaml
# .github/workflows/codespy-review.yml
name: CodeSpy Code Review

on:
  issue_comment:
    types: [created]

jobs:
  review:
    # Only run on PR comments containing '/codespy review'
    if: |
      github.event.issue.pull_request &&
      contains(github.event.comment.body, '/codespy review')
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write

    steps:
      - name: Run CodeSpy Review
        uses: khezen/codespy@v2
        with:
          model: 'anthropic/claude-opus-4-6'
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

**Trigger automatically on every PR:**

```yaml
# .github/workflows/codespy-review.yml
name: CodeSpy Code Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write

    steps:
      - name: Run CodeSpy Review
        uses: khezen/codespy@v2
        with:
          model: 'anthropic/claude-opus-4-6'
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

See [`../.github/workflows/codespy-review.yml.example`](../.github/workflows/codespy-review.yml.example) for more examples.

## Output

### Markdown (default)

```markdown
# Code Review: Add user authentication

**PR:** [owner/repo#123](https://github.com/owner/repo/pull/123)
**Reviewed at:** 2024-01-15 10:30 UTC
**Model:** anthropic/claude-opus-4-6

## Summary

This PR implements user authentication with JWT tokens...

## Statistics

- **Total Issues:** 3
- **Critical:** 1
- **Security:** 1
- **Bugs:** 1
- **Documentation:** 1
- **Smells:** 0
- **SEO:** 0

## Issues

### 🔴 Critical (1)

#### SQL Injection Vulnerability

**Location:** `src/auth/login.py:45`
**Category:** security

The user input is directly interpolated into the SQL query...

**Code:**
query = f"SELECT * FROM users WHERE username = '{username}'"

**Suggestion:**
Use parameterized queries instead...

**Reference:** [CWE-89](https://cwe.mitre.org/data/definitions/89.html)
```

### GitHub/GitLab Review Comments

CodeSpy can post reviews directly to GitHub PRs or GitLab MRs as native review comments with inline annotations.

**Enable via CLI:**
```bash
# GitHub
codespy review https://github.com/owner/repo/pull/123 --git-comment

# GitLab
codespy review https://gitlab.com/group/project/-/merge_requests/123 --git-comment

# Combine: only post to platform, no stdout
codespy review https://github.com/owner/repo/pull/123 --no-stdout --git-comment
```

**Enable via configuration:**
```bash
# Environment variable
export REVIEW_OUTPUT_GIT=true

# Or in codespy.yaml
review:
  output_git: true
```

**Features:**

- 🎯 **Inline Comments** - Issues are posted as review comments on the exact lines where they occur
- 📏 **Multi-line Support** - Issues spanning multiple lines are annotated with start/end line ranges
- 🔴🟠🟡🔵 **Severity Indicators** - Visual emoji markers for Critical, High, Medium, Low severity
- 📦 **Collapsible Sections** - Organized review body with expandable details:
  - 📋 Summary of changes
  - 🎯 Quality Assessment
  - 📊 Statistics table
  - 💰 Cost breakdown per signature
  - 💡 Recommendation
- 🔗 **CWE References** - Security issues link directly to MITRE CWE database

---

See [Configuration](configuration.md) for all available settings.

---

[← Back to README](../README.md#documentation)
