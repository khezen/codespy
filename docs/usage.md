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

# Learn from human reactions to codespy's reviews of a PR (posts nothing; needs memory)
codespy learn https://github.com/owner/repo/pull/123
```

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

Scopes that don't look like a website, and diffs with no web-relevant changes (backend-only, tests, CI), are skipped without any LLM call. Put project-specific SEO rules (canonical host, locales, retired terms, required disclosures, AI training-bot policy) in `AGENTS.md`. See [Configuration → SEO Review](configuration.md#seo-review) for the gating rules, the checks, and the severity mapping.

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
