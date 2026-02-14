<p align="center">
  <img src="assets/codespy-logo.png" alt="CodeSpy logo">
</p>

<h1 align="center">Code<a href="https://github.com/khezen/codespy">Spy</a></h1>

<p align="center">
  An open-source AI reviewer that catches bugs, improves code quality, and integrates directly into your PR workflow, without sacrificing control or security.
</p>

<p align="center">
  <a href="https://github.com/khezen/codespy/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/khezen/codespy/ci.yml">
  </a>
  <a href="https://github.com/khezen/codespy/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/khezen/codespy">
  </a>
  <a href="https://github.com/khezen/codespy/stargazers">
    <img src="https://img.shields.io/github/stars/khezen/codespy">
  </a>
  <a href="https://github.com/khezen/codespy/issues">
    <img src="https://img.shields.io/github/issues/khezen/codespy">
  </a>
</p>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Why CodeSpy?](#why-codespy)
- [Features](#features)
- [Installation](#installation)
  - [Using pip](#using-pip)
  - [Using Homebrew (macOS/Linux)](#using-homebrew-macoslinux)
  - [Using Docker](#using-docker)
  - [Using Poetry (for development)](#using-poetry-for-development)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [IDE Integration (MCP Server)](#ide-integration-mcp-server)
  - [Using Docker](#using-docker-1)
  - [GitHub Action](#github-action)
- [Configuration](#configuration)
  - [Setup](#setup)
  - [Git Platform Tokens](#git-platform-tokens)
    - [GitHub Token](#github-token)
    - [GitLab Token](#gitlab-token)
  - [LLM Provider](#llm-provider)
  - [Advanced Configuration (YAML)](#advanced-configuration-yaml)
  - [Recommended Model Strategy](#recommended-model-strategy)
- [Output](#output)
  - [Markdown (default)](#markdown-default)
  - [GitHub/GitLab Review Comments](#githubgitlab-review-comments)
- [Architecture](#architecture)
- [DSPy Signatures](#dspy-signatures)
- [Supported Languages](#supported-languages)
- [Development](#development)
- [Contributors](#contributors)
- [License](#license)

---

## Why CodeSpy?

Most AI code reviewers are:
- ❌ Black boxes  
- ❌ SaaS-only  
- ❌ Opaque about reasoning  
- ❌ Risky for sensitive codebases  

**CodeSpy is different:**

- 🔍 Transparent reasoning  
- 🔐 Self-hostable  
- 🧠 Configurable review rules  
- 🔄 Native PR integration  
- 🧩 Extensible architecture  
- 📦 100% open-source  

Built for **engineering teams that care about correctness, security, and control.**

---


## Features

- 🔒 **Security Analysis** - Detects common vulnerabilities (injection, auth issues, data exposure, etc.) with CWE references
- 🐛 **Bug Detection** - Identifies logic errors, null references, resource leaks, edge cases
- 📝 **Documentation Review** - Checks for missing docstrings, outdated comments, incomplete docs
- 🔍 **Intelligent Scope Detection** - Automatically identifies code scopes (frontend, backend, infra, microservice in mono repo, etc...)
- 🔄 **Smart Deduplication** - LLM-powered issue deduplication across reviewers
- 💰 **Cost Tracking** - Track LLM calls, tokens, and costs per review
- 🤖 **Model Agnostic** - Works with OpenAI, AWS Bedrock, Anthropic, Ollama, and more via LiteLLM
- 🐳 **Docker Ready** - Run locally or in the cloud with Docker
- <img src="assets/GitHub_Invertocat_Black.svg" height="20" alt="GitHub"> <img src="assets/gitlab-logo-500-rgb.png" height="20" alt="GitLab"> **GitHub & GitLab**  - Works with both platforms, auto-detects from URL
- 🖥️ **Local Reviews** - Review local git changes without GitHub/GitLab — diff against any branch, ref, or review uncommitted work
- 🧩 **MCP Server** - IDE integration via Model Context Protocol — trigger reviews from AI coding assistants like Cline without leaving your editor
- 🔌 **GitHub Action** - One-line integration for automatic PR reviews



---

## Installation

### Using pip

```bash
pip install codespy-ai
```

### Using Homebrew (macOS/Linux)

```bash
brew tap khezen/codespy
brew install codespy
```

### Using Docker

```bash
# Pull the pre-built image from GitHub Container Registry
docker pull ghcr.io/khezen/codespy:latest

# Or build locally
docker build -t codespy .
```

### Using Poetry (for development)

```bash
# Clone the repository
git clone https://github.com/khezen/codespy.git
cd codespy

# Install dependencies
poetry install

# Or install only production dependencies
poetry install --only main
```

---

## Quick Start

Get up and running in 30 seconds:

```bash
# 1. Set your Git token (or let codespy auto-discover from gh/glab CLI)
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx  # For GitHub
# OR
export GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx  # For GitLab

# 2. Set your LLM provider (example with Anthropic)
export DEFAULT_MODEL=anthropic/claude-opus-4-6
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx

# 3. Review a PR or MR!
codespy review https://github.com/owner/repo/pull/123
# OR
codespy review https://gitlab.com/group/project/-/merge_requests/123
```

codespy auto-discovers credentials from standard locations (`~/.aws/credentials`, `gh auth token`, `glab auth token`, etc.) - see [Configuration](#configuration) for details.

---

## Usage

### Command Line

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

# Review uncommitted changes (staged + unstaged)
codespy review-uncommitted              # Review current dir
codespy review-uncommitted /path/to/repo
codespy review-uncommitted --output json
```

### IDE Integration (MCP Server)

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
- `review_pr(mr_url)` — Review a GitHub PR or GitLab MR by URL

Then ask your AI assistant: *"Review my local changes"* or *"Review uncommitted work in /path/to/repo"*

### Using Docker

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

### GitHub Action

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
        uses: khezen/codespy@v1
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
        uses: khezen/codespy@v1
        with:
          model: 'anthropic/claude-opus-4-6'
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

See [`.github/workflows/codespy-review.yml.example`](.github/workflows/codespy-review.yml.example) for more examples.

---

## Configuration

codespy supports two configuration methods:
- **`.env` file** - Simple environment variables for basic setup
- **`codespy.yaml`** - Full YAML configuration for advanced options (per-module settings)

Priority: cmd options > Environment Variables > YAML Config > Defaults

### Setup

```bash
# Copy the example file
cp .env.example .env
```

### Git Platform Tokens

codespy automatically detects the platform (GitHub or GitLab) from the URL and discovers tokens from multiple sources.

#### GitHub Token

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

#### GitLab Token

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

### LLM Provider

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

### Advanced Configuration (YAML)

For per-signature settings, use `codespy.yaml`. See [`codespy.yaml`](codespy.yaml) for all available options including:
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
export DEFECT_MODEL=anthropic/claude-sonnet-4-5-20250929

# Output settings
export OUTPUT_STDOUT=false
export OUTPUT_GIT=true
```

See `codespy.yaml` for full configuration options.

### Recommended Model Strategy

codespy uses a tiered model approach to balance review quality and cost:

| Tier | Role | Default | Recommended Model | Used By |
|------|------|---------|-------------------|---------|
| 🧠 **Smart** | Core analysis & reasoning | `DEFAULT_MODEL` | `anthropic/claude-opus-4-6` | Code & doc review, supply chain, scope identification |
| ⚡ **Mid-tier** | Extraction & deduplication | Falls back to `DEFAULT_MODEL` | `anthropic/claude-sonnet-4-5-20250929` | TwoStepAdapter field extraction, issue deduplication |
| 💰 **Cheap** | Summarization | Falls back to `DEFAULT_MODEL` | `anthropic/claude-haiku-4-5-20251001` | PR summary generation |

By default, **all models use `DEFAULT_MODEL`** (`anthropic/claude-opus-4-6`). This works out of the box — just set your API credentials and go.

To optimize costs, override the mid-tier and cheap models:

```bash
# .env or environment variables
DEFAULT_MODEL=anthropic/claude-opus-4-6                    # Smart tier (default)
EXTRACTION_MODEL=anthropic/claude-sonnet-4-5-20250929      # Mid-tier: field extraction
DEDUPLICATION_MODEL=anthropic/claude-sonnet-4-5-20250929   # Mid-tier: issue deduplication
SUMMARIZATION_MODEL=anthropic/claude-haiku-4-5-20251001    # Cheap tier: PR summary
```

Or in `codespy.yaml`:

```yaml
default_model: anthropic/claude-opus-4-6
extraction_model: anthropic/claude-sonnet-4-5-20250929
signatures:
  deduplication:
    model: anthropic/claude-sonnet-4-5-20250929
  summarization:
    model: anthropic/claude-haiku-4-5-20251001
```

---

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
export OUTPUT_GIT=true

# Or in codespy.yaml
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

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           codespy CLI                               │
├─────────────────────────────────────────────────────────────────────┤
│  review <pr_url> [--config ...] [--output json|md] [--model ...]    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                  Git Platform Integration                           │
│  - GitHub: Fetch PR diff, changed files, commit messages            │
│  - GitLab: Fetch MR diff, changed files, commit messages            │
│  - Auto-detects platform from URL                                   │
│  - Clone/access full repository for context                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                   DSPy Review Pipeline                              │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                   Scope Identifier                         │     │
│  │  (identifies code scopes: frontend, backend, infra, etc.)  │     │
│  └──────────────────────────┬─────────────────────────────────┘     │
│                             │                                       │
│  ┌──────────────────────────▼─────────────────────────────────┐     │
│  │              Parallel Review Modules                       │     │
│  │  ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │     │
│  │  │ Supply Chain │ │  Defect  │ │   Doc    │ │  Smell   │   │     │
│  │  │   Auditor    │ │ Detector │ │ Reviewer │ │ Detector │   │     │
│  │  └──────────────┘ └──────────┘ └──────────┘ └──────────┘   │     │
│  └──────────────────────────┬─────────────────────────────────┘     │
│                             │                                       │
│  ┌──────────────────────────▼─────────────────────────────────┐     │
│  │                 Issue Deduplicator                         │     │
│  │  (LLM-powered deduplication across reviewers)              │     │
│  └──────────────────────────┬─────────────────────────────────┘     │
│                             │                                       │
│  ┌──────────────────────────▼─────────────────────────────────┐     │
│  │                   PR Summarizer                            │     │
│  │  (generates summary, quality assessment, recommendation)   │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                     │
│                     Cost Tracker (tokens, calls, $)                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        Tools Layer                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐   │
│  │ Filesystem │  │    Git     │  │    Web     │  │  Cyber/OSV   │   │
│  │            │  │ (GH + GL)  │  │            │  │              │   │
│  └────────────┘  └────────────┘  └────────────┘  └──────────────┘   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                      Parsers                                   │ │
│  │  ┌─────────────────┐  ┌────────────────────────────────────┐   │ │
│  │  │     Ripgrep     │  │           Tree-sitter              │   │ │
│  │  │  (code search)  │  │  (multi-language AST parsing)      │   │ │
│  │  └─────────────────┘  └────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      LLM Backend (LiteLLM)                          │
│    Bedrock | OpenAI | Anthropic | Ollama | Any OpenAI-compatible    │
└─────────────────────────────────────────────────────────────────────┘
```

## DSPy Signatures

The review is powered by DSPy signatures that structure the LLM's analysis:

| Signature | Config Key | Description |
|-----------|------------|-------------|
| **ScopeIdentifierSignature** | `scope` | Identifies code scopes (frontend, backend, infra, microservice in mono repo, etc...) |
| **DefectDetectorSignature** | `defect` | Detects verified bugs, removed defensive code, and security vulnerabilities |
| **DocReviewSignature** | `doc` | Detects stale or wrong documentation caused by code changes |
| **SmellDetectorSignature** | `smell` | Detects code smells — naming, complexity, data clumps, YAGNI |
| **SupplyChainSecuritySignature** | `supply_chain` | Analyzes artifacts (Dockerfiles) and dependencies for supply chain security |
| **IssueDeduplicationSignature** | `deduplication` | LLM-powered deduplication of issues across reviewers |
| **MRSummarySignature** | `summarization` | Generates summary, quality assessment, and recommendation |

## Supported Languages

Tree-sitter based parsing for context-aware analysis:

| Language | Extensions | Features |
|----------|-----------|----------|
| Python | `.py` | Functions, classes, imports |
| JavaScript | `.js`, `.jsx` | Functions, classes, imports |
| TypeScript | `.ts`, `.tsx` | Functions, classes, interfaces |
| Go | `.go` | Functions, structs, interfaces |
| Java | `.java` | Methods, classes, packages |
| Kotlin | `.kt` | Functions, classes, objects |
| Swift | `.swift` | Functions, classes, structs |
| Objective-C | `.m`, `.h` | Methods, interfaces, protocols |
| Rust | `.rs` | Functions, structs, traits, impl blocks |
| Terraform | `.tf` | Resources, data sources, modules, variables |

All languages are supported for security, bug, and documentation analysis.

## Development

```bash
# Quick setup (creates .env and installs dependencies)
make setup

# Or manually with Poetry:
poetry install           # Install all dependencies including dev
poetry lock              # Update lock file

# Available make targets
make help

# Run commands with Poetry
make lint                # Run ruff linter
make format              # Format code with ruff
make typecheck           # Run mypy type checker
make test                # Run pytest tests
make build               # Build package with Poetry
make clean               # Clean build artifacts

# Or run directly:
poetry run codespy review https://github.com/owner/repo/pull/123
poetry run ruff check src/
poetry run mypy src/
```

---

## Contributors
* @khezen
* @pranavsriram8

---

## License

MIT