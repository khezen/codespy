<p align="center">
  <img src="assets/codespy-logo.png" alt="CodeSpy logo">
</p>

<h1 align="center">Code<a href="https://github.com/khezen/codespy">Spy</a></h1>

<p align="center">
  <strong>Automated code reviews for teams who care about correctness.</strong>
</p>
<p align="center">
  An open-source AI reviewer that catches bugs, improves code quality, and integrates directly into your PR workflow, without sacrificing control or security.
</p>
<p align="center">
  <i>"Fast feedback. No black box. No vendor lock-in."</i>
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
- 🔌 **GitHub & GitLab** - Works with both platforms, auto-detects from URL
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
export DEFAULT_MODEL=claude-sonnet-4-5-20250929
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
codespy review https://github.com/owner/repo/pull/123 --model claude-sonnet-4-5-20250929

# Skip codebase context analysis
codespy review https://github.com/owner/repo/pull/123 --no-with-context

# Disable stdout output (useful with --git-comment)
codespy review https://github.com/owner/repo/pull/123 --no-stdout

# Post review as GitHub/GitLab comment
codespy review https://github.com/owner/repo/pull/123 --git-comment

# Combine: only post to Git platform, no stdout
codespy review https://github.com/owner/repo/pull/123 --no-stdout --git-comment

# Show current configuration
codespy config

# Show version
codespy --version
```

### Using Docker

```bash
# With docker run (using GHCR image)
docker run --rm \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  -e DEFAULT_MODEL=claude-sonnet-4-5-20250929 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  ghcr.io/khezen/codespy:latest review https://github.com/owner/repo/pull/123

# Or use a specific version
docker run --rm \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  -e DEFAULT_MODEL=claude-sonnet-4-5-20250929 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  ghcr.io/khezen/codespy:0.1.0 review https://github.com/owner/repo/pull/123
```

### GitHub Action

Add CodeSpy to your repository for automatic PR reviews:

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
          model: 'claude-sonnet-4-5-20250929'
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

**Available Providers:**

```yaml
# OpenAI
- uses: khezen/codespy@v1
  with:
    model: 'gpt-5'
    openai-api-key: ${{ secrets.OPENAI_API_KEY }}

# AWS Bedrock
- uses: khezen/codespy@v1
  with:
    model: 'bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0'
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: 'us-east-1'

# Google Gemini
- uses: khezen/codespy@v1
  with:
    model: 'gemini/gemini-2.5-pro'
    gemini-api-key: ${{ secrets.GEMINI_API_KEY }}
```

See [`.github/workflows/codespy-review.yml.example`](.github/workflows/codespy-review.yml.example) for more examples.

---

## Configuration

codespy supports two configuration methods:
- **`.env` file** - Simple environment variables for basic setup
- **`codespy.yaml`** - Full YAML configuration for advanced options (per-module settings)

Priority: Environment Variables > YAML Config > Defaults

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
DEFAULT_MODEL=claude-sonnet-4-5-20250929
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
DEFAULT_MODEL=gpt-5
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

For per-signature settings, use `codespy.yaml`:

```yaml
# codespy.yaml

# LLM provider settings (credentials are auto-discovered by default)
llm:
  auto_discover_openai: true       # Discover from ~/.config/openai/, ~/.openai/, $OPENAI_API_KEY
  auto_discover_anthropic: true    # Discover from ~/.config/anthropic/, ~/.anthropic/, $ANTHROPIC_API_KEY
  auto_discover_gemini: true       # Discover from $GEMINI_API_KEY, gcloud ADC
  auto_discover_aws: true          # Discover from ~/.aws/credentials, AWS CLI
  enable_prompt_caching: true      # Provider-side prompt caching (reduces latency and costs)

# GitHub settings (token is auto-discovered by default)
github:
  auto_discover_token: true        # Discover from gh CLI, git credentials, ~/.netrc

# GitLab settings (token is auto-discovered by default)
gitlab:
  auto_discover_token: true        # Discover from glab CLI, git credentials, ~/.netrc
  # url: https://gitlab.mycompany.com  # For self-hosted GitLab (default: gitlab.com)

# Default settings for all signatures
default_model: claude-sonnet-4-5-20250929  # Also settable via DEFAULT_MODEL env var
extraction_model: claude-haiku-4-5-20251001  # For field extraction (smaller model)
default_max_iters: 3
default_max_context_size: 50000
default_max_reasoning_tokens: 8000  # Limit reasoning verbosity
default_temperature: 0.1            # Lower = more deterministic output

# Global LLM reliability settings
llm_retries: 3                       # Number of retries for LLM API calls
llm_timeout: 120                     # Timeout in seconds

# Per-signature overrides (see signatures table below for all available)
signatures:
  code_security:
    enabled: true
    model: claude-sonnet-4-5-20250929

  supply_chain:
    enabled: true

  bug_detection:
    enabled: true

  doc_review:
    enabled: true
    model: claude-haiku-4-5-20251001  # Smaller model for simpler task

  domain_analysis:
    enabled: false                    # Disabled by default (expensive)
    max_iters: 6

  scope_identification:
    enabled: true
    max_iters: 10
    model: claude-opus-4-5-20251101   # Larger model for complex scope analysis

  deduplication:
    enabled: true
    model: claude-haiku-4-5-20251001  # Smaller model for simple task

  summarization:
    enabled: true
    model: claude-haiku-4-5-20251001

# Output destinations
output_format: markdown              # markdown or json
output_stdout: true                  # Print to stdout
output_git: true                    # Post as GitHub PR / GitLab MR review comment

# Directories to skip during review
excluded_directories:
  - vendor
  - node_modules
  - dist
  - build
  - __pycache__
```

Override YAML settings via environment variables using `_` separator:

```bash
# Default settings
export DEFAULT_MODEL=claude-sonnet-4-5-20250929
export DEFAULT_MAX_ITERS=20

# Per-signature settings (use signature name, not module name)
export DOMAIN_ANALYSIS_MAX_ITERS=20
export DOC_REVIEW_ENABLED=false
export CODE_SECURITY_MODEL=gpt-5

# Output settings
export OUTPUT_STDOUT=false
export OUTPUT_GIT=true
```

See `codespy.yaml` for full configuration options.

---

## Output

### Markdown (default)

```markdown
# Code Review: Add user authentication

**PR:** [owner/repo#123](https://github.com/owner/repo/pull/123)
**Reviewed at:** 2024-01-15 10:30 UTC
**Model:** claude-sonnet-4-5-20250929

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
│  review <pr_url> [--with-context] [--output json|md] [--model ...]  │
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
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │     │
│  │  │  Security   │  │    Bug      │  │  Documentation   │    │     │
│  │  │   Auditor   │  │  Detector   │  │    Reviewer      │    │     │
│  │  └─────────────┘  └─────────────┘  └──────────────────┘    │     │
│  │                                                            │     │
│  │              ┌───────────────────────┐                     │     │
│  │              │     Domain Expert     │                     │     │
│  │              │  (codebase awareness) │                     │     │
│  │              └───────────────────────┘                     │     │
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
| **ScopeIdentifierSignature** | `scope_identification` | Identifies code scopes (frontend, backend, infra, microservice in mono repo, etc...) |
| **CodeSecuritySignature** | `code_security` | Analyzes code changes for verified security vulnerabilities with CWE references |
| **SupplyChainSecuritySignature** | `supply_chain` | Analyzes artifacts (Dockerfiles) and dependencies for supply chain security |
| **BugDetectionSignature** | `bug_detection` | Detects verified bugs, logic errors, and resource leaks |
| **DocumentationReviewSignature** | `doc_review` | Reviews documentation for accuracy based on code changes |
| **DomainExpertSignature** (experimental, disabled by default)| `domain_analysis` | Analyzes business logic, architecture, patterns, and style consistency |
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

## License

MIT