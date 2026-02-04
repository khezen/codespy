# codespy

**Code review agent written with DSPy** - Automatically review GitHub pull requests for security vulnerabilities, bugs, and documentation issues.

## Features

- 🔒 **Security Analysis** - Detects common vulnerabilities (injection, auth issues, data exposure, etc.) with CWE references
- 🐛 **Bug Detection** - Identifies logic errors, null references, resource leaks, edge cases
- 📝 **Documentation Review** - Checks for missing docstrings, outdated comments, incomplete docs
- 🧠 **Domain Expert Analysis** - Analyzes changes in context of the broader codebase
- 🔍 **Intelligent Scope Detection** - Automatically identifies code scopes (frontend, backend, infra, microservice in mono repo, etc.)
- 🔄 **Smart Deduplication** - LLM-powered issue deduplication across reviewers
- 💰 **Cost Tracking** - Track LLM calls, tokens, and costs per review
- 🤖 **Model Agnostic** - Works with OpenAI, AWS Bedrock, Anthropic, Ollama, and more via LiteLLM
- 🐳 **Docker Ready** - Run locally or in the cloud with Docker

## Installation

### Using Poetry (recommended)

```bash
# Clone the repository
git clone https://github.com/khezen/codespy.git
cd codespy

# Install dependencies
poetry install

# Or install only production dependencies
poetry install --only main
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/khezen/codespy.git
cd codespy

# Install from pyproject.toml
pip install .
```

### Using Docker

```bash
# Build the image
docker build -t codespy .
```

## Configuration

codespy supports two configuration methods:
- **`.env` file** - Simple environment variables for basic setup
- **`codespy.yaml`** - Full YAML configuration for advanced options (per-module settings)

Priority: Environment Variables > YAML Config > Defaults

### Quick Start

```bash
# Copy the example files
cp .env.example .env
cp codespy.example.yaml codespy.yaml  # Optional, for advanced config
```

### Required Settings

1. **GitHub Token** - codespy automatically discovers your GitHub token from multiple sources:
   - `GITHUB_TOKEN` or `GH_TOKEN` environment variables
   - GitHub CLI (`gh auth token`)
   - Git credential helper
   - `~/.netrc` file
   
   Or create a token at https://github.com/settings/tokens with `repo` scope:
   ```bash
   GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
   ```

2. **LLM Provider** - Choose one:

   **OpenAI:**
   ```bash
   DEFAULT_MODEL=gpt-4o
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
   ```

   **AWS Bedrock:**
   ```bash
   DEFAULT_MODEL=bedrock/anthropic.claude-3-sonnet-20240229-v1:0
   AWS_REGION=us-east-1
   # Uses ~/.aws/credentials by default, or set explicitly:
   # AWS_ACCESS_KEY_ID=...
   # AWS_SECRET_ACCESS_KEY=...
   ```

   **Anthropic (direct):**
   ```bash
   DEFAULT_MODEL=claude-3-opus-20240229
   ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
   ```

   **Google Gemini:**
   ```bash
   DEFAULT_MODEL=gemini/gemini-1.5-pro
   GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxx
   ```

   **Local Ollama:**
   ```bash
   DEFAULT_MODEL=ollama/llama3
   ```

### Advanced Configuration (YAML)

For per-module settings, use `codespy.yaml`:

```yaml
# codespy.yaml

# Default settings for all modules
default_model: gpt-4o             # Also settable via DEFAULT_MODEL env var
default_max_iters: 10
default_max_context_size: 50000

# Per-module overrides
modules:
  security_auditor:
    enabled: true
    max_iters: 10

  doc_reviewer:
    enabled: true
    max_iters: 15

  domain_expert:
    enabled: true
    max_iters: 30               # More iterations for deep exploration

  deduplicator:
    model: gpt-3.5-turbo        # Cheaper model for simple task

output_format: markdown
```

Override YAML settings via environment variables using `__` separator:

```bash
# Default settings
export DEFAULT_MODEL=gpt-4o
export DEFAULT_MAX_ITERS=20

# Per-module settings
export DOMAIN_EXPERT__MAX_ITERS=20
export DOC_REVIEWER__ENABLED=false
```

See `codespy.example.yaml` for full configuration options.

## Usage

### Command Line

```bash
# Basic review
codespy review https://github.com/owner/repo/pull/123

# Output as JSON
codespy review https://github.com/owner/repo/pull/123 --output json

# Use a specific model
codespy review https://github.com/owner/repo/pull/123 --model bedrock/anthropic.claude-3-sonnet-20240229-v1:0

# Skip codebase context analysis
codespy review https://github.com/owner/repo/pull/123 --no-with-context

# Include vendor/dependency files in review
codespy review https://github.com/owner/repo/pull/123 --include-vendor

# Show current configuration
codespy config

# Show version
codespy --version
```

### Using Docker

```bash
# With docker compose
docker compose run codespy review https://github.com/owner/repo/pull/123

# With docker run
docker run --rm \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  -e LITELLM_MODEL=gpt-4o \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  codespy review https://github.com/owner/repo/pull/123
```

## Output

### Markdown (default)

```markdown
# Code Review: Add user authentication

**PR:** [owner/repo#123](https://github.com/owner/repo/pull/123)
**Reviewed at:** 2024-01-15 10:30 UTC
**Model:** gpt-4o

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

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           codespy CLI                               │
├─────────────────────────────────────────────────────────────────────┤
│  review <pr_url> [--with-context] [--output json|md] [--model ...]  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      GitHub Integration                             │
│  - Fetch PR diff, changed files, commit messages                    │
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
│  │ Filesystem │  │   GitHub   │  │    Web     │  │  Cyber/OSV   │   │
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

## DSPy Modules

The review is powered by DSPy modules that structure the LLM's analysis:

| Module | Description |
|--------|-------------|
| **ScopeIdentifier** | Identifies code scopes (frontend, backend, infrastructure, etc.) |
| **SecurityAuditor** | Analyzes code for security vulnerabilities with CWE references |
| **BugDetector** | Detects logic errors, resource leaks, and edge cases |
| **DocumentationReviewer** | Checks documentation completeness and accuracy |
| **DomainExpert** | Validates changes against codebase patterns and conventions |
| **IssueDeduplicator** | LLM-powered deduplication of issues across reviewers |
| **PRSummary** | Generates overall summary, quality assessment, and recommendation |

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

## License

MIT