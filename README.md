# codespy

**Code review agent powered by DSPy** - Automatically review GitHub pull requests for security vulnerabilities, bugs, and documentation issues.

## Features

- 🔒 **Security Analysis** - Detects common vulnerabilities (injection, auth issues, data exposure, etc.)
- 🐛 **Bug Detection** - Identifies logic errors, null references, resource leaks, edge cases
- 📝 **Documentation Review** - Checks for missing docstrings, outdated comments, incomplete docs
- 🔍 **Codebase Context** - Analyzes changes in context of the broader codebase (imports, dependencies)
- 🤖 **Model Agnostic** - Works with OpenAI, AWS Bedrock, Anthropic, Ollama, and more via LiteLLM
- 🐳 **Docker Ready** - Run locally or in the cloud with Docker

## Installation

### Using pip (recommended)

```bash
# Clone the repository
git clone https://github.com/khezen/codespy.git
cd codespy

# Install in development mode
pip install -e .
```

### Using Docker

```bash
# Build the image
docker build -t codespy .

# Or use docker compose
docker compose build
```

## Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

### Required Settings

1. **GitHub Token** - Create a token at https://github.com/settings/tokens with `repo` scope
   ```bash
   GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
   ```

2. **LLM Provider** - Choose one:

   **OpenAI:**
   ```bash
   LITELLM_MODEL=gpt-4o
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
   ```

   **AWS Bedrock:**
   ```bash
   LITELLM_MODEL=bedrock/anthropic.claude-3-sonnet-20240229-v1:0
   AWS_REGION=us-east-1
   # Uses ~/.aws/credentials by default, or set explicitly:
   # AWS_ACCESS_KEY_ID=...
   # AWS_SECRET_ACCESS_KEY=...
   ```

   **Anthropic (direct):**
   ```bash
   LITELLM_MODEL=claude-3-opus-20240229
   ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
   ```

   **Local Ollama:**
   ```bash
   LITELLM_MODEL=ollama/llama3
   ```

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
```python
query = f"SELECT * FROM users WHERE username = '{username}'"
```

**Suggestion:**
Use parameterized queries instead...

**Reference:** [CWE-89](https://cwe.mitre.org/data/definitions/89.html)

---
```

### JSON

```json
{
  "pr_number": 123,
  "pr_title": "Add user authentication",
  "pr_url": "https://github.com/owner/repo/pull/123",
  "repo": "owner/repo",
  "reviewed_at": "2024-01-15T10:30:00Z",
  "model_used": "gpt-4o",
  "file_reviews": [...],
  "overall_summary": "...",
  "recommendation": "REQUEST_CHANGES: Found 1 critical issues..."
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         codespy CLI                              │
├─────────────────────────────────────────────────────────────────┤
│  review <pr_url> [--with-context] [--output json|markdown]      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    GitHub Integration                            │
│  - Fetch PR diff, changed files, commit messages                │
│  - Clone/access full repository for context                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    DSPy Review Pipeline                          │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐        │
│  │  Security   │  │    Bug      │  │  Documentation   │        │
│  │  Analyzer   │  │  Detector   │  │    Reviewer      │        │
│  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘        │
│         └────────────────┼──────────────────┘                   │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │  Contextual Analyzer  │                          │
│              │  (codebase awareness) │                          │
│              └───────────┬───────────┘                          │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │   Review Aggregator   │                          │
│              └───────────────────────┘                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    LLM Backend (LiteLLM)                         │
│  Bedrock | OpenAI | Anthropic | Ollama | Any OpenAI-compatible  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
codespy/
├── src/codespy/
│   ├── __init__.py
│   ├── cli.py                 # CLI entry point
│   ├── config.py              # Settings management
│   ├── github/
│   │   ├── client.py          # GitHub API client
│   │   └── models.py          # PR data models
│   └── review/
│       ├── models.py          # Review result models
│       ├── pipeline.py        # Main review orchestration
│       ├── signatures.py      # DSPy signatures
│       └── modules/
│           ├── base.py        # Base review module
│           ├── security.py    # Security analyzer
│           ├── bugs.py        # Bug detector
│           ├── docs.py        # Documentation reviewer
│           └── context.py     # Codebase context analyzer
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## DSPy Signatures

The review is powered by DSPy signatures that structure the LLM's analysis:

- **SecurityAnalysis** - Analyzes code for security vulnerabilities with CWE references
- **BugDetection** - Detects logic errors, resource leaks, and edge cases
- **DocumentationReview** - Checks documentation completeness
- **ContextualAnalysis** - Validates changes against codebase patterns
- **PRSummary** - Generates overall summary and recommendation

## Supported Languages

Context-aware analysis (import resolution) is supported for:

- Python (`.py`)
- JavaScript/TypeScript (`.js`, `.ts`, `.jsx`, `.tsx`)
- Go (`.go`)

All languages are supported for security, bug, and documentation analysis.

## Development

```bash
# Quick setup (creates .env and installs dependencies)
make setup

# Or manually:
pip install -e ".[dev]"

# Available make targets
make help

# Run linter
make lint

# Format code
make format

# Run type checker
make typecheck

# Run tests
make test

# Build package
make build

# Clean build artifacts
make clean
```

## License

MIT