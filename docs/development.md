[← Back to README](../README.md#documentation)

# Development

## Setup

```bash
# Clone the repository
git clone https://github.com/khezen/codespy.git
cd codespy

# Quick setup (creates .env and installs dependencies)
make setup

# Or manually with Poetry:
poetry install           # Install all dependencies including dev
poetry lock              # Update lock file
```

## Make Targets

```bash
make help                # Show all available targets
make setup               # Create .env and install dependencies
make lint                # Run ruff linter
make format              # Format code with ruff
make typecheck           # Run mypy type checker
make test                # Run pytest tests
make build               # Build package with Poetry
make clean               # Clean build artifacts
```

## Running Directly

```bash
# Run codespy via Poetry
poetry run codespy review https://github.com/owner/repo/pull/123

# Run linters directly
poetry run ruff check src/
poetry run mypy src/
```

## Project Structure

```
src/codespy/
├── __init__.py
├── cli.py                    # Main CLI entrypoint
├── cli_local.py              # Local review commands (review-local, review-uncommitted)
├── cli_remote.py             # Remote PR/MR review command (review)
├── cli_path.py               # Full-path review command (review-path)
├── cli_mcp_server.py         # MCP server command
├── config.py                 # Main configuration
├── config_dspy.py            # Per-signature (agent) configuration
├── config_git.py             # Git platform configuration
├── config_io.py              # I/O configuration
├── config_llm.py             # LLM provider configuration
├── config_memory.py          # Memory system configuration
├── config_utils.py           # Lightweight config helpers (env var mapping)
├── agents/                   # DSPy agents
│   ├── context_safe.py       # Context window overflow resilience (RLM fallback)
│   ├── cost_tracker.py       # Token/cost tracking
│   ├── diagnostic_adapter.py # Logs details of empty LM responses
│   ├── dspy_config.py        # DSPy runtime config
│   ├── rlm_tools.py          # RLM agent builder and tool bridging
│   ├── memory/               # Memory system
│   │   ├── cerebral/         # Hindsight semantic memory
│   │   ├── hippocampus/      # Episode persistence, context memory, budget
│   │   ├── prefrontal/       # Pre-call recall and the recall_memory tool
│   │   ├── recall.py         # Records what Prefrontal gave each agent
│   │   ├── pg0_manager.py    # pg0-embedded PostgreSQL lifecycle
│   │   └── postgres.py       # PostgreSQL episode persistence
│   └── review/               # Review agents, one package per stage
│       ├── scope/            # Scope resolution
│       ├── summary/          # PR summary
│       ├── code_review/      # Code review
│       ├── doc/              # Documentation review
│       ├── seo/              # SEO & AI-visibility review (opt-in)
│       ├── supply_chain/     # Supply chain audit
│       ├── audit/            # Quality assessment + recommendation
│       ├── plan/             # Implementation planner (opt-in)
│       ├── implement/        # Implementation agent: carries out the plan (opt-in)
│       ├── feedback/         # Learning from human feedback on past reviews
│       ├── helpers.py        # Shared helpers
│       └── models.py         # Issue and review context models
├── workflows/
│   └── review/               # Review pipeline
│       ├── pipeline.py       # Orchestrates the stages above
│       ├── models.py         # Review result models and rendering
│       ├── server.py         # MCP server implementation
│       └── reporters/        # Output: review comments (git), report PR (pull_request),
│                             #   fix PR (fix_pull_request), stdout
└── tools/                    # Agent tools
    ├── git/                  # GitHub/GitLab clients, local diff, full-path review, patch utils
    ├── cyber/osv/            # OSV vulnerability scanning
    ├── parsers/              # Ripgrep + Tree-sitter
    ├── storage/              # Filesystem (read-only, or writable for the implementation agent) + S3
    ├── web/                  # Web search client
    └── mcp_utils.py          # MCP tool utilities
```

---

[← Back to README](../README.md#documentation)
