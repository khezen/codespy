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
├── cli_local.py              # Local review commands
├── cli_remote.py             # Remote PR/MR review commands
├── cli_mcp_server.py         # MCP server command
├── config.py                 # Main configuration
├── config_dspy.py            # DSPy configuration
├── config_git.py             # Git platform configuration
├── config_io.py              # I/O configuration
├── config_llm.py             # LLM provider configuration
├── config_memory.py          # Memory system configuration
├── agents/                   # DSPy agents and pipeline
│   ├── context_safe.py       # Context window overflow resilience (RLM fallback)
│   ├── cost_tracker.py       # Token/cost tracking
│   ├── dspy_config.py        # DSPy runtime config
│   ├── memory/               # Hippocampus memory system
│   │   └── hippocampus/      # Episode persistence, context memory, budget
│   └── reviewer/             # Review pipeline
│       ├── models.py         # Review data models
│       ├── reviewer.py       # Main review orchestrator
│       ├── server.py         # MCP server implementation
│       ├── modules/          # Pipeline stages (scope_resolver, summarizer, code_reviewer, doc_reviewer, supply_chain_auditor, auditor)
│       └── reporters/        # Output reporters (git comments, stdout)
└── tools/                    # Agent tools
    ├── git/                  # GitHub/GitLab clients, local diff, patch utils
    ├── cyber/osv/            # OSV vulnerability scanning
    ├── parsers/              # Ripgrep + Tree-sitter
    ├── storage/              # Filesystem + S3
    ├── web/                  # Web search client
    └── mcp_utils.py          # MCP tool utilities
```

---

[← Back to README](../README.md#documentation)
