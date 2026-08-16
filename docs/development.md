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
p poetry install           # Install all dependencies including dev
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
├── cli/                  # CLI commands and argument parsing
├── config/               # Configuration management
├── git/                  # GitHub/GitLab platform clients
├── llm/                  # LLM backend and DSPy integration
├── review/               # Review pipeline and signatures
│   ├── agents/           # ReAct and ChainOfThought agents
│   ├── signatures/       # DSPy signature definitions
│   └── tools/            # Agent tools (filesystem, git, web, etc.)
├── memory/               # Hippocampus memory system
├── output/               # Output formatters (markdown, json, git comments)
└── utils/                # Utility functions
```

---

[← Back to README](../README.md#documentation)
