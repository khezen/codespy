.PHONY: help install install-dev lint format typecheck test clean build docker-build docker-run review

# Default target
help:
	@echo "codespy - Code review agent powered by DSPy"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  install       Install package in production mode"
	@echo "  install-dev   Install package with dev dependencies"
	@echo "  lock          Generate/update poetry.lock"
	@echo "  lint          Run ruff linter"
	@echo "  format        Format code with ruff"
	@echo "  typecheck     Run mypy type checker"
	@echo "  test          Run pytest tests"
	@echo "  clean         Remove build artifacts and caches"
	@echo ""
	@echo "Build:"
	@echo "  build         Build Python package"
	@echo "  docker-build  Build Docker image"
	@echo ""
	@echo "Run:"
	@echo "  docker-run    Run codespy in Docker (use PR_URL=...)"
	@echo "  review        Run review on a PR (use PR_URL=...)"
	@echo "  config        Show current configuration"
	@echo ""
	@echo "Examples:"
	@echo "  make install-dev"
	@echo "  make lint"
	@echo "  make review PR_URL=https://github.com/owner/repo/pull/123"
	@echo "  make docker-run PR_URL=https://github.com/owner/repo/pull/123"

# ============================================================================
# Development
# ============================================================================

install:
	poetry install --only main

install-dev:
	poetry install

lock:
	poetry lock

lint:
	poetry run ruff check src/

format:
	poetry run ruff check --fix src/
	poetry run ruff format src/

typecheck:
	poetry run mypy src/

test:
	poetry run pytest tests/ -v

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================================================
# Build
# ============================================================================

build: clean
	poetry build

# Build Docker image (works with docker or podman)
docker-build:
	docker build -t codespy:latest .

# Check if image exists, build if not
docker-ensure-image:
	@docker image inspect codespy:latest >/dev/null 2>&1 || $(MAKE) docker-build

# ============================================================================
# Run
# ============================================================================

# Run review command (requires PR_URL environment variable)
# Usage: make review PR_URL=https://github.com/owner/repo/pull/123
review:
ifndef PR_URL
	$(error PR_URL is required. Usage: make review PR_URL=https://github.com/owner/repo/pull/123)
endif
	poetry run codespy review $(PR_URL)

# Run review with JSON output
review-json:
ifndef PR_URL
	$(error PR_URL is required. Usage: make review-json PR_URL=https://github.com/owner/repo/pull/123)
endif
	poetry run codespy review $(PR_URL) --output json

# Show current configuration
config:
	poetry run codespy config

# Run in Docker (requires PR_URL environment variable)
# Usage: make docker-run PR_URL=https://github.com/owner/repo/pull/123
# Works with both docker and podman
docker-run: docker-ensure-image
ifndef PR_URL
	$(error PR_URL is required. Usage: make docker-run PR_URL=https://github.com/owner/repo/pull/123)
endif
	@docker run --rm \
		--env-file .env \
		-v $${HOME}/.aws:/home/codespy/.aws:ro \
		-v codespy-cache:/home/codespy/.cache/codespy \
		codespy:latest review $(PR_URL)

# Run in Docker with JSON output
docker-run-json: docker-ensure-image
ifndef PR_URL
	$(error PR_URL is required. Usage: make docker-run-json PR_URL=https://github.com/owner/repo/pull/123)
endif
	@docker run --rm \
		--env-file .env \
		-v $${HOME}/.aws:/home/codespy/.aws:ro \
		-v codespy-cache:/home/codespy/.cache/codespy \
		codespy:latest review $(PR_URL) --output json

# ============================================================================
# Setup
# ============================================================================

# Create .env from example if it doesn't exist
setup-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example. Please edit it with your settings."; \
	else \
		echo ".env already exists."; \
	fi

# Full setup for new developers
setup: setup-env install-dev
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "1. Edit .env with your GitHub token and LLM settings"
	@echo "2. Run 'make review PR_URL=<github_pr_url>' to review a PR"