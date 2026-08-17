# syntax=docker/dockerfile:1

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    gcc \
    libffi-dev \
    && pip install --no-cache-dir poetry \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml poetry.lock* README.md ./
COPY src/ ./src/

# Configure Poetry: no virtualenv in container, install deps
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 codespy

# Copy Deno binary (glibc works natively on Debian)
COPY --from=denoland/deno:bin-2.9.5 /deno /usr/local/bin/deno

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/codespy /usr/local/bin/codespy

# Copy source code
COPY src/ ./src/

# Copy config to user's home directory
COPY codespy.yaml /home/codespy/codespy.yaml

# Pre-cache Deno/Pyodide dependencies and set up directories
ENV DENO_DIR=/home/codespy/.cache/deno
RUN mkdir -p /home/codespy/.cache/codespy \
    && (deno cache /usr/local/lib/python3.11/site-packages/dspy/primitives/runner.js || true) \
    && chown -R codespy:codespy /home/codespy/.cache /home/codespy/codespy.yaml

# Switch to non-root user
USER codespy

WORKDIR /home/codespy

ENV PYTHONUNBUFFERED=1
ENV HOME=/home/codespy
ENV DENO_DIR=/home/codespy/.cache/deno

ENTRYPOINT ["codespy"]
CMD ["--help"]
