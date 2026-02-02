# syntax=docker/dockerfile:1

# Build stage with Poetry
FROM python:3.11-alpine AS builder

WORKDIR /app

# Install build dependencies for Python packages with native extensions
RUN apk add --no-cache \
    git \
    curl \
    gcc \
    musl-dev \
    libffi-dev \
    && pip install --no-cache-dir poetry

# Copy project files
COPY pyproject.toml poetry.lock* README.md ./
COPY src/ ./src/

# Configure Poetry: no virtualenv in container, install deps
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Runtime stage
FROM python:3.11-alpine

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache \
    git \
    ripgrep \
    && adduser -D -u 1000 codespy

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/codespy /usr/local/bin/codespy

# Copy source code
COPY src/ ./src/

# Set up cache directory
RUN mkdir -p /home/codespy/.cache/codespy && \
    chown -R codespy:codespy /home/codespy/.cache

# Switch to non-root user
USER codespy

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/codespy

CMD ["codespy", "--help"]