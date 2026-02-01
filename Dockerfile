# syntax=docker/dockerfile:1

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 codespy

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

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

ENTRYPOINT ["codespy"]
CMD ["--help"]