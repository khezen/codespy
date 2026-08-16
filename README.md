<p align="center">
  <img src="assets/codespy-logo.png" alt="CodeSpy logo">
</p>

<h1 align="center">Code<a href="https://github.com/khezen/codespy">Spy</a></h1>

<p align="center">
  An open-source AI reviewer that catches bugs, improves code quality, and integrates directly into your PR workflow, without sacrificing control or security.
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

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Why CodeSpy?](#why-codespy)
- [Features](#features)
- [Installation](#installation)
  - [Using pip](#using-pip)
  - [Using Homebrew (macOS/Linux)](#using-homebrew-macoslinux)
  - [Using Docker](#using-docker)
  - [Using Poetry (for development)](#using-poetry-for-development)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Contributors](#contributors)
- [License](#license)

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
- 🔄 Native PR integration
- 🧩 Extensible architecture
- 📦 100% open-source

Built for **engineering teams that care about correctness, security, and control.**

---

## Features

- 🔒 **Security Analysis** — Detects common vulnerabilities (injection, auth issues, data exposure) with CWE references
- 🐛 **Bug Detection** — Identifies logic errors, null references, resource leaks, edge cases
- 📝 **Documentation Review** — Checks for missing docstrings, outdated comments, incomplete docs
- 🔍 **Intelligent Scope Detection** — Automatically identifies code scopes (frontend, backend, infra, microservice in monorepo, etc.)
- 🧠 **Cross-Review Memory** — Agents learn patterns, constants, and domain knowledge from past reviews of the same codebase
- 💰 **Cost Tracking** — Track LLM calls, tokens, and costs per review
- 🤖 **Model Agnostic** — Works with OpenAI, AWS Bedrock, Anthropic, Gemini, Ollama, and more via LiteLLM
- 🐳 **Docker Ready** — Run locally or in the cloud with Docker
- <img src="assets/GitHub_Invertocat_Black.svg" height="20" alt="GitHub"> <img src="assets/gitlab-logo-500-rgb.png" height="20" alt="GitLab"> **GitHub & GitLab** — Works with both platforms, auto-detects from URL
- 🖥️ **Local Reviews** — Review local git changes without GitHub/GitLab — diff against any branch, ref, or review uncommitted work
- 🧩 **MCP Server** — IDE integration via Model Context Protocol — trigger reviews from AI coding assistants without leaving your editor
- 🔌 **GitHub Action** — One-line integration for automatic PR reviews

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
export DEFAULT_MODEL=anthropic/claude-opus-4-6
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx

# 3. Review a PR or MR!
codespy review https://github.com/owner/repo/pull/123
# OR
codespy review https://gitlab.com/group/project/-/merge_requests/123
```

codespy auto-discovers credentials from standard locations (`~/.aws/credentials`, `gh auth token`, `glab auth token`, etc.) - see [Configuration](docs/configuration.md) for details.

---

## Documentation

| Guide | Contents |
|-------|----------|
| **[Usage](docs/usage.md)** | CLI commands, Docker, GitHub Action, MCP server, output formats |
| **[Configuration](docs/configuration.md)** | Environment variables, YAML config, model strategy, per-signature settings |
| **[Architecture](docs/architecture.md)** | Pipeline design, DSPy signatures, supported languages |
| **[Memory System](docs/memory.md)** | Hippocampus episodic memory for cross-review knowledge |
| **[Development](docs/development.md)** | Setup, build, test, lint |

---

## Contributors

* @khezen
* @pranavsriram8

---

## License

MIT
