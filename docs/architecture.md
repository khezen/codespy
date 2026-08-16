[← Back to README](../README.md#documentation)

# Architecture

## Pipeline Overview

CodeSpy's review pipeline follows a 4-step flow:

1. **Scope Identifier** (ReAct + tools) — Identifies code scopes (frontend, backend, infra, microservice in monorepo, etc.)
2. **Summarizer** (ChainOfThought) — Generates 2-3 sentence PR summary
3. **Parallel Review Modules** — Supply Chain Auditor, Code Reviewer, and Doc Reviewer run simultaneously
4. **Auditor** (ChainOfThought) — Generates quality assessment + recommendation (APPROVE / REQUEST_CHANGES / NEEDS_DISCUSSION)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           codespy CLI                               │
├─────────────────────────────────────────────────────────────────────┤
│  review <pr_url> | review-local | review-uncommitted | serve        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                  Git Platform Integration                           │
│  GitHub / GitLab — fetch diff, changed files, commit messages       │
│  Auto-detects platform · Sparse checkout for full context           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                   DSPy Review Pipeline                              │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ 1. Scope Identifier (ReAct + tools)                        │     │
│  │    Identifies code scopes: frontend, backend, infra, etc.  │     │
│  └──────────────────────────┬─────────────────────────────────┘     │
│                             │                                       │
│  ┌──────────────────────────▼─────────────────────────────────┐     │
│  │ 2. Summarizer (ChainOfThought)                             │     │
│  │    Generates 2-3 sentence PR summary                       │     │
│  └──────────────────────────┬─────────────────────────────────┘     │
│                             │                                       │
│  ┌──────────────────────────▼─────────────────────────────────┐     │
│  │ 3. Parallel Review Modules                                 │     │
│  │  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐       │     │
│  │  │ Supply Chain │ │    Code      │ │     Doc       │       │     │
│  │  │   Auditor    │ │  Reviewer    │ │   Reviewer    │       │     │
│  │  │ (ReAct+tools)│ │ (ReAct+tools)│ │(ChainOfThought│       │     │
│  │  └──────────────┘ └──────────────┘ └───────────────┘       │     │
│  └──────────────────────────┬─────────────────────────────────┘     │
│                             │                                       │
│  ┌──────────────────────────▼─────────────────────────────────┐     │
│  │ 4. Auditor (ChainOfThought)                                │     │
│  │    Quality assessment + recommendation                     │     │
│  │    (APPROVE / REQUEST_CHANGES / NEEDS_DISCUSSION)          │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                     │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐           │
│  │ Hippocampus Memory (cross-cutting)                   │           │
│  │ Episode persistence · Context memory · Distiller/    │           │
│  │ Cartographer reflection · Topic-based organization   │           │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘           │
│                                                                     │
│                     Cost Tracker (tokens, calls, $)                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        Tools Layer                                  │
│  Filesystem · Git (GH+GL) · Web · Cyber/OSV                         │
│  Parsers: Ripgrep (code search) · Tree-sitter (multi-lang AST)      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      LLM Backend (LiteLLM)                          │
│    Bedrock | OpenAI | Anthropic | Gemini | Ollama | Any compatible  │
└─────────────────────────────────────────────────────────────────────┘
```

## DSPy Signatures

| Signature | Config Key | Type | Description |
|-----------|------------|------|-------------|
| **ScopeIdentifierSignature** | `scope` | ReAct | Identifies code scopes (frontend, backend, infra, microservice in monorepo, etc.) |
| **PRSummarySignature** | `summary` | ChainOfThought | Generates PR summary |
| **CodeReviewSignature** | `code_review` | ReAct | Detects verified bugs, security vulnerabilities, removed defensive code, and code smells |
| **DocReviewSignature** | `doc` | ChainOfThought | Detects stale or wrong documentation caused by code changes |
| **SupplyChainSecuritySignature** | `supply_chain` | ReAct | Analyzes artifacts (Dockerfiles) and dependencies for supply chain security |
| **AuditSignature** | `audit` | ChainOfThought | Generates quality assessment and recommendation |

See [Configuration](configuration.md) for per-signature settings.

## Hippocampus Memory

Episode-based memory that wraps DSPy agents with persistent context across reviews. Agents accumulate knowledge about a codebase scope over time — patterns, constants, parsing schemas, and reuse it in subsequent reviews of the same code area.

See [Memory System](memory.md) for implementation details.

## Tools Layer

- **Filesystem**: `read_file`, `list_dir`
- **Git**: GitHub + GitLab clients, sparse checkout
- **Parsers**: Ripgrep (code search) + Tree-sitter (multi-language AST)
- **Web**: Browser-based web search
- **Cyber/OSV**: Vulnerability scanning

## Supported Languages

Tree-sitter based parsing for context-aware analysis:

| Language | Extensions | Features |
|----------|-----------|----------|
| Bash | `.sh`, `.bash` | Functions, commands |
| C/C++ | `.c`, `.cpp`, `.h`, `.hpp` | Functions, classes, structs |
| C# | `.cs` | Methods, classes, interfaces |
| Go | `.go` | Functions, structs, interfaces |
| Java | `.java` | Methods, classes, packages |
| JavaScript | `.js`, `.jsx` | Functions, classes, imports |
| Kotlin | `.kt` | Functions, classes, objects |
| Objective-C | `.m`, `.h` | Methods, interfaces, protocols |
| PHP | `.php` | Functions, classes, namespaces |
| Python | `.py` | Functions, classes, imports |
| Ruby | `.rb` | Methods, classes, modules |
| Rust | `.rs` | Functions, structs, traits, impl blocks |
| Swift | `.swift` | Functions, classes, structs |
| Terraform | `.tf` | Resources, data sources, modules, variables |
| TypeScript | `.ts`, `.tsx` | Functions, classes, interfaces |

All languages are supported for security, bug, and documentation analysis.

## LLM Backend

LiteLLM routing to Bedrock, OpenAI, Anthropic, Gemini, Ollama, Azure, and any OpenAI-compatible endpoint.

---

[← Back to README](../README.md#documentation)
