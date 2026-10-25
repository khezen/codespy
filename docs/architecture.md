[← Back to README](../README.md#documentation)

# Architecture

## Pipeline Overview

CodeSpy's review pipeline follows a 5-step flow:

1. **Scope Identifier** (RLM + tools) — Identifies code scopes (frontend, backend, infra, microservice in monorepo, etc.)
2. **PR Summary** (RLM) — Generates 2-3 sentence PR summary
3. **Parallel Review Modules** — Supply Chain Auditor, Code Reviewer, Doc Reviewer, and SEO Reviewer (opt-in) run simultaneously
4. **Auditor** (RLM) — Generates quality assessment + recommendation (APPROVE / REQUEST_CHANGES / NEEDS_DISCUSSION)
5. **Implementation Planner** (RLM + tools, opt-in) — Turns the findings into an ordered plan, one run per scope with issues, given the audit's assessment and recommendation. Skipped when there are no issues; a failure never blocks the review

The same pipeline serves three input modes. `review` fetches a PR/MR diff. `review-local` and `review-uncommitted` build one from local git. `review-path` builds a synthetic PR from a repository path where every reviewable file is newly added (`tools/git/full_path.py`), and delivers the report as a PR/MR that codespy opens (`workflows/review/reporters/pull_request.py`) instead of review comments.

### Sparse checkout

Remote reviews clone the repository treeless (`--filter=tree:0`) with a non-cone sparse checkout (`build_sparse_patterns` in `agents/review/scope/agent.py`). The pattern list is ordered, because the last matching pattern wins:

1. `/*` and `!/*/` keep the files at the root, then `/A/` and `!/A/*/` keep the files directly in each parent directory `A` of the changed files. This covers manifests, lock files, `project.json` and `AGENTS.md` without listing them.
2. The scope subtrees (`/<scope>/`), AI instruction directories and Xcode bundles of each parent directory.

After scope resolution the set grows to each scope's full subtree. A root scope switches sparse checkout off in PR reviews, never in `review-path`. Each change is applied with `git read-tree -mu HEAD`, which also prunes cached clones that hold more than the current set.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           codespy CLI                               │
├─────────────────────────────────────────────────────────────────────┤
│  review <pr_url> | review-path <repo_url> | learn | serve           │
│  review-local | review-uncommitted                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                  Git Platform Integration                           │
│  GitHub / GitLab — fetch diff, changed files, commit messages       │
│  review-path: every file under a path presented as added            │
│  Auto-detects platform · Sparse checkout for full context           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                   DSPy Review Pipeline                              │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ 1. Scope Identifier (RLM + tools)                            │     │
│  │    Identifies code scopes: frontend, backend, infra, etc.    │     │
│  └──────────────────────────┬─────────────────────────────────┘     │
│                             │                                       │
│  ┌──────────────────────────▼─────────────────────────────────┐     │
│  │ 2. PR Summary (RLM)                                         │     │
│  │    Generates 2-3 sentence PR summary                         │     │
│  └──────────────────────────┬─────────────────────────────────┘     │
│                             │                                       │
│  ┌──────────────────────────▼─────────────────────────────────┐     │
│  │ 3. Parallel Review Modules                                 │     │
│  │ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐│     │
│  │ │Supply Chain│ │    Code    │ │    Doc     │ │    SEO     ││     │
│  │ │  Auditor   │ │  Reviewer  │ │  Reviewer  │ │  Reviewer  ││     │
│  │ │(RLM+tools) │ │(RLM+tools) │ │   (RLM)    │ │(RLM+tools) ││     │
│  │ └────────────┘ └────────────┘ └────────────┘ └────────────┘│     │
│  └──────────────────────────┬─────────────────────────────────┘     │
│                             │                                       │
│  ┌──────────────────────────▼─────────────────────────────────┐     │
│  │ 4. Auditor (RLM)                                             │     │
│  │    Quality assessment + recommendation                     │     │
│  │    (APPROVE / REQUEST_CHANGES / NEEDS_DISCUSSION)          │     │
│  └──────────────────────────┬─────────────────────────────────┘     │
│                             │                                       │
│  ┌──────────────────────────▼─────────────────────────────────┐     │
│  │ 5. Implementation Planner (RLM + tools, opt-in)            │     │
│  │    Ordered fix plan per scope (collapsible section)        │     │
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
| **ScopeRefinementSignature** | `scope` | RLM | Identifies code scopes (frontend, backend, infra, microservice in monorepo, etc.) |
| **PRSummarySignature** | `summary` | RLM | Generates PR summary |
| **CodeReviewSignature** | `code_review` | RLM | Detects verified bugs, security vulnerabilities, removed defensive code, and code smells |
| **DocReviewSignature** | `doc` | RLM | Detects stale or wrong documentation caused by code changes |
| **SupplyChainSecuritySignature** | `supply_chain` | RLM | Analyzes artifacts (Dockerfiles) and dependencies for supply chain security |
| **SEOReviewSignature** | `seo` | RLM | Opt-in. Detects website changes that hurt crawling, indexing, ranking, or AI citation. Runs only on scopes detected as websites (or whitelisted) with web-relevant changes. A regex hotspot pre-scan (`seo/hotspots.py`, no LLM) passes candidate render-gating shapes as an optional `hotspots` input. The model verifies them and also runs its own sweep. In `review-path` mode the prompt switches from diff review to a full-scope audit |
| **AuditSignature** | `audit` | RLM | Generates quality assessment and recommendation |
| **ImplementationPlanSignature** | `plan` | RLM | Opt-in. Turns the findings of one scope into ordered steps (files, change, verification, addressed issues), using filesystem, ripgrep and tree-sitter tools to check call sites and tests |

See [Configuration](configuration.md) for per-signature settings.

### Context Window Overflow Resilience

All signature modules are wrapped in `ContextSafe`, which provides three-layer
defense against context-related quality degradation:

1. **Proactive threshold (context rot)**: Estimates input tokens and compares
   against a configurable ratio of `max_input_tokens`. When the threshold is
   exceeded (and input >= 8192 tokens), skips the inner module and uses a fallback
   RLM directly — preventing quality degradation that occurs well before the hard
   context limit. Thresholds differ by module type: RLM/ReAct (0.30),
   ChainOfThought (0.40), Predict (0.50).
2. **Pre-flight overflow**: For models in litellm's database, detects when
   input + output budget + safety margin would exceed the hard context limit.
3. **Try/catch**: For models not in litellm's database, catches the provider
   error and retries with RLM automatically.

RLM puts inputs into a sandboxed Python interpreter rather than the LLM prompt.
The model writes code to access and process variables (e.g., chunking large
patches via `llm_query()`), eliminating context window limits entirely.

The composition order is: `ContextSafe(Module)` — ContextSafe handles context quality.
Hippocampus is a separate component that agents call explicitly.

## Memory

CodeSpy's memory system has three independent components:

- **Hippocampus**: Episode-based memory that learns from the current review. Agents call
  `Hippocampus()` to create an episode, `aobserve()` to feed results, and `end_episode()`
  to run the Distiller/Cartographer reflection pipeline. Agents never receive Hippocampus
  context memory directly; prior knowledge reaches them only through Cerebral → Prefrontal.

- **Cerebral**: Retains Hippocampus episodes into Hindsight semantic memory, consolidates
  observations, and refreshes mental models (briefings).

- **Prefrontal**: Reads Cerebral before agent calls and injects a read-only `prefrontal_memory`
  input (briefings, recalled facts). RLM agents also get a `recall_memory` tool.

See [Memory System](memory.md) for implementation details.

## Tools Layer

- **Filesystem**: `read_file`, `list_directory`, `find_files`, `search_text`, `get_file_summary`
- **Git**: GitHub + GitLab clients, sparse checkout
- **Parsers**: Ripgrep (code search) + Tree-sitter (multi-language AST)
- **Site config** (SEO only): read-only `list_site_config` / `read_site_config` over tracked crawl and hosting files anywhere in the repo (robots, sitemap, redirects, hosting and framework config)
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
