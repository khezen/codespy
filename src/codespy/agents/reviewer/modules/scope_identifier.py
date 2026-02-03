"""Scope identifier module for detecting code scopes in repositories."""

import asyncio
import logging
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]

from codespy.agents.reviewer.models import ScopeResult, ScopeType
from codespy.tools.github.models import PullRequest
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class ScopeIdentifierSignature(dspy.Signature):
    """Identify code scopes in a repository for a pull request.

    You have tools to clone the repository, explore its filesystem, and analyze code.
    Your goal is to identify logical code boundaries (scopes) and assign each changed file to exactly one scope.

    FIRST STEP - CLONE THE REPOSITORY:
    Before exploring, you MUST clone the repository using clone_repository tool:
    1. Use the repo_owner, repo_name, and head_sha provided in the inputs
    2. Clone to the target_repo_path provided
    3. For efficiency, derive sparse_paths from changed_files:
       - Extract unique parent directories from changed files
       - Include root directory for package manifests (use "." or specific manifest files)
       - Example: ["services/auth/", "libs/common/", "package.json", "go.mod"]
    4. Use depth=1 for fastest clone (single commit)

    EXPLORATION STRATEGY (after cloning):
    1. Examine the changed files list to understand what areas are affected
    2. Use get_tree to explore the repository structure (start with depth 3-4)
    3. Use file_exists and read_file to check for package manifest files (package.json, go.mod, pyproject.toml, Cargo.toml, etc.)
    4. Identify logical boundaries based on directory structure and conventions

    SCOPE TYPE CLASSIFICATION:
    - library: Shared code that others import (libs/, packages/, shared/, common/, core/)
      * Directories named: libs/, packages/, shared/, common/, core/
      * Multiple consumers importing from this scope
      * Generic/reusable code patterns
    - service: Isolated microservice with APIs (services/, microservices/, svc/, has HTTP handlers/gRPC)
      * Directories named: services/, microservices/, svc/
      * Own package manifest, often with server/API code
      * HTTP handlers, gRPC definitions, message consumers
    - application: Standalone app or frontend (apps/, web/, frontend/, mobile/, has main entry point)
      * Directories named: apps/, web/, frontend/, mobile/
      * Entry points (main.go, index.ts, App.tsx)
      * UI components, routing, state management
    - script: Build/deployment scripts, tooling (scripts/, bin/, tools/, hack/, ci/, .github/)
      * Directories named: scripts/, bin/, tools/, hack/, ci/, .github/
      * Shell scripts, Makefiles, Dockerfiles
      * CI/CD configuration, deployment scripts

    MONO-REPO PATTERNS:
    - Check root package.json for workspaces configuration
    - Look for lerna.json, pnpm-workspace.yaml, turbo.json
    - Directories under packages/, services/, apps/ often have their own package manifests

    PACKAGE MANIFESTS TO DETECT:
    - package.json (npm) with lock files: package-lock.json, yarn.lock, pnpm-lock.yaml
    - pyproject.toml (pip) with lock files: poetry.lock, uv.lock
    - go.mod (go) with lock file: go.sum
    - Cargo.toml (cargo) with lock file: Cargo.lock
    - pom.xml (maven), build.gradle (gradle), composer.json (composer), Gemfile (bundler)

    CRITICAL RULES:
    1. EVERY changed file must be assigned to exactly ONE scope
    2. Don't create overlapping scopes (parent contains child)
    3. For flat repos without sub-projects, use "." as the single scope containing all files
    4. Prefer the most specific scope that contains each file
    5. Use tools to verify package manifest existence - don't guess
    """

    changed_files: list[str] = dspy.InputField(
        desc="List of changed file paths from the PR. Use these to derive sparse_paths for efficient cloning."
    )
    repo_owner: str = dspy.InputField(desc="Repository owner (e.g., 'facebook')")
    repo_name: str = dspy.InputField(desc="Repository name (e.g., 'react')")
    head_sha: str = dspy.InputField(desc="Git commit SHA to checkout")
    target_repo_path: str = dspy.InputField(
        desc="Absolute path where repository should be cloned. Clone here before exploring."
    )
    pr_title: str = dspy.InputField(desc="PR title for additional context")
    pr_description: str = dspy.InputField(desc="PR description for additional context")
    
    scopes: list[ScopeResult] = dspy.OutputField(
        desc="List of identified scopes. EVERY changed file must appear in exactly one scope's changed_files list."
    )


class ScopeIdentifier(dspy.Module):
    """Agentic scope identifier using ReAct pattern with MCP tools.

    This module uses an LLM agent to explore the repository structure
    and identify logical code scopes for focused code review.
    """

    def __init__(self) -> None:
        """Initialize the scope identifier."""
        super().__init__()

    async def _create_mcp_tools(self, repo_path: Path) -> tuple[list[Any], list[Any]]:
        """Create DSPy tools from MCP servers."""
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        repo_path_str = str(repo_path)
        tools.extend(await connect_mcp_server(tools_dir / "filesystem" / "server.py", [repo_path_str], contexts))
        tools.extend(await connect_mcp_server(tools_dir / "parsers" / "ripgrep" / "server.py", [repo_path_str], contexts))
        tools.extend(await connect_mcp_server(tools_dir / "parsers" / "treesitter" / "server.py", [repo_path_str], contexts))
        tools.extend(await connect_mcp_server(tools_dir / "github" / "server.py", [], contexts))
        return tools, contexts

    async def aforward(self, pr: PullRequest, repo_path: Path) -> list[ScopeResult]:
        """Identify scopes in the repository for the given PR."""
        tools, contexts = await self._create_mcp_tools(repo_path)
        changed_file_paths = [f.filename for f in pr.changed_files]
        try:
            agent = dspy.ReAct(
                signature=ScopeIdentifierSignature,
                tools=tools,
                max_iters=20,
            )
            logger.info(f"Identifying scopes for {len(changed_file_paths)} changed files...")
            result = await agent.acall(
                changed_files=changed_file_paths,
                repo_owner=pr.repo_owner,
                repo_name=pr.repo_name,
                head_sha=pr.head_sha,
                target_repo_path=str(repo_path),
                pr_title=pr.title or "No title",
                pr_description=pr.body or "No description",
            )
            scopes = result.scopes
            # Ensure we got valid scopes
            if not scopes:
                raise ValueError("No scopes returned by agent")
        except Exception as e:
            logger.error(f"Agent failed: {e}")
            scopes = [ScopeResult(
                subroot=".",
                scope_type=ScopeType.APPLICATION,
                has_changes=True,
                is_dependency=False,
                confidence=0.5,
                language=None,
                package_manifest=None,
                changed_files=changed_file_paths,
                reason=f"Fallback due to agent error: {e}",
            )]
        finally:
            await cleanup_mcp_contexts(contexts)
        # Log results
        total_files = sum(len(s.changed_files) for s in scopes)
        logger.info(f"Identified {len(scopes)} scopes covering {total_files} files")
        return scopes

    def forward(self, pr: PullRequest, repo_path: Path) -> list[ScopeResult]:
        """Identify scopes (sync wrapper)."""
        return asyncio.run(self.aforward(pr, repo_path))