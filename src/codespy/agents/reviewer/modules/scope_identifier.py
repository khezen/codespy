"""Scope identifier module for detecting code scopes in repositories."""

import asyncio
import logging
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.reviewer.models import PackageManifest, ScopeResult, ScopeType
from codespy.config import get_settings
from codespy.tools.git.models import ChangedFile, MergeRequest, should_review_file
from codespy.tools.mcp_utils import cleanup_mcp_contexts, connect_mcp_server

logger = logging.getLogger(__name__)


class ScopeAssignment(BaseModel):
    """LLM-friendly scope assignment with string file paths.
    
    This intermediate model is used for LLM output since the LLM can only
    produce string file paths, not full ChangedFile objects with patches/content.
    It gets converted to ScopeResult with proper ChangedFile objects after LLM call.
    """

    subroot: str = Field(description="Path relative to repo root (e.g., packages/auth)")
    scope_type: ScopeType = Field(description="Type of scope (library, service, etc.)")
    has_changes: bool = Field(
        default=False, description="Whether this scope has changed files from PR"
    )
    is_dependency: bool = Field(
        default=False, description="Whether this scope depends on a changed scope"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence score for scope identification"
    )
    language: str | None = Field(default=None, description="Primary language detected")
    package_manifest: PackageManifest | None = Field(
        default=None, description="Package manifest info if present"
    )
    changed_files: list[str] = Field(
        default_factory=list, description="Changed file paths belonging to this scope"
    )
    reason: str = Field(description="Explanation for why this scope was identified")


class ScopeIdentifierSignature(dspy.Signature):
    """Identify code scopes in a repository for a merge request.

    You have tools to clone the repository, explore its filesystem, and analyze code.
    Your goal is to identify logical code boundaries (scopes) and assign each changed file to exactly one scope.

    STEP 1 - ANALYZE CHANGED FILE PATHS (before cloning):
    The changed file paths are your MOST IMPORTANT signal for scope detection.
    1. Extract common directory prefixes from changed files to find candidate scopes
    2. Look for scope indicator patterns at ANY DEPTH in the path:
       - svc/, services/, microservices/ → likely a service scope
       - libs/, packages/, shared/, common/, core/ → likely a library scope
       - apps/, web/, frontend/, mobile/ → likely an application scope
       - scripts/, bin/, tools/, hack/, ci/, .github/, .gitlab/ → likely a script scope
    3. EXAMPLES of nested scope detection from file paths:
       - Files: mono/svc/my-service-v1/internal/handler.go, mono/svc/my-service-v1/cmd/main.go
         → Candidate scope: mono/svc/my-service-v1 (the "svc/" pattern indicates service)
       - Files: platform/packages/auth/src/index.ts, platform/packages/auth/package.json
         → Candidate scope: platform/packages/auth (the "packages/" pattern indicates library)
       - Files: company/backend/services/user-api/main.go
         → Candidate scope: company/backend/services/user-api
    4. Group files by their longest common directory prefix that contains a scope indicator

    STEP 2 - CLONE THE REPOSITORY:
    Clone using clone_repository tool:
    1. Use the repo_owner, repo_name, and head_sha provided in the inputs
    2. Clone to the target_repo_path provided
    3. Derive sparse_paths from candidate scopes identified in STEP 1:
       - Include each candidate scope directory
       - Example: ["mono/svc/my-service-v1/", "libs/common/"]
    4. Use depth=1 for fastest clone (single commit)

    STEP 3 - VERIFY SCOPES WITH PACKAGE MANIFESTS:
    For each candidate scope from STEP 1:
    1. Check if a package manifest exists at that path (go.mod, package.json, pyproject.toml, Cargo.toml, etc.)
    2. If found → CONFIRM that directory as the scope root
    3. If NOT found → Walk UP parent directories until you find a package manifest
       - Example: If candidate is mono/svc/my-service-v1/internal, check:
         * mono/svc/my-service-v1/internal/go.mod (not found)
         * mono/svc/my-service-v1/go.mod (FOUND → this is the scope)
    4. The directory containing the package manifest is the authoritative scope root

    SCOPE TYPE CLASSIFICATION:
    These patterns can appear at ANY NESTING DEPTH - not just at the repository root!
    - library: Shared code that others import
      * Patterns at any depth: */libs/*, */packages/*, */shared/*, */common/*, */core/*
      * Multiple consumers importing from this scope
      * Generic/reusable code patterns
    - service: Isolated microservice with APIs
      * Patterns at any depth: */services/*, */microservices/*, */svc/*
      * Own package manifest, often with server/API code
      * HTTP handlers, gRPC definitions, message consumers
    - application: Standalone app or frontend
      * Patterns at any depth: */apps/*, */web/*, */frontend/*, */mobile/*
      * Entry points (main.go, index.ts, App.tsx)
      * UI components, routing, state management
    - script: Build/deployment scripts, tooling
      * Patterns at any depth: */scripts/*, */bin/*, */tools/*, */hack/*, */ci/*, */.github/*
      * Shell scripts, Makefiles, Dockerfiles
      * CI/CD configuration, deployment scripts

    MONO-REPO PATTERNS (can be nested!):
    - Check root AND nested package.json for workspaces configuration
    - Look for lerna.json, pnpm-workspace.yaml, turbo.json at various depths
    - Scope indicator directories (packages/, services/, apps/, svc/) can appear at any level:
      * repo/services/api/ ← traditional mono-repo
      * repo/mono/svc/user-api/ ← nested mono-repo
      * repo/platform/backend/services/api/ ← deeply nested

    PACKAGE MANIFESTS TO DETECT:
    - package.json (npm) with lock files: package-lock.json, yarn.lock, pnpm-lock.yaml
    - pyproject.toml (pip) with lock files: poetry.lock, uv.lock
    - go.mod (go) with lock file: go.sum
    - Cargo.toml (cargo) with lock file: Cargo.lock
    - pom.xml (maven), build.gradle (gradle), composer.json (composer), Gemfile (bundler)

    CRITICAL RULES:
    1. EVERY changed file must be assigned to exactly ONE scope
    2. Don't create overlapping scopes (parent contains child)
    3. ALWAYS prefer the most specific scope - the deepest directory with a package manifest
       - If files are in mono/svc/my-service-v1/, use that as scope, NOT "." or "mono/"
    4. Use "." as scope ONLY when:
       - Files are truly at the repo root with no nested project structure
       - No package manifest exists at any deeper level
       - Changed files span multiple unrelated directories with no common scope indicator
    5. Use tools to verify package manifest existence - don't guess
    6. Trust the file paths - if they contain svc/, services/, packages/ etc., that's a strong scope signal

    OUTPUT EFFICIENCY: Group files by common directory prefix in reasoning. Do not reason about each file path individually.
    Keep each reasoning step to 1-2 sentences.
    """

    changed_files: list[str] = dspy.InputField(
        desc="List of changed file paths from the MR. Use these to derive sparse_paths for efficient cloning."
    )
    repo_owner: str = dspy.InputField(desc="Repository owner/namespace (e.g., 'facebook' or 'group/subgroup')")
    repo_name: str = dspy.InputField(desc="Repository name (e.g., 'react')")
    head_sha: str = dspy.InputField(desc="Git commit SHA to checkout")
    target_repo_path: str = dspy.InputField(
        desc="Absolute path where repository should be cloned. Clone here before exploring."
    )
    mr_title: str = dspy.InputField(desc="MR title for additional context")
    mr_description: str = dspy.InputField(desc="MR description for additional context")
    
    scopes: list[ScopeAssignment] = dspy.OutputField(
        desc="Identified scopes. Every changed file must appear in exactly one scope. Use concise reasons (<2 sentences)."
    )


class ScopeIdentifier(dspy.Module):
    """Agentic scope identifier using ReAct pattern with MCP tools.

    This module uses an LLM agent to explore the repository structure
    and identify logical code scopes for focused code review.
    """

    def __init__(self) -> None:
        """Initialize the scope identifier."""
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    async def _create_mcp_tools(self, repo_path: Path) -> tuple[list[Any], list[Any]]:
        """Create DSPy tools from MCP servers."""
        tools: list[Any] = []
        contexts: list[Any] = []
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        repo_path_str = str(repo_path)
        caller = "scope_identifier"
        tools.extend(await connect_mcp_server(tools_dir / "filesystem" / "server.py", [repo_path_str], contexts, caller))
        tools.extend(await connect_mcp_server(tools_dir / "parsers" / "ripgrep" / "server.py", [repo_path_str], contexts, caller))
        tools.extend(await connect_mcp_server(tools_dir / "parsers" / "treesitter" / "server.py", [repo_path_str], contexts, caller))
        tools.extend(await connect_mcp_server(tools_dir / "git" / "server.py", [], contexts, caller))
        return tools, contexts

    async def aforward(self, mr: MergeRequest, repo_path: Path) -> list[ScopeResult]:
        """Identify scopes in the repository for the given MR."""
        # Get excluded directories from settings
        excluded_dirs = self._settings.excluded_directories
        
        # Filter out binary, lock files, minified files, excluded directories, etc.
        reviewable_files = [f for f in mr.changed_files if should_review_file(f, excluded_dirs)]
        excluded_count = len(mr.changed_files) - len(reviewable_files)
        if excluded_count > 0:
            excluded_files = [f.filename for f in mr.changed_files if not should_review_file(f, excluded_dirs)]
            logger.info(f"Excluded {excluded_count} non-reviewable files: {excluded_files[:10]}{'...' if len(excluded_files) > 10 else ''}")
        
        if not reviewable_files:
            logger.warning("No reviewable files in MR - all files are binary, lock files, or in excluded directories")
            return []
        
        # Check if signature is enabled
        if not self._settings.is_signature_enabled("scope_identification"):
            logger.warning("scope_identification is disabled - using fallback single scope")
            return [ScopeResult(
                subroot=".",
                scope_type=ScopeType.APPLICATION,
                has_changes=True,
                is_dependency=False,
                confidence=0.5,
                language=None,
                package_manifest=None,
                changed_files=reviewable_files,
                reason="Scope identification disabled - fallback to single scope",
            )]
        tools, contexts = await self._create_mcp_tools(repo_path)
        changed_file_paths = [f.filename for f in reviewable_files]
        # Build map from filename to ChangedFile for post-processing
        changed_files_map: dict[str, ChangedFile] = {f.filename: f for f in reviewable_files}
        try:
            # Get per-signature config
            max_iters = self._settings.get_max_iters("scope_identification")
            temperature = self._settings.get_temperature("scope_identification")
            max_reasoning = self._settings.get_max_reasoning_tokens("scope_identification")
            
            # Create ReAct agent
            agent = dspy.ReAct(
                signature=ScopeIdentifierSignature,
                tools=tools,
                max_iters=max_iters,
            )
            logger.info(f"Identifying scopes for {len(changed_file_paths)} changed files...")
            # Track scope_identification signature costs
            async with SignatureContext("scope_identification", self._cost_tracker):
                result = await agent.acall(
                    changed_files=changed_file_paths,
                    repo_owner=mr.repo_owner,
                    repo_name=mr.repo_name,
                    head_sha=mr.head_sha,
                    target_repo_path=str(repo_path),
                    mr_title=mr.title or "No title",
                    mr_description=mr.body or "No description",
                )
            scope_assignments: list[ScopeAssignment] = result.scopes
            # Ensure we got valid scopes
            if not scope_assignments:
                raise ValueError("No scopes returned by agent")
            # Convert ScopeAssignment (with string paths) to ScopeResult (with ChangedFile objects)
            scopes = self._convert_assignments_to_results(scope_assignments, changed_files_map)
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
                changed_files=list(mr.changed_files),
                reason=f"Fallback due to agent error: {e}",
            )]
        finally:
            await cleanup_mcp_contexts(contexts)
        # Log results
        total_files = sum(len(s.changed_files) for s in scopes)
        logger.info(f"Identified {len(scopes)} scopes covering {total_files} files")
        return scopes

    def _convert_assignments_to_results(
        self, 
        assignments: list[ScopeAssignment], 
        changed_files_map: dict[str, ChangedFile]
    ) -> list[ScopeResult]:
        """Convert LLM scope assignments to ScopeResults with proper ChangedFile objects.
        
        Args:
            assignments: Scope assignments from LLM with string file paths
            changed_files_map: Map from filename to ChangedFile object
            
        Returns:
            List of ScopeResult with ChangedFile objects instead of strings
        """
        results: list[ScopeResult] = []
        for assignment in assignments:
            # Map string paths to ChangedFile objects
            changed_files: list[ChangedFile] = []
            for filepath in assignment.changed_files:
                if filepath in changed_files_map:
                    changed_files.append(changed_files_map[filepath])
                else:
                    logger.warning(f"File '{filepath}' from scope assignment not found in PR changed files")
            results.append(ScopeResult(
                subroot=assignment.subroot,
                scope_type=assignment.scope_type,
                has_changes=assignment.has_changes,
                is_dependency=assignment.is_dependency,
                confidence=assignment.confidence,
                language=assignment.language,
                package_manifest=assignment.package_manifest,
                changed_files=changed_files,
                reason=assignment.reason,
            ))
        return results

    def forward(self, mr: MergeRequest, repo_path: Path) -> list[ScopeResult]:
        """Identify scopes (sync wrapper)."""
        return asyncio.run(self.aforward(mr, repo_path))
