"""Main review pipeline that orchestrates all review modules."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Sequence

import dspy  # type: ignore[import-untyped]

from codespy.agents import configure_dspy, get_cost_tracker, verify_model_access
from codespy.config import Settings, get_settings
from codespy.tools.git import GitClient, get_client, ChangedFile, MergeRequest
from codespy.tools.git.local_diff import build_mr_from_diff
from codespy.tools.git.patch_utils import compact_patches
from codespy.agents.memory.hippocampus import ContextMap
from codespy.agents.reviewer.models import (
    Issue,
    PRContext,
    ReviewContext,
    SignatureStatsResult,
    ReviewResult,
    ReviewConfig,
    RemoteReviewConfig,
    LocalReviewConfig,
)
from codespy.agents.reviewer.modules import (
    Auditor,
    CodeReviewer,
    DocReviewer,
    ScopeIdentifier,
    Summarizer,
    SupplyChainAuditor,
)
from codespy.agents.reviewer.modules.helpers import build_patches

logger = logging.getLogger(__name__)


class ReviewPipeline(dspy.Module):
    """Orchestrates the code review process using DSPy modules."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the review pipeline."""
        super().__init__()
        self.settings = settings or get_settings()
        self._git_client: GitClient | None = None
        self.cost_tracker = get_cost_tracker()
        configure_dspy(self.settings)

        # Initialize all modules - they internally check if their signatures are enabled
        self.scope_identifier = ScopeIdentifier()
        self.code_reviewer = CodeReviewer()
        self.doc_reviewer = DocReviewer()
        self.supply_chain_auditor = SupplyChainAuditor()
        self.summarizer = Summarizer()
        self.auditor = Auditor()

    def _verify_model_access(self) -> None:
        """Verify LLM model access."""
        logger.info("Verifying model access...")
        success, message = verify_model_access(self.settings)
        if not success:
            raise ValueError(f"Model access failed: {message}")
        logger.info(f"Model access: {message}")

    def _get_git_client(self, url: str) -> GitClient:
        """Get or create a Git client for the given URL."""
        if self._git_client is None:
            self._git_client = get_client(url, self.settings)
        return self._git_client

    def _fetch_mr(self, mr_url: str) -> MergeRequest:
        """Fetch merge request data from Git platform."""
        client = self._get_git_client(mr_url)
        logger.info(f"Fetching MR data from {client.platform_name}...")
        mr = client.fetch_merge_request(mr_url)
        logger.info(f"MR #{mr.number}: {mr.title} ({len(mr.changed_files)} files)")
        return mr

    def _get_repo_path(self, mr: MergeRequest) -> Path:
        """Get the local repository path for a MR, creating directories if needed."""
        cache_dir = self.settings.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Handle nested namespaces for GitLab
        owner_path = mr.repo_owner.replace("/", "_")
        return cache_dir / owner_path / mr.repo_name

    async def _run_review_modules(
        self,
        scopes: list,
        repo_path: Path,
        module_names: list[str],
        run_id: str | None = None,
        review_context: ReviewContext | None = None,
    ) -> tuple[list[Issue], dict[str, ContextMap | None]]:
        """Run review modules concurrently in a single event loop.

        Uses asyncio.gather instead of dspy.Parallel to avoid the
        multi-thread + multi-event-loop conflict that causes
        'cannot schedule new futures after shutdown' errors.

        Args:
            scopes: Identified scopes with changed files
            repo_path: Path to the cloned repository
            module_names: Names of modules (for error logging)
            run_id: Identifier of the pipeline run, shared across all agents
                invoked within the same review run
            review_context: ReviewContext for Hippocampus question and memory inheritance

        Returns:
            Tuple of (aggregated list of issues, dict of module_name -> context_map)
        """
        tasks = [
            self.code_reviewer.aforward(scopes=scopes, repo_path=repo_path, run_id=run_id, review_context=review_context),
            self.doc_reviewer.aforward(scopes=scopes, repo_path=repo_path, run_id=run_id, review_context=review_context),
            self.supply_chain_auditor.aforward(scopes=scopes, repo_path=repo_path, run_id=run_id, review_context=review_context),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_issues: list[Issue] = []
        context_maps: dict[str, ContextMap | None] = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"{module_names[i]} failed: {result}", exc_info=result)
            elif result is not None:
                issues, ctx_map = result
                all_issues.extend(issues)
                context_maps[module_names[i]] = ctx_map
        return all_issues, context_maps

    def _build_local_mr(self, config: LocalReviewConfig) -> MergeRequest:
        """Build a MergeRequest from local git changes.
        
        Args:
            config: Local review configuration
            
        Returns:
            MergeRequest object built from local git changes
        """
        logger.info(f"Building MR from local changes in {config.repo_path}...")
        return build_mr_from_diff(
            repo_path=config.repo_path,
            base_ref=config.base_ref,
            include_uncommitted=config.uncommitted
        )

    def forward(self, config: ReviewConfig) -> ReviewResult:
        """Run the complete review pipeline.

        Args:
            config: Review configuration (RemoteReviewConfig or LocalReviewConfig)

        Returns:
            ReviewResult with issues, summary, costs, etc.
        """
        self.cost_tracker.reset()

        # Generate a single run_id shared across all agents/modules invoked
        # within this pipeline run, used to correlate Episode records.
        run_id = uuid.uuid4().hex

        # Always verify model access
        self._verify_model_access()

        # Determine mode and fetch/build MR accordingly
        if isinstance(config, RemoteReviewConfig):
            # Remote mode: fetch from GitHub/GitLab
            logger.info(f"Starting review of {config.url}")
            mr = self._fetch_mr(config.url)
            repo_path = self._get_repo_path(mr)
        elif isinstance(config, LocalReviewConfig):
            # Local mode: build MR from local git changes
            mode = "uncommitted changes" if config.uncommitted else f"changes vs {config.base_ref}"
            logger.info(f"Starting local review: {mode} in {config.repo_path}")
            mr = self._build_local_mr(config)
            repo_path = config.repo_path.resolve()
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        # Step 1: Run Summarizer (before scope identification)
        changed_file_paths = [f.filename for f in mr.changed_files]
        patches = build_patches(mr.changed_files)
        pr_summary, summarizer_memory = self.summarizer(
            mr_title=mr.title,
            mr_description=mr.body or "No description provided.",
            mr_number=mr.number,
            changed_file_paths=changed_file_paths,
            patches=patches,
            repo_slug=mr.repo_slug,
            run_id=run_id,
        )

        # Build PRContext and ReviewContext after summarizer runs
        pr_context = PRContext(
            repo_slug=mr.repo_slug,
            mr_number=mr.number,
            mr_title=mr.title,
            summary=pr_summary,
        )
        # Summarizer is first stage - no inherited memory yet
        review_ctx = ReviewContext(pr_context=pr_context, memory=summarizer_memory)

        # Step 2: Identify scopes (inherits Summarizer memory)
        is_local = isinstance(config, LocalReviewConfig)
        logger.info("Identifying code scopes...")
        scopes, scope_memory = self.scope_identifier(
            mr, repo_path, is_local=is_local, run_id=run_id, review_context=review_ctx
        )
        for scope in scopes:
            logger.info(f"  Scope: {scope.subroot} ({scope.scope_type.value}) - {len(scope.changed_files)} files")
            if scope.package_manifest:
                manifest = scope.package_manifest
                logger.info(f"    Manifest: {manifest.manifest_path} ({manifest.package_manager})")
                if manifest.lock_file_path:
                    logger.info(f"    Lock file: {manifest.lock_file_path}")
                if manifest.dependencies_changed:
                    logger.info(f"    Dependencies changed: Yes")

        # Update ReviewContext with Scope Identifier's memory for downstream modules
        review_ctx = ReviewContext(pr_context=pr_context, memory=scope_memory)

        # Compact patches: expand context to function bodies for better review context
        logger.info("Compacting patches to function boundaries...")
        compact_patches(scopes, repo_path)

        # Step 3: Run review modules concurrently via asyncio.gather (inherit Scope Identifier memory)
        module_names = ["code_reviewer", "doc_reviewer", "supply_chain_auditor"]
        logger.info(f"Running review modules concurrently: {', '.join(module_names)}...")
        all_issues, parallel_memories = asyncio.run(
            self._run_review_modules(scopes, repo_path, module_names, run_id=run_id, review_context=review_ctx)
        )
        logger.info(f"Found {len(all_issues)} issues")

        # Merge parallel context maps for Auditor
        maps_to_merge = [m for m in parallel_memories.values() if m is not None]
        merged_memory = ContextMap.merge(*maps_to_merge) if maps_to_merge else scope_memory
        review_ctx = ReviewContext(pr_context=pr_context, memory=merged_memory)

        # Step 4: Run Audit (inherits merged memory from parallel modules)
        scoped_files = self._collect_scoped_files(scopes)
        logger.info(
            f"Audit input: {len(scoped_files)} in-scope files "
            f"(filtered from {len(mr.changed_files)} total)"
        )
        quality_assessment, recommendation = self.auditor(
            review_context=review_ctx,
            changed_files=scoped_files,
            all_issues=all_issues,
            run_id=run_id,
        )

        # Collect per-signature statistics
        signature_stats_list = self._collect_signature_stats()

        return ReviewResult(
            mr_number=mr.number,
            mr_title=mr.title,
            mr_url=mr.url,
            repo=mr.repo_full_name,
            run_id=run_id,
            model_used=self.settings.default_model,
            issues=all_issues,
            overall_summary=pr_summary,
            quality_assessment=quality_assessment,
            recommendation=recommendation,
            total_cost=self.cost_tracker.total_cost,
            total_tokens=self.cost_tracker.total_tokens,
            llm_calls=self.cost_tracker.call_count,
            signature_stats=signature_stats_list,
        )

    @staticmethod
    def _collect_scoped_files(scopes: list) -> list[ChangedFile]:
        """Collect de-duplicated changed files from identified scopes.

        The scope identifier already filters out binaries, vendor directories,
        lock files, etc. This method collects only the in-scope files so the
        summarizer operates on the same focused set as the review modules.

        Args:
            scopes: Identified scopes from scope_identifier

        Returns:
            De-duplicated list of ChangedFile objects from all scopes
        """
        seen: set[str] = set()
        scoped_files: list[ChangedFile] = []
        for scope in scopes:
            for f in scope.changed_files:
                if f.filename not in seen:
                    seen.add(f.filename)
                    scoped_files.append(f)
        return scoped_files

    def _collect_signature_stats(self) -> list[SignatureStatsResult]:
        """Collect statistics from all signatures that executed.

        Returns:
            List of SignatureStatsResult for each signature that ran
        """
        stats_list: list[SignatureStatsResult] = []
        all_signature_stats = self.cost_tracker.get_all_signature_stats()

        for signature_name, stats in all_signature_stats.items():
            stats_list.append(SignatureStatsResult(
                name=signature_name,
                cost=stats.cost,
                tokens=stats.tokens,
                call_count=stats.call_count,
                duration_seconds=stats.duration_seconds,
            ))

        return stats_list