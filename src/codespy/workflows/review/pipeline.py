"""Main review pipeline that orchestrates all review modules."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]

from codespy.agents import configure_dspy, get_cost_tracker, verify_model_access

from codespy.agents.memory.hippocampus.context_memory import Topic
from codespy.agents.memory.prefrontal import build_facets
from codespy.agents.memory.prefrontal.query import MAX_CONTEXT_PATHS, MAX_ENTITY_TERMS
from codespy.agents.review.models import Issue, PRContext, ReviewContext, ReviewMetadata
from codespy.agents.review import (
    Auditor,
    CodeReviewer,
    DocReviewer,
    ScopeResolver,
    Summarizer,
    SupplyChainAuditor,
)
from codespy.agents.memory.hippocampus.episode import (
    defer_episode_saves,
    join_episode_saves,
    start_deferred_episode_saves,
)
from codespy.agents.review.helpers import build_patches
from codespy.agents.review.scope import MANIFEST_FILES, MANIFEST_GLOBS, build_sparse_patterns
from codespy.config import Settings, get_loaded_config_path, get_settings
from codespy.config_memory import (
    get_cerebral,
    get_episode_store,
    get_run_prefrontal,
    verify_memory_access,
)
from codespy.tools.git import ChangedFile, GitClient, PullRequest, get_client
from codespy.tools.git.local_diff import build_pr_from_diff

from codespy.workflows.review.models import (
    LocalReviewConfig,
    MemorySection,
    RecalledMemory,
    RemoteReviewConfig,
    ReviewConfig,
    ReviewResult,
    SignatureStatsResult,
)

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
        self.scope_resolver = ScopeResolver()
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

    def _verify_memory_access(self) -> None:
        """Verify memory storage access."""
        logger.info("Verifying memory storage access...")
        success, message = verify_memory_access(self.settings)
        if not success:
            raise ValueError(f"Memory storage access failed: {message}")
        logger.info(f"Memory storage: {message}")

    def _get_git_client(self, url: str) -> GitClient:
        """Get or create a Git client for the given URL."""
        if self._git_client is None:
            self._git_client = get_client(url, self.settings)
        return self._git_client

    def _fetch_pr(self, pr_url: str) -> PullRequest:
        """Fetch pull request data from Git platform."""
        client = self._get_git_client(pr_url)
        logger.info(f"Fetching PR data from {client.platform_name}...")
        pr = client.fetch_pull_request(pr_url)
        logger.info(f"PR #{pr.number}: {pr.title} ({len(pr.changed_files)} files)")
        return pr

    def _get_repo_path(self, pr: PullRequest) -> Path:
        """Get the local repository path for a MR, creating directories if needed."""
        cache_dir = self.settings.review.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Handle nested namespaces for GitLab
        owner_path = pr.repo_owner.replace("/", "_")
        return cache_dir / owner_path / pr.repo_name

    async def _run_review_modules(
        self,
        scopes: list,
        module_names: list[str],
        review_context: ReviewContext,
    ) -> list[Issue]:
        """Run review modules concurrently in a single event loop.

        Uses asyncio.gather instead of dspy.Parallel to avoid the
        multi-thread + multi-event-loop conflict that causes
        'cannot schedule new futures after shutdown' errors.

        Args:
            scopes: Identified scopes with changed files
            module_names: Names of modules (for error logging)
            review_context: ReviewContext for PR identity and pipeline metadata

        Returns:
            Aggregated list of issues
        """
        tasks = [
            self.code_reviewer.aforward(scopes=scopes, review_context=review_context),
            self.doc_reviewer.aforward(scopes=scopes, review_context=review_context),
            self.supply_chain_auditor.aforward(scopes=scopes, review_context=review_context),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_issues: list[Issue] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"{module_names[i]} failed: {result}", exc_info=result)
            elif result is not None:
                issues, _ = result
                all_issues.extend(issues)
        return all_issues

    def _build_local_pr(self, config: LocalReviewConfig) -> PullRequest:
        """Build a PullRequest from local git changes.

        Args:
            config: Local review configuration

        Returns:
            PullRequest object built from local git changes
        """
        logger.info(f"Building PR from local changes in {config.repo_path}...")
        return build_pr_from_diff(
            repo_path=config.repo_path,
            base_ref=config.base_ref,
            include_uncommitted=config.uncommitted,
        )

    def forward(self, config: ReviewConfig) -> ReviewResult:
        """Run the complete review pipeline (review phase only).

        The review phase includes scope identification, summarizer, review modules,
        and audit. Episode saves are deferred until finish_memory() is called.

        Args:
            config: Review configuration (RemoteReviewConfig or LocalReviewConfig)

        Returns:
            ReviewResult with issues, summary, costs, etc. (memory_pending=True)
        """
        self.cost_tracker.reset()

        # Generate a single run_id shared across all agents/modules invoked
        # within this pipeline run, used to correlate Episode records.
        run_id = uuid.uuid4().hex

        # Always verify model access
        self._verify_model_access()

        # Verify memory storage access
        self._verify_memory_access()

        # Log configuration details
        config_path = get_loaded_config_path()
        if config_path:
            logger.info(f"Config loaded from: {config_path}")
        logger.info(f"min_confidence={self.settings.review.min_confidence:.2f}")
        for sig_name in ["code_review", "doc", "supply_chain"]:
            max_iters = self.settings.get_max_iters(sig_name)
            max_llm_calls = self.settings.get_max_llm_calls(sig_name)
            logger.info(f"  {sig_name}: max_iters={max_iters}, max_llm_calls={max_llm_calls}")

        # Determine mode and fetch/build PR accordingly
        if isinstance(config, RemoteReviewConfig):
            # Remote mode: fetch from GitHub/GitLab
            logger.info(f"Starting review of {config.url}")
            pr = self._fetch_pr(config.url)
            repo_path = self._get_repo_path(pr)
        elif isinstance(config, LocalReviewConfig):
            # Local mode: build PR from local git changes
            mode = "uncommitted changes" if config.uncommitted else f"changes vs {config.base_ref}"
            logger.info(f"Starting local review: {mode} in {config.repo_path}")
            pr = self._build_local_pr(config)
            repo_path = config.repo_path.resolve()
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        # Run the review phase with deferred episode saves
        try:
            with defer_episode_saves():
                result = self._run_review_phase(config, pr, repo_path, run_id)
                return result
        except Exception:
            # Ensure saves are started on exception before re-raising
            start_deferred_episode_saves()
            raise

    def _run_review_phase(
        self,
        config: ReviewConfig,
        pr: Any,
        repo_path: Path,
        run_id: str,
    ) -> ReviewResult:
        """Run the review phase (scope identification through audit).

        Args:
            config: Review configuration
            pr: PullRequest object
            repo_path: Path to the repository
            run_id: Unique run identifier

        Returns:
            ReviewResult with memory_pending=True
        """
        is_local = isinstance(config, LocalReviewConfig)

        # Step 1: Identify scopes FIRST
        logger.info("Identifying code scopes...")
        pr_context = PRContext(
            repo_slug=pr.repo_slug,
            pr_number=pr.number,
            pr_title=pr.title,
            pr_url=pr.url,
            pr_description=pr.body or "",
            summary=pr.title,  # Use title as placeholder since summary hasn't run
        )
        metadata = ReviewMetadata(repo_path=repo_path, run_id=run_id, pr=pr, is_local=is_local)
        review_ctx = ReviewContext(pr_context=pr_context, memory=None, metadata=metadata)
        scopes = self.scope_resolver(review_context=review_ctx)
        for scope in scopes:
            logger.info(
                f"  Scope: {scope.subroot} ({scope.scope_type.value}) - "
                f"{len(scope.changed_files)} files"
            )
            if scope.package_manifest:
                manifest = scope.package_manifest
                logger.info(f"    Manifest: {manifest.manifest_path} ({manifest.package_manager})")
                if manifest.lock_file_path:
                    logger.info(f"    Lock file: {manifest.lock_file_path}")
                if manifest.dependencies_changed:
                    logger.info("    Dependencies changed: Yes")

        # Expand sparse checkout to cover full scope subtrees
        changed_file_paths = [f.filename for f in pr.changed_files]
        if not is_local:
            self._expand_sparse_for_scopes(scopes, repo_path, changed_file_paths)
        patches = build_patches(pr.changed_files)

        # Build all scope topics (scope topics + PR topic)
        all_scope_topics = [s.topic(pr.repo_full_name) for s in scopes]
        all_scope_topics.append(pr_context.to_topic())

        # Build scope topic IDs for Prefrontal
        scope_topic_ids = [t.id for t in all_scope_topics if t.type == "project_scope"]

        # Run-level Prefrontal recall: one load shared by all consumer agents
        run_pf = get_run_prefrontal(self.settings, pr.repo_full_name, scope_topic_ids)
        pf_text = ""
        if run_pf:
            # Build facets for the run-level load
            all_paths = list({f.filename for s in scopes for f in s.changed_files})
            all_packages = list({
                s.package_manifest.package_name
                for s in scopes
                if s.package_manifest and s.package_manifest.package_name
            })
            facets = build_facets(
                task="review",
                target=pr.repo_full_name,
                pr_title=pr.title,
                summary=pr.body or "",
                paths=all_paths[:MAX_CONTEXT_PATHS],
                packages=all_packages[:MAX_ENTITY_TERMS],
            )
            pf_text = run_pf.load(facets)
            # Persist run-level recalls
            try:
                store = get_episode_store(self.settings)
                if store and run_pf.recalls:
                    store.save_recalls(
                        run_pf.recalls,
                        run_id=run_id,
                        episode_id=None,
                        task="review",
                    )
            except Exception as e:
                logger.warning(f"Failed to save run-level recalls: {e}")

        # Step 2: Run Summarizer
        pr_summary = self.summarizer(
            pr_context=pr_context,
            changed_file_paths=changed_file_paths,
            patches=patches,
            run_id=run_id,
            scopes=scopes,
            topics=all_scope_topics,
            prefrontal_memory=pf_text,
        )
        # Enrich review_ctx with actual summary and prefrontal_memory
        pr_context.summary = pr_summary
        review_ctx = ReviewContext(
            pr_context=pr_context, memory=None, metadata=metadata, prefrontal_memory=pf_text
        )

        # Step 3: Run review modules concurrently via asyncio.gather
        module_names = ["code_reviewer", "doc_reviewer", "supply_chain_auditor"]
        logger.info(f"Running review modules concurrently: {', '.join(module_names)}...")
        all_issues = asyncio.run(
            self._run_review_modules(scopes, module_names, review_context=review_ctx)
        )
        logger.info(f"Found {len(all_issues)} issues")

        # Step 4: Run Audit
        quality_assessment, recommendation = self.auditor(
            review_context=review_ctx,
            all_issues=all_issues,
            run_id=run_id,
            scopes=scopes,
            topics=all_scope_topics,
        )

        # Build memories list from scope and run-level Prefrontal loads
        # Run-level first (flat), then scope (nested)
        memories: list[RecalledMemory] = []
        if run_pf and run_pf.last_sections:
            memories.append(
                RecalledMemory(
                    task="review",
                    nested=False,
                    sections=[MemorySection(title=t, text=b) for t, b in run_pf.last_sections],
                )
            )
        scope_pf_sections = self.scope_resolver.prefrontal_sections
        if scope_pf_sections:
            memories.append(
                RecalledMemory(
                    task="scope",
                    sections=[MemorySection(title=t, text=b) for t, b in scope_pf_sections],
                )
            )

        # Collect stats at end of review phase (before memory phase)
        signature_stats_list = self._collect_signature_stats()

        return ReviewResult(
            pr_number=pr.number,
            pr_title=pr.title,
            pr_url=pr.url,
            repo=pr.repo_full_name,
            run_id=run_id,
            model_used=self.settings.llm.default_model,
            issues=all_issues,
            overall_summary=pr_summary,
            quality_assessment=quality_assessment,
            recommendation=recommendation,
            total_cost=self.cost_tracker.total_cost,
            total_tokens=self.cost_tracker.total_tokens,
            llm_calls=self.cost_tracker.call_count,
            signature_stats=signature_stats_list,
            memories=memories,
            memory_pending=True,
        )

    def finish_memory(self, result: ReviewResult) -> ReviewResult:
        """Run the memory phase: start deferred saves, join them, consolidate.

        Args:
            result: ReviewResult from forward() with memory_pending=True

        Returns:
            ReviewResult with updated costs and memory_pending=False.
            Never raises - errors are logged and the result is returned.
        """
        if not result.memory_pending:
            return result

        try:
            # Start deferred saves (they were queued during review phase)
            start_deferred_episode_saves()
            # Wait for all saves to complete
            join_episode_saves()
            # Run consolidation and mental model update
            self._consolidate_run(result.run_id)
            # Recollect stats with memory costs
            signature_stats_list = self._collect_signature_stats()

            return result.model_copy(update={
                "signature_stats": signature_stats_list,
                "total_cost": self.cost_tracker.total_cost,
                "total_tokens": self.cost_tracker.total_tokens,
                "llm_calls": self.cost_tracker.call_count,
                "memory_pending": False,
            })
        except Exception as e:
            logger.warning("Memory phase failed: %s", e, exc_info=True)
            # Return result as-is, but mark memory as no longer pending
            # (we tried and failed, don't keep showing "pending")
            return result.model_copy(update={"memory_pending": False})

    def _consolidate_run(self, run_id: str) -> None:
        """Trigger one consolidation per run after all episode saves complete.

        Consolidation runs only when:
        - Memory is enabled globally
        - Cerebral is available
        - At least one signature is enabled
        """
        from codespy.config_dspy import SIGNATURE_NAMES

        # Check if memory is enabled and at least one signature is enabled
        if not self.settings.memory.enabled or not any(
            self.settings.is_signature_enabled(sig) for sig in SIGNATURE_NAMES
        ):
            return

        try:
            cerebral = get_cerebral(self.settings)
            if cerebral:
                cerebral.consolidate_run(run_id)
        except Exception:
            logger.warning("Consolidation failed for run %s", run_id, exc_info=True)

    def _collect_signature_stats(self) -> list[SignatureStatsResult]:
        """Collect statistics from all signatures that executed.

        Returns:
            List of SignatureStatsResult for each signature that ran
        """
        stats_list: list[SignatureStatsResult] = []
        all_signature_stats = self.cost_tracker.get_all_signature_stats()

        for signature_name, stats in all_signature_stats.items():
            stats_list.append(
                SignatureStatsResult(
                    name=signature_name,
                    cost=stats.cost,
                    tokens=stats.tokens,
                    call_count=stats.call_count,
                    duration_seconds=stats.duration_seconds,
                    input_tokens=stats.input_tokens,
                    output_tokens=stats.output_tokens,
                    input_cost=stats.input_cost,
                    output_cost=stats.output_cost,
                )
            )

        return stats_list

    def _expand_sparse_for_scopes(self, scopes: list, repo_path: Path, changed_files: list[str] | None = None) -> None:
        """Expand sparse checkout to cover full subtree of each identified scope.

        Called after scope identification and before review modules,
        to ensure read_file and patch compaction have full scope context available.

        Uses the same sparse pattern builder as derive_sparse_paths to ensure
        manifests and AI instruction files are consistently anchored to ancestor
        dirs of changed files, avoiding repo-wide pattern matching.

        Args:
            scopes: List of identified scope results
            repo_path: Path to the repository root
            changed_files: Optional list of changed file paths (if None, extracts from scopes)
        """
        from git import Repo
        from git.exc import GitCommandError

        git_dir = repo_path / ".git"
        if not git_dir.exists():
            return

        # Check for root scope — disable sparse checkout entirely
        has_root_scope = any(s.subroot == "." for s in scopes)
        if has_root_scope:
            try:
                repo = Repo(repo_path)
                repo.git.update_environment(GIT_TERMINAL_PROMPT="0")
                # Disable sparse checkout to get full repo
                repo.git.config("core.sparseCheckout", "false")
                # Re-checkout to materialize everything
                if repo.head.is_valid():
                    repo.git.checkout()
                    logger.info("Root scope: disabled sparse checkout, full repo checked out")
                return
            except (GitCommandError, ValueError, TypeError) as e:
                logger.warning("Failed to disable sparse checkout for root scope: %s", e)
                return

        # Build sparse patterns using the canonical builder
        if changed_files is None:
            changed_files = []
            for scope in scopes:
                changed_files.extend(f.filename for f in scope.changed_files)

        sparse_paths = build_sparse_patterns(changed_files)

        # Add scope subtrees (full directories) for expanded scope context
        for scope in scopes:
            if scope.subroot != ".":
                sparse_paths.append(f"/{scope.subroot.rstrip('/')}/")

        # Deduplicate and sort
        sparse_paths = sorted(set(sparse_paths))

        sparse_file = git_dir / "info" / "sparse-checkout"
        sparse_file.write_text("\n".join(sparse_paths) + "\n")

        # Re-checkout to materialize newly included paths
        try:
            repo = Repo(repo_path)
            repo.git.update_environment(GIT_TERMINAL_PROMPT="0")
            # Guard: repo.head.is_valid() is False for unborn branches (no commits fetched)
            if not repo.head.is_valid():
                logger.warning(
                    "Skipping sparse expansion: HEAD is not valid (clone may have failed)"
                )
                return
            repo.git.checkout()
            logger.debug("Sparse checkout expanded for %d scope(s)", len(scopes))
        except (GitCommandError, ValueError, TypeError) as e:
            logger.warning("Sparse checkout expansion failed (non-fatal): %s", e)
