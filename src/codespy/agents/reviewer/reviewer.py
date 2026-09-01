"""Main review pipeline that orchestrates all review modules."""

import asyncio
import logging
import uuid
from pathlib import Path

import dspy  # type: ignore[import-untyped]

from codespy.agents import configure_dspy, get_cost_tracker, verify_model_access

from codespy.agents.memory.hippocampus.context_memory import Topic
from codespy.agents.reviewer.models import (
    Issue,
    LocalReviewConfig,
    PRContext,
    RemoteReviewConfig,
    ReviewConfig,
    ReviewContext,
    ReviewMetadata,
    ReviewResult,
    SignatureStatsResult,
)
from codespy.agents.reviewer.modules import (
    Auditor,
    CodeReviewer,
    DocReviewer,
    ScopeResolver,
    Summarizer,
    SupplyChainAuditor,
)
from codespy.agents.memory.hippocampus.episode import join_episode_saves
from codespy.agents.reviewer.modules.helpers import build_patches
from codespy.agents.reviewer.modules.scope_resolver import MANIFEST_FILES, MANIFEST_GLOBS
from codespy.config import Settings, get_settings
from codespy.config_memory import verify_memory_access
from codespy.tools.git import ChangedFile, GitClient, PullRequest, get_client
from codespy.tools.git.local_diff import build_pr_from_diff
from codespy.tools.git.patch_utils import compact_patches

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
        cache_dir = self.settings.cache_dir
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

        # Verify memory storage access
        self._verify_memory_access()

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

        # Step 1: Identify scopes FIRST
        is_local = isinstance(config, LocalReviewConfig)
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
        if not is_local:
            self._expand_sparse_for_scopes(scopes, repo_path)
        changed_file_paths = [f.filename for f in pr.changed_files]
        patches = build_patches(pr.changed_files)
        if self.settings.compact_patches:
            logger.info("Compacting patches to function boundaries...")
            compact_patches(scopes, repo_path)
        else:
            logger.debug("Compact patches disabled, using original PR patches")
        # Step 2: Run Summarizer (now receives scopes for per-scope episode persistence)
        # Build all scope topics (scope topics + PR topic)
        all_scope_topics = [s.topic(pr.repo_full_name) for s in scopes]
        all_scope_topics.append(pr_context.to_topic())
        pr_summary = self.summarizer(
            pr_context=pr_context,
            changed_file_paths=changed_file_paths,
            patches=patches,
            run_id=run_id,
            scopes=scopes,
            topics=all_scope_topics,
        )
        # Enrich review_ctx with actual summary
        pr_context.summary = pr_summary
        review_ctx = ReviewContext(pr_context=pr_context, memory=None, metadata=metadata)
        # Step 3: Run review modules concurrently via asyncio.gather
        module_names = ["code_reviewer", "doc_reviewer", "supply_chain_auditor"]
        logger.info(f"Running review modules concurrently: {', '.join(module_names)}...")
        all_issues = asyncio.run(
            self._run_review_modules(scopes, module_names, review_context=review_ctx)
        )
        logger.info(f"Found {len(all_issues)} issues")
        # Step 4: Run Audit (loads own prior episodes per scope, no memory inheritance from parallel modules)
        quality_assessment, recommendation = self.auditor(
            review_context=review_ctx,
            all_issues=all_issues,
            run_id=run_id,
            scopes=scopes,
            topics=all_scope_topics,
        )
        # Collect per-signature statistics
        signature_stats_list = self._collect_signature_stats()
        # Ensure all background episode saves complete before returning
        join_episode_saves()
        return ReviewResult(
            pr_number=pr.number,
            pr_title=pr.title,
            pr_url=pr.url,
            repo=pr.repo_full_name,
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
                )
            )

        return stats_list

    def _expand_sparse_for_scopes(self, scopes: list, repo_path: Path) -> None:
        """Expand sparse checkout to cover full subtree of each identified scope.

        Called after scope identification, before compact_patches and review modules,
        to ensure read_file and patch compaction have full scope context available.
        """
        from git import Repo
        from git.exc import GitCommandError

        git_dir = repo_path / ".git"
        if not git_dir.exists():
            return

        # Build scope-aware sparse paths
        sparse_paths: set[str] = set()
        for scope in scopes:
            if scope.subroot == ".":
                # Root scope — need everything; disable sparse checkout effectively
                sparse_paths.add("/*")
                sparse_paths.add("*/")
                break
            else:
                sparse_paths.add(scope.subroot.rstrip("/") + "/")

        # Always include root-level files and manifests
        sparse_paths.add("/*")
        for manifest in MANIFEST_FILES:
            sparse_paths.add(manifest)
        for pattern in MANIFEST_GLOBS:
            sparse_paths.add(pattern)

        sparse_file = git_dir / "info" / "sparse-checkout"
        sparse_file.write_text("\n".join(sorted(sparse_paths)) + "\n")

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
        except (GitCommandError, ValueError, TypeError) as e:
            logger.warning("Sparse checkout expansion failed (non-fatal): %s", e)
