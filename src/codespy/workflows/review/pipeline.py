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
    SEOReviewer,
    Summarizer,
    SupplyChainAuditor,
)
from codespy.agents.memory.hippocampus.episode import (
    defer_episode_saves,
    join_episode_saves,
    start_deferred_episode_saves,
)
from codespy.agents.review.helpers import build_patches
from codespy.agents.review.feedback import (
    drop_rejected,
    extract_feedback,
    record_feedback,
    render_commits,
    render_discussion,
)
from codespy.agents.review.scope import (
    MANIFEST_FILES,
    MANIFEST_GLOBS,
    append_sparse_patterns,
    apply_sparse_checkout,
    build_sparse_patterns,
)
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
    FullPathInfo,
    LocalReviewConfig,
    MemorySection,
    PathReviewConfig,
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
        self._path_info: FullPathInfo | None = None
        self.cost_tracker = get_cost_tracker()
        configure_dspy(self.settings)

        # Initialize all modules - they internally check if their signatures are enabled
        self.scope_resolver = ScopeResolver()
        self.code_reviewer = CodeReviewer()
        self.doc_reviewer = DocReviewer()
        self.seo_reviewer = SEOReviewer()
        self.supply_chain_auditor = SupplyChainAuditor()
        self.summarizer = Summarizer()
        self.auditor = Auditor()

    def _verify_code_interpreter(self) -> None:
        """Verify the Deno runtime that every RLM review agent needs.

        Without it each agent fails on its first step and soft-fails to "no
        findings", so the run would report a clean APPROVE without reviewing
        anything. Fail before any clone or LLM call instead.
        """
        from dspy.primitives.python_interpreter import (
            _find_deno_executable,
            _validate_deno_version,
        )

        deno = _find_deno_executable()
        try:
            _validate_deno_version(deno)
        except Exception as e:
            raise ValueError(
                f"Code interpreter unavailable: {e} "
                "codespy's review agents run in dspy's Deno sandbox. Install Deno 2.x "
                "(https://docs.deno.com/runtime/getting_started/installation/) and make sure "
                "`deno` is on PATH, or use the Docker image, which bundles it."
            ) from None
        logger.info(f"Code interpreter: {deno}")

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
            self.seo_reviewer.aforward(scopes=scopes, review_context=review_context),
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
            max_commits=self.settings.review.pr_context.max_commits,
        )

    def _build_path_pr(self, config: PathReviewConfig) -> tuple[PullRequest, Path, FullPathInfo]:
        """Sparse-clone ``config.path`` and build a synthetic all-added PullRequest.

        Fails fast (ValueError) on a non-branch ref, an empty path, or a cap
        breach, before any review LLM call.

        Returns:
            Tuple of (PullRequest, repo_path, FullPathInfo)
        """
        import shutil

        from codespy.tools.git.client import get_repo_client, parse_repo_url
        from codespy.tools.git.full_path import (
            build_pr_from_tree,
            check_caps,
            normalize_path,
            path_slug,
        )
        from codespy.tools.git.models import GitPlatform

        path = normalize_path(config.path)
        platform, host, owner, repo_name = parse_repo_url(config.repo_url)
        client = get_repo_client(config.repo_url, self.settings)

        ref = config.ref or client.get_default_branch(owner, repo_name)
        sha = client.resolve_branch(owner, repo_name, ref)  # ValueError if not a branch
        logger.info("Full-path review of %s @ %s (%s)", path, ref, sha[:8])

        # Dedicated clone per path, rebuilt every run: its sparse set is path-scoped
        cache_dir = Path(self.settings.review.cache_dir).expanduser()
        paths_root = (cache_dir / "paths").resolve()
        repo_path = paths_root / owner.replace("/", "_") / repo_name / path_slug(path)
        if not repo_path.resolve().is_relative_to(paths_root) or repo_path.resolve() == paths_root:
            raise ValueError(f"Refusing to use clone directory outside the cache: {repo_path}")
        if repo_path.exists():
            shutil.rmtree(repo_path)
        repo_path.parent.mkdir(parents=True, exist_ok=True)

        sparse_paths = build_sparse_patterns([], recursive_dirs=[path])
        logger.info("Sparse checkout patterns: %d", len(sparse_paths))
        client.clone_repository(
            owner=owner,
            repo_name=repo_name,
            ref=sha,
            target_path=repo_path,
            depth=1,
            sparse_paths=sparse_paths,
        )

        git_platform = GitPlatform.GITLAB if platform == "gitlab" else GitPlatform.GITHUB
        pr, stats = build_pr_from_tree(
            repo_path=repo_path,
            path=path,
            ref=ref,
            sha=sha,
            owner=owner,
            repo=repo_name,
            host=host,
            platform=git_platform,
            excluded_dirs=self.settings.review.excluded_directories,
        )
        caps = self.settings.review.full_path
        check_caps(
            stats,
            max_files=config.max_files if config.max_files is not None else caps.max_files,
            max_bytes=caps.max_bytes,
        )
        logger.info(
            "Path review: %d files (%s bytes), %d skipped",
            stats.files_reviewed, f"{stats.bytes_reviewed:,}", stats.files_skipped,
        )
        info = FullPathInfo(
            repo_url=config.repo_url,
            path=path,
            ref=ref,
            sha=sha,
            platform=platform,
            host=host,
            owner=owner,
            repo=repo_name,
            files_reviewed=stats.files_reviewed,
            files_skipped=stats.files_skipped,
            bytes_reviewed=stats.bytes_reviewed,
            skipped=dict(stats.skipped),
        )
        return pr, repo_path, info

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
        self._path_info = None

        # Every review agent needs the Deno sandbox: check before cloning or calling a model
        self._verify_code_interpreter()

        # Path mode: clone + cap check first, so an over-cap path fails before any LLM call
        path_setup: tuple[PullRequest, Path] | None = None
        if isinstance(config, PathReviewConfig):
            logger.info(f"Starting path review: {config.path} @ {config.ref or '<default>'} "
                        f"from {config.repo_url}")
            pr_, repo_path_, self._path_info = self._build_path_pr(config)
            path_setup = (pr_, repo_path_)

        # Always verify model access
        self._verify_model_access()

        # Verify memory storage access
        self._verify_memory_access()

        # Log configuration details
        config_path = get_loaded_config_path()
        if config_path:
            logger.info(f"Config loaded from: {config_path}")
        logger.info(f"min_confidence={self.settings.review.min_confidence:.2f}")
        for sig_name in ["code_review", "doc", "seo", "supply_chain", "summary", "audit"]:
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
        elif path_setup is not None:
            pr, repo_path = path_setup
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
        pr: PullRequest,
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
        is_full_path = isinstance(config, PathReviewConfig)

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
        self._fill_conversation_context(pr_context, pr)
        metadata = ReviewMetadata(
            repo_path=repo_path,
            run_id=run_id,
            pr=pr,
            is_local=is_local,
            is_full_path=is_full_path,
        )
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

        # Retain human feedback on earlier codespy reviews (deferred episode saves)
        if pr_context.review_feedback:
            record_feedback(self.settings, pr_context, scopes, run_id)

        # Expand sparse checkout to cover full scope subtrees
        changed_file_paths = [f.filename for f in pr.changed_files]
        if not is_local:
            self._expand_sparse_for_scopes(
                scopes,
                repo_path,
                changed_file_paths,
                full_path=self._path_info.path if self._path_info else None,
            )
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
            # Persist run-level recalls (opt-in: memory.hippocampus.persist_recalls)
            try:
                store = get_episode_store(self.settings)
                if store and store.persist_recalls and run_pf.recalls:
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
        module_names = ["code_reviewer", "doc_reviewer", "seo_reviewer", "supply_chain_auditor"]
        logger.info(f"Running review modules concurrently: {', '.join(module_names)}...")
        all_issues = asyncio.run(
            self._run_review_modules(scopes, module_names, review_context=review_ctx)
        )
        # Drop findings humans already rejected on this PR
        all_issues = drop_rejected(all_issues, pr_context.review_feedback)
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

        # Full-path info (path mode only), with the resolved scopes
        full_path_info = None
        if self._path_info is not None:
            full_path_info = self._path_info.model_copy(
                update={"scopes": [s.subroot for s in scopes]}
            )

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
            full_path=full_path_info,
        )

    def _fill_conversation_context(self, pr_context: PRContext, pr: Any) -> None:
        """Render commits, human discussion and review feedback into ``pr_context``.

        codespy's own comments are excluded (classified at fetch time).
        Never raises: conversation context is best-effort.
        """
        caps = self.settings.review.pr_context
        try:
            pr_context.commit_messages = render_commits(
                pr.commits, caps.max_comment_chars, caps.max_block_chars
            )
            pr_context.pr_discussion = render_discussion(
                pr, caps.max_comment_chars, caps.max_block_chars
            )
            if caps.feedback:
                pr_context.review_feedback = extract_feedback(pr)
            else:
                logger.info("Review feedback disabled (review.pr_context.feedback=false)")
        except Exception:
            logger.warning("Failed to build PR conversation context", exc_info=True)
        logger.info(
            "PR context: %d commits, %d comments, %d feedback items",
            len(pr.commits or []), len(pr.comments or []), len(pr_context.review_feedback),
        )

    def learn(self, url: str) -> int:
        """Harvest human feedback on codespy's reviews of a PR and retain it to memory.

        Posts nothing. Intended for CI on PR merge/close so feedback given after
        the last review is not lost. Unchanged feedback is skipped (hash check).

        Returns:
            Number of feedback episodes retained (0 when memory is disabled or
            there is no new feedback).
        """
        self.cost_tracker.reset()
        if not self.settings.memory.enabled:
            logger.info("Memory is disabled (memory.enabled=false); nothing to learn")
            return 0
        if not self.settings.review.pr_context.feedback:
            logger.info("Review feedback is disabled (review.pr_context.feedback=false)")
            return 0
        run_id = uuid.uuid4().hex
        self._verify_memory_access()
        pr = self._fetch_pr(url)
        pr_context = PRContext(
            repo_slug=pr.repo_slug,
            pr_number=pr.number,
            pr_title=pr.title,
            pr_url=pr.url,
            pr_description=pr.body or "",
            summary=pr.title,
        )
        self._fill_conversation_context(pr_context, pr)
        if not pr_context.review_feedback:
            logger.info("No human feedback on codespy reviews of this PR")
            return 0

        # Distiller needs the LLM; scopes must match the review run's topics.
        self._verify_model_access()
        repo_path = self._get_repo_path(pr)
        metadata = ReviewMetadata(repo_path=repo_path, run_id=run_id, pr=pr, is_local=False)
        review_ctx = ReviewContext(pr_context=pr_context, memory=None, metadata=metadata)
        try:
            with defer_episode_saves():
                scopes = self.scope_resolver(review_context=review_ctx)
                count = record_feedback(self.settings, pr_context, scopes, run_id)
        finally:
            start_deferred_episode_saves()
        join_episode_saves()
        if count:
            self._consolidate_run(run_id)
        return count

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

    def _expand_sparse_for_scopes(
        self,
        scopes: list,
        repo_path: Path,
        changed_files: list[str] | None = None,
        full_path: str | None = None,
    ) -> None:
        """Expand sparse checkout to cover full subtree of each identified scope.

        Called after scope identification and before review modules,
        to ensure read_file and patch compaction have full scope context available.

        PR mode: a root scope disables sparse checkout (the whole repo is the scope).
        Path mode (``full_path`` set): never widens beyond ``full_path`` plus the
        non-root scope subtrees (which may sit above the path, as read context).

        Args:
            scopes: List of identified scope results
            repo_path: Path to the repository root
            changed_files: Optional list of changed file paths (if None, extracts from scopes)
            full_path: Reviewed path for full-path mode, None for PR mode
        """
        from git import Repo
        from git.exc import GitCommandError

        git_dir = repo_path / ".git"
        if not git_dir.exists():
            return

        subtrees = [s.subroot.strip("/") for s in scopes if s.subroot not in (".", "")]

        if full_path is not None:
            sparse_paths = build_sparse_patterns([], recursive_dirs=[full_path, *subtrees])
        else:
            # Check for root scope — disable sparse checkout entirely
            has_root_scope = any(s.subroot == "." for s in scopes)
            if has_root_scope:
                try:
                    repo = Repo(repo_path)
                    if repo.head.is_valid():
                        # git-read-tree docs: select everything, re-apply, then disable
                        apply_sparse_checkout(repo_path, ["/*"])
                        repo.git.config("core.sparseCheckout", "false")
                        logger.info("Root scope: disabled sparse checkout, full repo checked out")
                    return
                except (GitCommandError, ValueError, TypeError) as e:
                    logger.warning("Failed to disable sparse checkout for root scope: %s", e)
                    return

            if changed_files is None:
                changed_files = [f.filename for scope in scopes for f in scope.changed_files]
            # Ordered list: never re-sort (last matching pattern wins)
            sparse_paths = append_sparse_patterns(
                build_sparse_patterns(changed_files), [f"/{s}/" for s in subtrees]
            )

        try:
            if not Repo(repo_path).head.is_valid():
                logger.warning("Skipping sparse expansion: HEAD is not valid (failed clone?)")
                return
            apply_sparse_checkout(repo_path, sparse_paths)
            logger.debug("Sparse checkout expanded for %d scope(s)", len(scopes))
        except (GitCommandError, ValueError, TypeError) as e:
            logger.warning("Sparse checkout expansion failed (non-fatal): %s", e)
