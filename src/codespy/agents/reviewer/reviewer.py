"""Main review pipeline that orchestrates all review modules."""

import logging
from pathlib import Path

import dspy  # type: ignore[import-untyped]

from codespy.agents import SignatureContext, configure_dspy, get_cost_tracker, verify_model_access
from codespy.config import Settings, get_settings
from codespy.tools.github.client import GitHubClient
from codespy.tools.github.models import ChangedFile, PullRequest
from codespy.agents.reviewer.models import Issue, SignatureStatsResult, ReviewResult
from codespy.agents.reviewer.modules import (
    BugDetector,
    DomainExpert,
    DocumentationReviewer,
    IssueDeduplicator,
    ScopeIdentifier,
    SecurityAuditor,
)

logger = logging.getLogger(__name__)


class PRSummarySignature(dspy.Signature):
    """Generate an overall summary and recommendation for a pull request.

    Based on all the issues found during review, provide:
    - A concise summary of what the PR does
    - An overall assessment of the code quality
    - A recommendation (approve, request changes, or needs discussion)
    """

    pr_title: str = dspy.InputField(desc="Title of the pull request")
    pr_description: str = dspy.InputField(desc="Description/body of the PR")
    changed_files: list[ChangedFile] = dspy.InputField(
        desc="List of changed files with their status and line counts"
    )
    all_issues: list[Issue] = dspy.InputField(
        desc="All issues found during review"
    )

    summary: str = dspy.OutputField(
        desc="2-3 sentence summary of what this PR accomplishes"
    )
    quality_assessment: str = dspy.OutputField(
        desc="Overall assessment of code quality (e.g., well-structured, needs refactoring, follows best practices, etc.)"
    )
    recommendation: str = dspy.OutputField(
        desc="One of: APPROVE, REQUEST_CHANGES, or NEEDS_DISCUSSION with brief justification"
    )


class ReviewPipeline(dspy.Module):
    """Orchestrates the code review process using DSPy modules."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the review pipeline."""
        super().__init__()
        self.settings = settings or get_settings()
        self.github_client = GitHubClient(self.settings)
        self.cost_tracker = get_cost_tracker()
        configure_dspy(self.settings)

        # Initialize all modules - they internally check if their signatures are enabled
        self.scope_identifier = ScopeIdentifier()
        self.security_auditor = SecurityAuditor()
        self.bug_detector = BugDetector()
        self.doc_reviewer = DocumentationReviewer()
        self.domain_expert = DomainExpert()
        self.deduplicator = IssueDeduplicator()

    def _verify_model_access(self) -> None:
        """Verify LLM model access."""
        logger.info("Verifying model access...")
        success, message = verify_model_access(self.settings)
        if not success:
            raise ValueError(f"Model access failed: {message}")
        logger.info(f"Model access: {message}")

    def _fetch_pr(self, pr_url: str) -> PullRequest:
        """Fetch PR data from GitHub."""
        logger.info("Fetching PR data from GitHub...")
        pr = self.github_client.fetch_pull_request(pr_url)
        logger.info(f"PR #{pr.number}: {pr.title} ({len(pr.changed_files)} files)")
        return pr

    def _get_repo_path(self, pr: PullRequest) -> Path:
        """Get the local repository path for a PR, creating directories if needed."""
        cache_dir = self.settings.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / pr.repo_owner / pr.repo_name

    def forward(self, pr_url: str, verify_model: bool = True) -> ReviewResult:
        """Run the complete review pipeline on a pull request."""
        self.cost_tracker.reset()
        logger.info(f"Starting review of {pr_url}")
        if verify_model:
            self._verify_model_access()
        pr = self._fetch_pr(pr_url)
        repo_path = self._get_repo_path(pr)

        # Identify scopes (the module internally checks if signature is enabled)
        logger.info("Identifying code scopes...")
        scopes = self.scope_identifier(pr, repo_path)
        for scope in scopes:
            logger.info(f"  Scope: {scope.subroot} ({scope.scope_type.value}) - {len(scope.changed_files)} files")

        # Build list of review modules (they check signature enabled status internally)
        all_issues: list[Issue] = []
        exec_pairs = [
            (self.bug_detector, {"scopes": scopes, "repo_path": repo_path}),
            (self.security_auditor, {"scopes": scopes, "repo_path": repo_path}),
            (self.doc_reviewer, {"scopes": scopes, "repo_path": repo_path}),
            (self.domain_expert, {"scopes": scopes, "repo_path": repo_path}),
        ]
        module_names = ["bug_detector", "security_auditor", "doc_reviewer", "domain_expert"]

        logger.info(f"Running review modules in parallel: {', '.join(module_names)}...")
        # Execute in parallel with error handling
        if exec_pairs:
            parallel = dspy.Parallel(num_threads=len(exec_pairs), return_failed_examples=True, provide_traceback=True)
            results, failed_examples, exceptions = parallel(exec_pairs)
            # Log any failures
            for i, (failed, exc) in enumerate(zip(failed_examples, exceptions)):
                if failed is not None and exc is not None:
                    logger.error(f"{module_names[i]} failed: {exc}")
            # Aggregate successful results
            for result in results:
                if result is not None:
                    all_issues.extend(result)
            logger.info(f"Found {len(all_issues)} issues before deduplication")

        # Deduplicate issues across reviewers (deduplicator checks if enabled internally)
        if all_issues:
            logger.info("Deduplicating issues...")
            all_issues = self.deduplicator(all_issues)
            logger.info(f"After deduplication: {len(all_issues)} unique issues")
        # Generate summary, quality assessment, and recommendation
        if self.settings.is_signature_enabled("summarization"):
            logger.info("Generating PR summary...")
            try:
                summarizer = dspy.ChainOfThought(PRSummarySignature)
                # Track the summarization signature's costs
                with SignatureContext("summarization", self.cost_tracker):
                    result = summarizer(
                        pr_title=pr.title,
                        pr_description=pr.body or "No description provided.",
                        changed_files=pr.changed_files,
                        all_issues=all_issues,
                    )
                summary = result.summary
                quality_assessment = result.quality_assessment
                recommendation = result.recommendation
            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
                summary = f"Reviewed {len(pr.changed_files)} files with {len(all_issues)} issues."
                quality_assessment = "Unable to assess due to error."
                recommendation = "NEEDS_DISCUSSION: Summary generation failed."
        else:
            logger.debug("Skipping summarization: disabled")
            summary = f"Reviewed {len(pr.changed_files)} files with {len(all_issues)} issues."
            quality_assessment = "Summarization disabled."
            recommendation = "NEEDS_DISCUSSION" if all_issues else "APPROVE"
        # Collect per-signature statistics
        signature_stats_list = self._collect_signature_stats()
        
        return ReviewResult(
            pr_number=pr.number,
            pr_title=pr.title,
            pr_url=pr.url,
            repo=pr.repo_full_name,
            model_used=self.settings.default_model,
            issues=all_issues,
            overall_summary=summary,
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
            stats_list.append(SignatureStatsResult(
                name=signature_name,
                cost=stats.cost,
                tokens=stats.tokens,
                call_count=stats.call_count,
                duration_seconds=stats.duration_seconds,
            ))
        
        return stats_list
