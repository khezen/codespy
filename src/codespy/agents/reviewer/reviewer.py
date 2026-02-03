"""Main review pipeline that orchestrates all review modules."""

import logging

import dspy  # type: ignore[import-untyped]

from codespy.agents import configure_dspy, get_cost_tracker, verify_model_access
from codespy.config import Settings, get_settings
from codespy.tools.github.client import GitHubClient
from codespy.tools.github.models import ChangedFile
from codespy.agents.reviewer.models import Issue, ReviewResult
from codespy.agents.reviewer.modules import (
    BugDetector,
    DomainExpert,
    DocumentationReviewer,
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
        self.scope_identifier = ScopeIdentifier()
        self.security_auditor = SecurityAuditor()
        self.bug_detector = BugDetector()
        self.doc_reviewer = DocumentationReviewer()
        self.domain_expert = DomainExpert()

    def _verify_model_access(self) -> None:
        """Verify LLM model access."""
        logger.info("Verifying model access...")
        success, message = verify_model_access(self.settings)
        if not success:
            raise ValueError(f"Model access failed: {message}")
        logger.info(f"Model access: {message}")

    def _fetch_pr(self, pr_url: str):
        """Fetch PR data from GitHub."""
        logger.info("Fetching PR data from GitHub...")
        pr = self.github_client.fetch_pull_request(pr_url)
        logger.info(f"PR #{pr.number}: {pr.title} ({len(pr.changed_files)} files)")
        return pr

    def forward(self, pr_url: str, verify_model: bool = True) -> ReviewResult:
        """Run the complete review pipeline on a pull request."""
        self.cost_tracker.reset()
        logger.info(f"Starting review of {pr_url}")
        if verify_model:
            self._verify_model_access()
        pr = self._fetch_pr(pr_url)

        # ScopeIdentifier clones the repo and identifies scopes
        cache_dir = self.settings.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        repo_path = cache_dir / pr.repo_owner / pr.repo_name
        logger.info("Identifying code scopes...")
        scopes = self.scope_identifier.forward(pr, repo_path)
        for scope in scopes:
            logger.info(f"  Scope: {scope.subroot} ({scope.scope_type.value}) - {len(scope.changed_files)} files")
        # Run all review modules
        all_issues: list[Issue] = []
        logger.info("Running bug detection...")
        try:
            bugs = self.bug_detector.forward(pr.code_files)
            all_issues.extend(bugs)
        except Exception as e:
            logger.error(f"Bug detection failed: {e}")

        logger.info("Running security analysis...")
        try:
            security_issues = self.security_auditor.forward(pr.code_files)
            all_issues.extend(security_issues)
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")

        logger.info("Running documentation review...")
        try:
            doc_issues = self.doc_reviewer.forward(scopes, repo_path)
            all_issues.extend(doc_issues)
        except Exception as e:
            logger.error(f"Documentation review failed: {e}")

        logger.info("Running domain expert analysis...")
        try:
            domain_issues = self.domain_expert.forward(scopes, repo_path)
            all_issues.extend(domain_issues)
        except Exception as e:
            logger.error(f"Domain expert analysis failed: {e}")
        logger.info(f"Found {len(all_issues)} total issues")

        # Generate summary, quality assessment, and recommendation
        logger.info("Generating PR summary...")
        try:
            summarizer = dspy.ChainOfThought(PRSummarySignature)
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
        return ReviewResult(
            pr_number=pr.number,
            pr_title=pr.title,
            pr_url=pr.url,
            repo=pr.repo_full_name,
            model_used=self.settings.litellm_model,
            issues=all_issues,
            overall_summary=summary,
            quality_assessment=quality_assessment,
            recommendation=recommendation,
            total_cost=self.cost_tracker.total_cost,
            total_tokens=self.cost_tracker.total_tokens,
            llm_calls=self.cost_tracker.call_count,
        )
