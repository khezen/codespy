"""Main review pipeline that orchestrates all review modules."""

import json
import logging
from collections import defaultdict

import dspy  # type: ignore[import-untyped]

from codespy.agents import configure_dspy, get_cost_tracker, verify_model_access
from codespy.config import Settings, get_settings
from codespy.tools.github.client import GitHubClient
from codespy.agents.reviewer.models import FileReview, Issue, ReviewResult
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
    changed_files: str = dspy.InputField(
        desc="List of changed files with change counts"
    )
    all_issues: str = dspy.InputField(
        desc="JSON array of all issues found during review"
    )

    summary: str = dspy.OutputField(
        desc="2-3 sentence summary of what this PR accomplishes"
    )
    recommendation: str = dspy.OutputField(
        desc="One of: APPROVE, REQUEST_CHANGES, or NEEDS_DISCUSSION with brief justification"
    )

class ReviewPipeline(dspy.Module):
    """Orchestrates the code review process using DSPy modules."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the review pipeline.

        Args:
            settings: Application settings. Uses global settings if not provided.
        """
        super().__init__()
        self.settings = settings or get_settings()
        self.github_client = GitHubClient(self.settings)
        self.cost_tracker = get_cost_tracker()

        # Initialize DSPy with LiteLLM using shared configuration
        configure_dspy(self.settings)

        # Initialize review modules
        self.scope_identifier = ScopeIdentifier()
        self.security_analyzer = SecurityAuditor()
        self.bug_detector = BugDetector()
        self.docs_reviewer = DocumentationReviewer()
        self.domain_expert = DomainExpert()

        # Summary generator
        self.summary_generator = dspy.ChainOfThought(PRSummarySignature)

    def forward(self, pr_url: str, verify_model: bool = True) -> ReviewResult:
        """Run the complete review pipeline on a pull request.

        Args:
            pr_url: GitHub PR URL to review
            verify_model: Whether to verify model access before starting

        Returns:
            ReviewResult with all findings
        """
        # Reset cost tracker for this run
        self.cost_tracker.reset()

        logger.info(f"Starting review of {pr_url}")

        # Verify model access if requested
        if verify_model:
            logger.info("Verifying model access...")
            success, message = verify_model_access(self.settings)
            if success:
                logger.info(f"Model access: {message}")
            else:
                raise ValueError(f"Model access failed: {message}")

        # Fetch PR data
        logger.info("Fetching PR data from GitHub...")
        pr = self.github_client.fetch_pull_request(pr_url)
        if pr.excluded_files_count > 0:
            logger.info(
                f"PR #{pr.number}: {pr.title} ({len(pr.changed_files)} files to review, "
                f"{pr.excluded_files_count} vendor files excluded)"
            )
        else:
            logger.info(f"PR #{pr.number}: {pr.title} ({len(pr.changed_files)} files changed)")

        # Determine target path for repository (ScopeIdentifier agent will clone it)
        cache_dir = self.settings.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        repo_path = cache_dir / pr.repo_owner / pr.repo_name
        logger.info(f"Repository target path: {repo_path}")

        # Identify scopes in the repository (agent clones repo efficiently with sparse checkout)
        logger.info("Identifying code scopes...")
        scopes = self.scope_identifier.forward(pr, repo_path)
        for scope in scopes:
            manifest_info = ""
            if scope.package_manifest:
                manifest_info = f" [{scope.package_manifest.package_manager}]"
                if scope.package_manifest.dependencies_changed:
                    manifest_info += " (deps changed)"
            logger.info(
                f"  Scope: {scope.subroot} ({scope.scope_type.value}){manifest_info} - "
                f"{len(scope.changed_files)} files"
            )

        # Run review modules on file lists
        all_issues: list[Issue] = []

        # Security analysis on code files
        logger.info("Running security analysis...")
        try:
            security_issues = self.security_analyzer.forward(pr.code_files)
            all_issues.extend(security_issues)
            logger.info(f"  Security: {len(security_issues)} issues")
        except Exception as e:
            logger.error(f"  Security analysis failed: {e}")

        # Bug detection on code files
        logger.info("Running bug detection...")
        try:
            bug_issues = self.bug_detector.forward(pr.code_files)
            all_issues.extend(bug_issues)
            logger.info(f"  Bugs: {len(bug_issues)} issues")
        except Exception as e:
            logger.error(f"  Bug detection failed: {e}")

        # Documentation review on all scopes
        logger.info("Running documentation review...")
        try:
            doc_issues = self.docs_reviewer.forward(scopes, repo_path)
            all_issues.extend(doc_issues)
            logger.info(f"  Documentation: {len(doc_issues)} issues")
        except Exception as e:
            logger.error(f"  Documentation review failed: {e}")

        # Cross-file contextual analysis
        logger.info("Running domain expert analysis...")
        try:
            context_issues = self.domain_expert.forward(pr.code_files)
            all_issues.extend(context_issues)
            logger.info(f"  Context: {len(context_issues)} issues")
        except Exception as e:
            logger.error(f"  Contextual analysis failed: {e}")

        cost_info = f"(${self.cost_tracker.total_cost:.4f} total)" if self.cost_tracker.total_cost > 0 else ""
        logger.info(f"Found {len(all_issues)} total issues {cost_info}")

        # Group issues by filename into FileReview objects
        file_reviews = self._group_issues_by_file(pr, all_issues)

        # Generate overall summary
        logger.info("Generating PR summary...")

        summary, recommendation = self._generate_summary(pr, all_issues)

        result = ReviewResult(
            pr_number=pr.number,
            pr_title=pr.title,
            pr_url=pr.url,
            repo=pr.repo_full_name,
            model_used=self.settings.litellm_model,
            file_reviews=file_reviews,
            overall_summary=summary,
            recommendation=recommendation,
            total_cost=self.cost_tracker.total_cost,
            total_tokens=self.cost_tracker.total_tokens,
            llm_calls=self.cost_tracker.call_count,
        )

        cost_str = f", Cost: ${result.total_cost:.4f}" if result.total_cost > 0 else ""
        logger.info(f"Review complete. Total issues: {result.total_issues}, LLM calls: {result.llm_calls}{cost_str}")
        return result

    def _group_issues_by_file(self, pr, all_issues: list[Issue]) -> list[FileReview]:
        """Group issues by filename into FileReview objects.

        Args:
            pr: The pull request
            all_issues: All issues from all modules

        Returns:
            List of FileReview objects, one per changed file
        """
        # Group issues by filename
        issues_by_file: dict[str, list[Issue]] = defaultdict(list)
        for issue in all_issues:
            issues_by_file[issue.filename].append(issue)

        # Create FileReview for each changed file
        file_reviews: list[FileReview] = []
        for changed_file in pr.changed_files:
            filename = changed_file.filename
            if not changed_file.patch:
                file_reviews.append(FileReview(
                    filename=filename,
                    reviewed=False,
                    skip_reason="No diff available (binary or too large)",
                ))
            else:
                file_reviews.append(FileReview(
                    filename=filename,
                    issues=issues_by_file.get(filename, []),
                    summary=self._summarize_file_changes(changed_file),
                ))

        return file_reviews

    def _summarize_file_changes(self, changed_file) -> str:
        """Generate a brief summary of file changes."""
        status = changed_file.status.value
        additions = changed_file.additions
        deletions = changed_file.deletions
        return f"{status.title()}: +{additions}/-{deletions} lines"

    def _generate_summary(
        self,
        pr,
        all_issues: list[Issue],
    ) -> tuple[str, str]:
        """Generate overall PR summary and recommendation.

        Args:
            pr: The pull request
            all_issues: All issues found across all files

        Returns:
            Tuple of (summary, recommendation)
        """
        try:
            # Prepare changed files summary
            changed_files_str = "\n".join(
                f"- {f.filename} ({f.status.value}): +{f.additions}/-{f.deletions}"
                for f in pr.changed_files
            )

            # Prepare issues summary
            issues_summary = json.dumps(
                [
                    {
                        "category": i.category.value,
                        "severity": i.severity.value,
                        "title": i.title,
                        "file": i.filename,
                    }
                    for i in all_issues
                ],
                indent=2,
            )

            result = self.summary_generator(
                pr_title=pr.title,
                pr_description=pr.body or "No description provided.",
                changed_files=changed_files_str,
                all_issues=issues_summary,
            )

            return result.summary, result.recommendation

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")

            # Fallback summary
            critical_count = sum(1 for i in all_issues if i.severity.value == "critical")
            high_count = sum(1 for i in all_issues if i.severity.value == "high")

            summary = f"Reviewed {len(pr.changed_files)} files with {len(all_issues)} total issues."

            if critical_count > 0:
                recommendation = f"REQUEST_CHANGES: Found {critical_count} critical issues that must be addressed."
            elif high_count > 0:
                recommendation = f"REQUEST_CHANGES: Found {high_count} high-severity issues to review."
            elif len(all_issues) > 5:
                recommendation = "NEEDS_DISCUSSION: Multiple issues found that should be discussed."
            else:
                recommendation = "APPROVE: No critical issues found."

            return summary, recommendation