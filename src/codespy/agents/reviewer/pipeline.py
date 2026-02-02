"""Main review pipeline that orchestrates all review modules."""

import json
import logging

import dspy  # type: ignore[import-untyped]

from codespy.agents import configure_dspy, get_cost_tracker, verify_model_access
from codespy.config import Settings, get_settings
from codespy.tools.github.client import GitHubClient
from codespy.tools.github.models import ReviewContext
from codespy.agents.reviewer.models import FileReview, Issue, ReviewResult
from codespy.agents.reviewer.modules import (
    BugDetector,
    ContextAnalyzer,
    DocumentationReviewer,
    SecurityAuditor,
)
from codespy.agents.reviewer.modules.helpers import build_context_string

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

class ReviewPipeline:
    """Orchestrates the code review process using DSPy modules."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the review pipeline.

        Args:
            settings: Application settings. Uses global settings if not provided.
        """
        self.settings = settings or get_settings()
        self.github_client = GitHubClient(self.settings)
        self.cost_tracker = get_cost_tracker()

        # Initialize DSPy with LiteLLM using shared configuration
        configure_dspy(self.settings)

        # Initialize review modules
        self.security_analyzer = SecurityAuditor()
        self.bug_detector = BugDetector()
        self.docs_reviewer = DocumentationReviewer()
        self.context_analyzer = ContextAnalyzer()

        # Summary generator
        self.summary_generator = dspy.ChainOfThought(PRSummarySignature)

    def run(self, pr_url: str, verify_model: bool = True) -> ReviewResult:
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

        # Create review context
        review_context = ReviewContext(
            pull_request=pr,
            related_files={},
            repository_structure=None,
            callers={},
        )

        # Review each code file
        file_reviews: list[FileReview] = []
        for i, changed_file in enumerate(pr.code_files, 1):
            logger.info(f"[{i}/{len(pr.code_files)}] Reviewing {changed_file.filename}...")
            file_review = self._review_file(changed_file, review_context)
            file_reviews.append(file_review)
            cost_info = f"(${self.cost_tracker.total_cost:.4f} total)" if self.cost_tracker.total_cost > 0 else ""
            logger.info(f"  Found {file_review.issue_count} issues {cost_info}")

        # Generate overall summary
        logger.info("Generating PR summary...")
        all_issues = []
        for fr in file_reviews:
            all_issues.extend(fr.issues)

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

    def _review_file(
        self,
        changed_file,
        review_context: ReviewContext,
    ) -> FileReview:
        """Review a single file with all modules.

        Args:
            changed_file: The file to review
            review_context: Full review context

        Returns:
            FileReview with all issues found
        """
        issues: list[Issue] = []

        # Skip files without patches (binary, too large, etc.)
        if not changed_file.patch:
            return FileReview(
                filename=changed_file.filename,
                reviewed=False,
                skip_reason="No diff available (binary or too large)",
            )

        # Get context for this specific file
        file_context = review_context.get_context_for_file(changed_file.filename)

        # Run security analysis
        try:
            security_issues = self.security_analyzer.forward(changed_file, file_context)
            issues.extend(security_issues)
            logger.debug(f"  Security: {len(security_issues)} issues")
        except Exception as e:
            logger.error(f"  Security analysis failed: {e}")

        # Run bug detection
        try:
            bug_issues = self.bug_detector.forward(changed_file, file_context)
            issues.extend(bug_issues)
            logger.debug(f"  Bugs: {len(bug_issues)} issues")
        except Exception as e:
            logger.error(f"  Bug detection failed: {e}")

        # Run documentation review
        try:
            doc_issues = self.docs_reviewer.forward(changed_file)
            issues.extend(doc_issues)
            logger.debug(f"  Documentation: {len(doc_issues)} issues")
        except Exception as e:
            logger.error(f"  Documentation review failed: {e}")

        # Run contextual analysis (if we have context)
        if review_context.related_files or review_context.repository_structure:
            try:
                related_files_str, repo_structure = build_context_string(
                    changed_file, review_context
                )
                context_issues = self.context_analyzer.forward(
                    changed_file, related_files_str, repo_structure
                )
                issues.extend(context_issues)
                logger.debug(f"  Context: {len(context_issues)} issues")
            except Exception as e:
                logger.error(f"  Contextual analysis failed: {e}")

        return FileReview(
            filename=changed_file.filename,
            issues=issues,
            summary=self._summarize_file_changes(changed_file),
        )

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
                        "file": i.file,
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