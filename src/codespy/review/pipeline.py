"""Main review pipeline that orchestrates all review modules."""

import json
import logging
import threading
from typing import Optional

import dspy
import litellm

from codespy.config import Settings, get_settings
from codespy.github.client import GitHubClient
from codespy.github.models import ReviewContext
from codespy.review.models import FileReview, Issue, ReviewResult
from codespy.review.modules import (
    BugDetector,
    ContextAnalyzer,
    DocumentationReviewer,
    SecurityAnalyzer,
)
from codespy.review.signatures import PRSummary

logger = logging.getLogger(__name__)


class CostTracker:
    """Track LLM costs across multiple calls."""

    def __init__(self) -> None:
        """Initialize the cost tracker."""
        self._lock = threading.Lock()
        self._total_cost = 0.0
        self._total_tokens = 0
        self._call_count = 0
        self._last_call_cost = 0.0
        self._last_call_tokens = 0

    def reset(self) -> None:
        """Reset all tracking."""
        with self._lock:
            self._total_cost = 0.0
            self._total_tokens = 0
            self._call_count = 0
            self._last_call_cost = 0.0
            self._last_call_tokens = 0

    def add_call(self, cost: float, tokens: int) -> None:
        """Record a call's cost and tokens."""
        with self._lock:
            self._total_cost += cost
            self._total_tokens += tokens
            self._call_count += 1
            self._last_call_cost = cost
            self._last_call_tokens = tokens

    @property
    def total_cost(self) -> float:
        """Get total cost in USD."""
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self._total_tokens

    @property
    def call_count(self) -> int:
        """Get number of calls made."""
        return self._call_count

    @property
    def last_call_cost(self) -> float:
        """Get cost of last call."""
        return self._last_call_cost

    @property
    def last_call_tokens(self) -> int:
        """Get tokens of last call."""
        return self._last_call_tokens


# Global cost tracker instance
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    return _cost_tracker


def _litellm_success_callback(kwargs, completion_response, start_time, end_time):
    """Callback for successful LiteLLM calls to track costs."""
    try:
        # Get cost from response (LiteLLM provides this)
        cost = kwargs.get("response_cost", 0.0)
        if cost == 0.0 and hasattr(completion_response, "_hidden_params"):
            cost = getattr(completion_response._hidden_params, "response_cost", 0.0) or 0.0

        # Calculate tokens
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(completion_response, "usage") and completion_response.usage:
            prompt_tokens = getattr(completion_response.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(completion_response.usage, "completion_tokens", 0) or 0

        total_tokens = prompt_tokens + completion_tokens

        # Try to get cost from litellm's cost calculator if not available
        if cost == 0.0 and total_tokens > 0:
            try:
                model = kwargs.get("model", "")
                cost = litellm.completion_cost(
                    model=model,
                    prompt=str(prompt_tokens),
                    completion=str(completion_tokens),
                )
            except Exception:
                pass

        _cost_tracker.add_call(cost, total_tokens)

        logger.debug(
            f"LLM call: {total_tokens} tokens, ${cost:.4f} "
            f"(total: {_cost_tracker.call_count} calls, ${_cost_tracker.total_cost:.4f})"
        )
    except Exception as e:
        logger.debug(f"Cost tracking error: {e}")


class ReviewPipeline:
    """Orchestrates the code review process using DSPy modules."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the review pipeline.

        Args:
            settings: Application settings. Uses global settings if not provided.
        """
        self.settings = settings or get_settings()
        self.github_client = GitHubClient(self.settings)
        self.cost_tracker = get_cost_tracker()

        # Initialize DSPy with LiteLLM
        self._configure_dspy()

        # Initialize review modules
        self.security_analyzer = SecurityAnalyzer()
        self.bug_detector = BugDetector()
        self.docs_reviewer = DocumentationReviewer()
        self.context_analyzer = ContextAnalyzer()

        # Summary generator
        self.summary_generator = dspy.ChainOfThought(PRSummary)

    def _configure_dspy(self) -> None:
        """Configure DSPy with the LLM backend."""
        model = self.settings.litellm_model

        # Configure LiteLLM environment if needed
        if self.settings.openai_api_key:
            litellm.openai_key = self.settings.openai_api_key
        if self.settings.anthropic_api_key:
            litellm.anthropic_key = self.settings.anthropic_api_key

        # Set up AWS credentials for Bedrock if using Bedrock model
        if model.startswith("bedrock/"):
            import os

            os.environ["AWS_REGION_NAME"] = self.settings.aws_region
            if self.settings.aws_access_key_id:
                os.environ["AWS_ACCESS_KEY_ID"] = self.settings.aws_access_key_id
            if self.settings.aws_secret_access_key:
                os.environ["AWS_SECRET_ACCESS_KEY"] = self.settings.aws_secret_access_key

        # Register cost tracking callback
        if _litellm_success_callback not in litellm.success_callback:
            litellm.success_callback.append(_litellm_success_callback)

        # Configure DSPy with LiteLLM
        lm = dspy.LM(model=model)
        dspy.configure(lm=lm)

        logger.info(f"Configured DSPy with model: {model}")

    def verify_model_access(self) -> tuple[bool, str]:
        """Verify that the model is accessible.

        Returns:
            Tuple of (success, message)
        """
        model = self.settings.litellm_model
        try:
            # Make a minimal test call
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return True, "Model access verified"
        except litellm.AuthenticationError as e:
            return False, f"Authentication failed: {e}"
        except litellm.RateLimitError as e:
            return False, f"Rate limit exceeded: {e}"
        except litellm.APIConnectionError as e:
            return False, f"Connection error: {e}"
        except Exception as e:
            return False, f"Model access error: {e}"

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
            success, message = self.verify_model_access()
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

        # Build review context (includes related files)
        logger.info("Building review context...")
        review_context = self.github_client.build_review_context(
            pr, include_repo_context=self.settings.include_repo_context
        )
        logger.info(f"Found {len(review_context.related_files)} related files for context")

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
            security_issues = self.security_analyzer.analyze(changed_file, file_context)
            issues.extend(security_issues)
            logger.debug(f"  Security: {len(security_issues)} issues")
        except Exception as e:
            logger.error(f"  Security analysis failed: {e}")

        # Run bug detection
        try:
            bug_issues = self.bug_detector.analyze(changed_file, file_context)
            issues.extend(bug_issues)
            logger.debug(f"  Bugs: {len(bug_issues)} issues")
        except Exception as e:
            logger.error(f"  Bug detection failed: {e}")

        # Run documentation review
        try:
            doc_issues = self.docs_reviewer.analyze(changed_file)
            issues.extend(doc_issues)
            logger.debug(f"  Documentation: {len(doc_issues)} issues")
        except Exception as e:
            logger.error(f"  Documentation review failed: {e}")

        # Run contextual analysis (if we have context)
        if review_context.related_files or review_context.repository_structure:
            try:
                context_issues = self.context_analyzer.analyze_with_context(
                    changed_file, review_context
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