"""PR summarizer module — produces a concise summary before scope identification."""

import logging

import dspy

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.memory.hippocampus import Hippocampus
from codespy.config import get_settings
from codespy.config_memory import get_memory_store

logger = logging.getLogger(__name__)


class PRSummarySignature(dspy.Signature):
    """Summarize what a merge request does in 2-3 sentences.

    You are a busy Principal Engineer. Be extremely terse. State facts only.
    Based on the title, description, and changed file paths, describe
    what this MR accomplishes. No polite filler. No conversational language.
    """

    mr_title: str = dspy.InputField(desc="Title of the merge request")
    mr_description: str = dspy.InputField(desc="Description/body of the MR")
    changed_file_paths: list[str] = dspy.InputField(
        desc="List of changed file paths from the MR"
    )

    summary: str = dspy.OutputField(
        desc="2-3 sentence summary of what this MR accomplishes"
    )


class Summarizer(dspy.Module):
    """Produces a concise PR summary used as Hippocampus question for all downstream modules."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    def forward(
        self,
        mr_title: str,
        mr_description: str,
        mr_number: int,
        changed_file_paths: list[str],
        repo_slug: str,
        run_id: str | None = None,
    ) -> str:
        """Generate a PR summary.

        Args:
            mr_title: Title of the merge request
            mr_description: Description/body of the MR
            mr_number: MR/PR number
            changed_file_paths: List of changed file paths
            repo_slug: Host-qualified repo slug for episode path
            run_id: Pipeline run identifier

        Returns:
            The summary string (2-3 sentences)
        """
        if not self._settings.is_signature_enabled("summary"):
            logger.debug("Skipping summary: disabled")
            return mr_title or "No title"

        summarizer = dspy.ChainOfThought(PRSummarySignature)
        logger.info("Generating PR summary...")

        question = f"summarize {repo_slug}: pull request {mr_number} {mr_title}"

        with SignatureContext("summary", self._cost_tracker):
            if self._settings.get_memory_enabled("summary"):
                mem = Hippocampus(
                    summarizer,
                    budget=self._settings.get_memory_budget("summary"),
                    max_reflects=self._settings.get_memory_max_reflects("summary"),
                    question=question,
                    task_name="summary",
                    run_id=run_id,
                )
                result = mem(
                    mr_title=mr_title,
                    mr_description=mr_description,
                    changed_file_paths=changed_file_paths,
                )
                mem.end_episode(
                    get_memory_store(self._settings),
                    f"/{repo_slug}/",
                    artifacts={"summary": result.summary},
                )
            else:
                result = summarizer(
                    mr_title=mr_title,
                    mr_description=mr_description,
                    changed_file_paths=changed_file_paths,
                )

        logger.info(f"PR summary: {result.summary[:80]}...")
        return result.summary
