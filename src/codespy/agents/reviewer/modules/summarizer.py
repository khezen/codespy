"""PR summarizer module — produces a concise summary before scope identification."""

import logging
from typing import TYPE_CHECKING

import dspy

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.context_safe import ContextSafe
from codespy.agents.memory.hippocampus import ContextMemory, Hippocampus
from codespy.agents.memory.hippocampus.episode import submit_episode_save
from codespy.agents.memory.hippocampus.context_memory import Topic
from codespy.agents.reviewer.modules.scope_resolver import _deepest_common_folder
from codespy.config import get_settings
from codespy.config_memory import get_episode_store

if TYPE_CHECKING:
    from codespy.agents.reviewer.models import PRContext, ScopeResult

logger = logging.getLogger(__name__)


class PRSummarySignature(dspy.Signature):
    """Summarize what a pull request does in 2-3 sentences.

    You are a busy Principal Engineer. Be extremely terse. State facts only.
    Based on the title, description, changed file paths, and code patches,
    describe what this PR accomplishes. No polite filler. No conversational language.
    """

    pr_title: str = dspy.InputField(desc="Title of the pull request")
    pr_description: str = dspy.InputField(desc="Description/body of the PR")
    changed_file_paths: list[str] = dspy.InputField(desc="List of changed file paths from the PR")
    patches: str = dspy.InputField(
        desc="Unified diff patches showing code changes. Each patch is prefixed with the filename."
    )

    summary: str = dspy.OutputField(desc="2-3 sentence summary of what this PR accomplishes")


class Summarizer(dspy.Module):
    """Produces a concise PR summary used as Hippocampus question for all downstream modules."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    def forward(
        self,
        pr_context: "PRContext",
        changed_file_paths: list[str],
        patches: str,
        run_id: str | None = None,
        scopes: list["ScopeResult"] | None = None,
        topics: list[Topic] | None = None,
    ) -> str:
        """Generate a PR summary.

        Args:
            pr_context: PRContext with PR identity (repo_slug, pr_number, pr_title, pr_description, pr_url)
            changed_file_paths: List of changed file paths
            patches: Unified diff patches showing code changes
            run_id: Pipeline run identifier
            scopes: List of resolved scopes for per-scope episode persistence
            topics: Optional list of Topic objects for auto-tagging

        Returns:
            Summary string
        """

        if not self._settings.is_signature_enabled("summary"):
            logger.debug("Skipping summary: disabled")
            return pr_context.pr_title or "No title"

        # Load latest "summary" episode for the given topics
        initial_memory: ContextMemory | None = None
        store = None
        topic_ids: list[str] | None = None
        if self._settings.get_memory_enabled("summary") and scopes:
            store = get_episode_store(self._settings)
            if store is not None:
                # Build topic_ids from scope topics
                topic_ids = []
                for scope in scopes:
                    if scope.topic(pr_context.repo_full_name):
                        topic_ids.append(scope.topic(pr_context.repo_full_name).id)
                initial_memory = store.load_context(
                    task="summary",
                    topic_ids=topic_ids if topic_ids else None,
                )
                if initial_memory:
                    logger.info("Loaded prior summary episode(s) into summarizer memory")
                else:
                    logger.info("No prior summary episode found")

        summarizer = ContextSafe(
            dspy.ChainOfThought(PRSummarySignature),
            PRSummarySignature,
            name="summary",
            max_iters=self._settings.get_max_iters("summary"),
            max_llm_calls=self._settings.get_max_llm_calls("summary"),
            rlm_threshold=self._settings.get_rlm_threshold("chain_of_thought"),
        )
        logger.info("Generating PR summary...")

        question = f"summarize {pr_context.repo_slug}: pull request {pr_context.pr_number} {pr_context.pr_title}"

        mem: Hippocampus | None = None
        with SignatureContext("summary", self._cost_tracker):
            if self._settings.get_memory_enabled("summary") and store is not None:
                # Build topics list for Hippocampus
                scope_topics: list[Topic] = []
                for scope in scopes or []:
                    scope_topic = scope.topic(pr_context.repo_full_name)
                    if scope_topic:
                        scope_topics.append(scope_topic)

                mem = Hippocampus(
                    summarizer,
                    budget=self._settings.get_memory_budget("summary"),
                    max_reflects=self._settings.get_memory_max_reflects("summary"),
                    question=question,
                    task_name="summary",
                    run_id=run_id,
                    initial_memory=initial_memory,
                    topics=scope_topics if scope_topics else topics,
                )
                result = mem(
                    pr_title=pr_context.pr_title,
                    pr_description=pr_context.pr_description,
                    changed_file_paths=changed_file_paths,
                    patches=patches,
                )
                # Fire-and-forget episode save
                _summary_text = result.summary
                def _persist():
                    try:
                        mem.end_episode(store, artifacts={"summary": _summary_text})
                    except Exception:
                        logger.warning("Background summary episode save failed", exc_info=True)
                submit_episode_save(_persist, name="summary-episode-save")
            else:
                result = summarizer(
                    pr_title=pr_context.pr_title,
                    pr_description=pr_context.pr_description,
                    changed_file_paths=changed_file_paths,
                    patches=patches,
                )

        logger.info(f"PR summary: {result.summary[:80]}...")
        return result.summary
