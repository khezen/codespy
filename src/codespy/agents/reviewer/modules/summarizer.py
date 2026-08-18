"""PR summarizer module — produces a concise summary before scope identification."""

import logging
from typing import TYPE_CHECKING

import dspy

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.context_safe import ContextSafe
from codespy.agents.memory.hippocampus import ContextMemory, Hippocampus
from codespy.agents.reviewer.modules.scope_resolver import _deepest_common_folder
from codespy.config import get_settings
from codespy.config_memory import get_memory_store

if TYPE_CHECKING:
    from codespy.agents.reviewer.models import ScopeResult

logger = logging.getLogger(__name__)


class PRSummarySignature(dspy.Signature):
    """Summarize what a pull request does in 2-3 sentences.

    You are a busy Principal Engineer. Be extremely terse. State facts only.
    Based on the title, description, changed file paths, and code patches,
    describe what this PR accomplishes. No polite filler. No conversational language.
    """

    pr_title: str = dspy.InputField(desc="Title of the pull request")
    pr_description: str = dspy.InputField(desc="Description/body of the PR")
    changed_file_paths: list[str] = dspy.InputField(
        desc="List of changed file paths from the PR"
    )
    patches: str = dspy.InputField(
        desc="Unified diff patches showing code changes. Each patch is prefixed with the filename."
    )

    summary: str = dspy.OutputField(
        desc="2-3 sentence summary of what this PR accomplishes"
    )


class Summarizer(dspy.Module):
    """Produces a concise PR summary used as Hippocampus question for all downstream modules."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    def forward(
        self,
        pr_title: str,
        pr_description: str,
        pr_number: int,
        changed_file_paths: list[str],
        patches: str,
        repo_slug: str,
        run_id: str | None = None,
        scopes: list["ScopeResult"] | None = None,
        initial_memory: ContextMemory | None = None,
        topic_ids: list[str] | None = None,
    ) -> tuple[str, ContextMemory | None]:
        """Generate a PR summary.

        Args:
            pr_title: Title of the pull request
            pr_description: Description/body of the PR
            pr_number: PR number
            changed_file_paths: List of changed file paths
            patches: Unified diff patches showing code changes
            repo_slug: Host-qualified repo slug for episode path
            run_id: Pipeline run identifier
            scopes: List of resolved scopes for per-scope episode persistence
            initial_memory: Optional context memory from scope resolver
            topic_ids: Optional list of topic IDs for auto-tagging

        Returns:
            Tuple of (summary string, final context memory or None)
        """

        if not self._settings.is_signature_enabled("summary"):
            logger.debug("Skipping summary: disabled")
            return pr_title or "No title", initial_memory
        # Load latest episode per scope and merge with inherited memory
        if self._settings.get_memory_enabled("summary") and scopes:
            from codespy.agents.memory.hippocampus.episode import find_latest_episode
            store = get_memory_store(self._settings)
            per_scope_memories: list[ContextMemory] = []
            for scope in scopes:
                ep = find_latest_episode(store, scope.scope_path(), task=None, exclude_task="scope", exclude_run_id=run_id)
                if ep is not None:
                    per_scope_memories.append(ep.context_memory)
            if per_scope_memories:
                all_memories = ([initial_memory] if initial_memory else []) + per_scope_memories
                initial_memory = ContextMemory.merge(*all_memories)
                logger.info(
                    "Merged %d prior scope episode(s) into summarizer memory",
                    len(per_scope_memories),
                )
        summarizer = ContextSafe(
            dspy.ChainOfThought(PRSummarySignature),
            PRSummarySignature,
            name="summary",
            max_llm_calls=self._settings.get_max_llm_calls("summary"),
        )
        logger.info("Generating PR summary...")

        question = f"summarize {repo_slug}: pull request {pr_number} {pr_title}"

        mem: Hippocampus | None = None
        with SignatureContext("summary", self._cost_tracker):
            if self._settings.get_memory_enabled("summary"):
                mem = Hippocampus(
                    summarizer,
                    budget=self._settings.get_memory_budget("summary"),
                    max_reflects=self._settings.get_memory_max_reflects("summary"),
                    question=question,
                    task_name="summary",
                    run_id=run_id,
                    initial_memory=initial_memory,
                    topic_ids=topic_ids,
                )
                result = mem(
                    pr_title=pr_title,
                    pr_description=pr_description,
                    changed_file_paths=changed_file_paths,
                    patches=patches,
                )
                mem.end_episode(
                    get_memory_store(self._settings),
                    _deepest_common_folder(scopes, repo_slug) if scopes else f"/{repo_slug}/",
                    artifacts={"summary": result.summary},
                )
            else:
                result = summarizer(
                    pr_title=pr_title,
                    pr_description=pr_description,
                    changed_file_paths=changed_file_paths,
                    patches=patches,
                )

        logger.info(f"PR summary: {result.summary[:80]}...")
        # Return final context memory when memory is enabled
        final_memory = mem.cmem.model_copy(deep=True) if mem else initial_memory

        # Persist episode at each scope location if memory is enabled and scopes are provided
        if mem is not None and scopes:
            store = get_memory_store(self._settings)
            for scope in scopes:
                path = mem.episode_file_path(scope.scope_path())
                mem.save_episode(store, path)

        return result.summary, final_memory
