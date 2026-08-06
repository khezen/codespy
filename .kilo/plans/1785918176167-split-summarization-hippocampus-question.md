# Plan: Split Summarization & Replace question_field with question

## Goal

1. Split `MRSummarySignature` into **Summarizer** (runs before scope identification) and **Auditor** (runs after reviews).
2. Replace `Hippocampus.question_field` with a `question: str | None` parameter that directly accepts a pre-computed string.
3. Each module constructs a task-specific question string for its Hippocampus episodes.

## Pipeline Flow (Before → After)

**Before:**
```
Fetch MR → Scope ID → Reviews (code, doc, supply_chain) → Summarization (summary + assessment + recommendation)
```

**After:**
```
Fetch MR → Summary → Scope ID → Reviews (code, doc, supply_chain) → Audit (assessment + recommendation)
```

## Question Formats Per Module

| Module | Question Template |
|--------|------------------|
| Summary | `"summarize {repo_slug}: pull request {mr_number} {mr_title}"` |
| Scope | `"identify scopes of {repo_slug}: pull request {mr_number} {mr_title}: {summary}"` |
| Code Review | `"review code change of {repo_slug}: {scope.subroot}: pull request {mr_number} {mr_title}: {summary}"` |
| Doc | `"review documentation of {repo_slug}: {scope.subroot}: pull request {mr_number} {mr_title}: {summary}"` |
| Supply Chain | `"review supply chain of {repo_slug}: {scope.subroot}: pull request {mr_number} {mr_title}: {summary}"` |
| Audit | `"final audit of {repo_slug}: pull request {mr_number} {mr_title}: {summary}"` |

**Data availability:**
- `repo_slug`: available from `mr.repo_slug` (orchestrator) or `scope.repo` (per-scope modules)
- `mr_number`: from `mr.number` — must be passed to per-scope modules
- `mr_title`: from `mr.title` — must be passed to per-scope modules
- `scope.subroot`: available inside per-scope iteration loops
- `summary`: produced by Summarizer, passed downstream as `pr_summary`

---

## Tasks

### 1. Hippocampus: Replace `question_field` with `question`

**File:** `src/codespy/agents/memory/hippocampus/hippocampus.py`

- Remove `question_field: str | None = None` from `__init__()` (line 110)
- Add `question: str | None = None` in its place
- Store as `self.question = question` (replaces `self.question_field` at line 169)
- Update `_make_question()` (lines 413-416):
  ```python
  def _make_question(self, inputs: dict) -> str:
      if self.question is not None:
          return self.question
      return format_inputs(inputs, self.budget.max_question_tokens)
  ```
- Update docstrings referencing `question_field` throughout the file

### 2. Update docstrings referencing `question_field` elsewhere

**Files:**
- `src/codespy/agents/memory/hippocampus/budget.py` — lines 54, 58
- `src/codespy/agents/memory/hippocampus/episode.py` — line 28
- `src/codespy/config_memory.py` — line 106
- `src/codespy/config.py` — line 310

Replace references to `question_field` with `question`.

### 3. Create Summarizer module

**New file:** `src/codespy/agents/reviewer/modules/summarizer.py`

Follows the same pattern as existing modules (owns its signature, Hippocampus wrapping, config access):

```python
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
                result = mem.forward(
                    mr_title=mr_title,
                    mr_description=mr_description,
                    changed_file_paths=changed_file_paths,
                )
                mem.end_episode(
                    get_memory_store(self._settings),
                    f"/{repo_slug}/root/",
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
```

### 4. Create Auditor module

**New file:** `src/codespy/agents/reviewer/modules/auditor.py`

```python
"""Auditor module — assesses code quality and provides recommendation after reviews."""

import logging
from typing import Sequence

import dspy

from codespy.agents import SignatureContext, get_cost_tracker
from codespy.agents.memory.hippocampus import Hippocampus
from codespy.agents.reviewer.models import Issue
from codespy.config import get_settings
from codespy.config_memory import get_memory_store
from codespy.tools.git.models import ChangedFile

logger = logging.getLogger(__name__)


class AuditSignature(dspy.Signature):
    """Assess code quality and provide a recommendation for a merge request.

    You are a busy Principal Engineer. Be extremely terse. State facts only.
    Based on the summary, changed files, and issues found during review, provide:
    - An overall assessment of the code quality
    - A recommendation (approve, request changes, or needs discussion)

    No polite filler. No conversational language.
    """

    mr_title: str = dspy.InputField(desc="Title of the merge request")
    summary: str = dspy.InputField(desc="Summary of what this MR accomplishes")
    changed_files: list[ChangedFile] = dspy.InputField(
        desc="In-scope reviewable files with status and line counts"
    )
    all_issues: list[Issue] = dspy.InputField(
        desc="All issues found during review"
    )

    quality_assessment: str = dspy.OutputField(
        desc="Overall assessment of code quality"
    )
    recommendation: str = dspy.OutputField(
        desc="One of: APPROVE, REQUEST_CHANGES, or NEEDS_DISCUSSION with brief justification"
    )


class Auditor(dspy.Module):
    """Assesses code quality and recommends action after all reviews complete."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_tracker = get_cost_tracker()
        self._settings = get_settings()

    def forward(
        self,
        mr_title: str,
        mr_number: int,
        pr_summary: str,
        changed_files: Sequence[ChangedFile],
        all_issues: Sequence[Issue],
        repo_slug: str,
        run_id: str | None = None,
    ) -> tuple[str, str]:
        """Assess quality and recommend action.

        Args:
            mr_title: Title of the merge request
            mr_number: MR/PR number
            pr_summary: Summary produced by the Summarizer
            changed_files: In-scope reviewable files
            all_issues: All issues found during review
            repo_slug: Host-qualified repo slug for episode path
            run_id: Pipeline run identifier

        Returns:
            Tuple of (quality_assessment, recommendation)
        """
        if not self._settings.is_signature_enabled("audit"):
            logger.debug("Skipping audit: disabled")
            return (
                "Audit disabled.",
                "NEEDS_DISCUSSION" if all_issues else "APPROVE",
            )

        auditor = dspy.ChainOfThought(AuditSignature)
        logger.info("Running audit...")

        question = f"final audit of {repo_slug}: pull request {mr_number} {mr_title}: {pr_summary}"

        with SignatureContext("audit", self._cost_tracker):
            if self._settings.get_memory_enabled("audit"):
                mem = Hippocampus(
                    auditor,
                    budget=self._settings.get_memory_budget("audit"),
                    max_reflects=self._settings.get_memory_max_reflects("audit"),
                    question=question,
                    task_name="audit",
                    run_id=run_id,
                )
                result = mem.forward(
                    mr_title=mr_title,
                    summary=pr_summary,
                    changed_files=list(changed_files),
                    all_issues=list(all_issues),
                )
                mem.end_episode(
                    get_memory_store(self._settings),
                    f"/{repo_slug}/root/",
                    artifacts={
                        "audit": (
                            f"## Quality Assessment\n\n{result.quality_assessment}\n\n"
                            f"## Recommendation\n\n{result.recommendation}\n"
                        )
                    },
                )
            else:
                result = auditor(
                    mr_title=mr_title,
                    summary=pr_summary,
                    changed_files=list(changed_files),
                    all_issues=list(all_issues),
                )

        return result.quality_assessment, result.recommendation
```

### 5. Export new modules

**File:** `src/codespy/agents/reviewer/modules/__init__.py`

Add `Summarizer` and `Auditor` to imports and `__all__`.

### 6. Remove `MRSummarySignature` and update `ReviewPipeline`

**File:** `src/codespy/agents/reviewer/reviewer.py`

- Delete the `MRSummarySignature` class (lines 34-65)
- Remove its related imports (no longer needs `Hippocampus`, `get_memory_store` in this file)
- Add imports: `from codespy.agents.reviewer.modules import Summarizer, Auditor`
- Add `self.summarizer = Summarizer()` and `self.auditor = Auditor()` in `__init__()`
- Restructure `forward()`:

```python
# After fetching/building MR, BEFORE scope identification:

# 1. Run Summarizer
changed_file_paths = [f.filename for f in mr.changed_files]
pr_summary = self.summarizer(
    mr_title=mr.title,
    mr_description=mr.body or "No description provided.",
    mr_number=mr.number,
    changed_file_paths=changed_file_paths,
    repo_slug=mr.repo_slug,
    run_id=run_id,
)

# 2. Scope identification (pass pr_summary)
scopes = self.scope_identifier(mr, repo_path, is_local=is_local, run_id=run_id, pr_summary=pr_summary)

# 3. Reviews (pass pr_summary, mr_number, mr_title)
all_issues = asyncio.run(
    self._run_review_modules(
        scopes, repo_path, module_names,
        run_id=run_id, pr_summary=pr_summary,
        mr_number=mr.number, mr_title=mr.title,
    )
)

# 4. Audit
scoped_files = self._collect_scoped_files(scopes)
quality_assessment, recommendation = self.auditor(
    mr_title=mr.title,
    mr_number=mr.number,
    pr_summary=pr_summary,
    changed_files=scoped_files,
    all_issues=all_issues,
    repo_slug=mr.repo_slug,
    run_id=run_id,
)

# Build ReviewResult with overall_summary=pr_summary, quality_assessment, recommendation
```

- Remove the entire old summarization block (lines 231-285)

### 7. Update `_run_review_modules`

**File:** `src/codespy/agents/reviewer/reviewer.py`

Add `pr_summary: str`, `mr_number: int`, `mr_title: str` parameters and pass to each module:

```python
async def _run_review_modules(
    self, ..., pr_summary: str, mr_number: int, mr_title: str
) -> list[Issue]:
    tasks = [
        self.code_reviewer.aforward(
            scopes=scopes, repo_path=repo_path, run_id=run_id,
            pr_summary=pr_summary, mr_number=mr_number, mr_title=mr_title,
        ),
        self.doc_reviewer.aforward(
            scopes=scopes, repo_path=repo_path, run_id=run_id,
            pr_summary=pr_summary, mr_number=mr_number, mr_title=mr_title,
        ),
        self.supply_chain_auditor.aforward(
            scopes=scopes, repo_path=repo_path, run_id=run_id,
            pr_summary=pr_summary, mr_number=mr_number, mr_title=mr_title,
        ),
    ]
    ...
```

### 8. Update Scope Identifier

**File:** `src/codespy/agents/reviewer/modules/scope_identifier.py`

- Add `pr_summary: str | None = None` to `aforward()` and `forward()` signatures
- Construct question inside the memory-enabled block:
  ```python
  question = f"identify scopes of {mr.repo_slug}: pull request {mr.number} {mr.title}: {pr_summary}"
  mem = Hippocampus(
      agent,
      budget=self._settings.get_memory_budget("scope"),
      max_reflects=self._settings.get_memory_max_reflects("scope"),
      question=question,
      task_name="scope",
      run_id=run_id,
  )
  ```
- Remove the old `question_field="mr_title"` and its comments (lines 310-312)

### 9. Update Code Reviewer

**File:** `src/codespy/agents/reviewer/modules/code_reviewer.py`

- Add `pr_summary: str | None = None`, `mr_number: int | None = None`, `mr_title: str | None = None` to `aforward()` and `forward()` signatures
- Construct per-scope question inside the scope loop:
  ```python
  question = (
      f"review code change of {scope.repo}: {scope.subroot}: "
      f"pull request {mr_number} {mr_title}: {pr_summary}"
  ) if pr_summary else None
  mem = Hippocampus(
      agent,
      budget=self._settings.get_memory_budget("code_review"),
      max_reflects=self._settings.get_memory_max_reflects("code_review"),
      question=question,
      task_name="code_review",
      run_id=run_id,
  )
  ```
- Remove comment about "No question_field" (lines 239-241)

### 10. Update Doc Reviewer

**File:** `src/codespy/agents/reviewer/modules/doc_reviewer.py`

- Add `pr_summary: str | None = None`, `mr_number: int | None = None`, `mr_title: str | None = None` to `aforward()` and `forward()` signatures
- Construct per-scope question:
  ```python
  question = (
      f"review documentation of {scope.repo}: {scope.subroot}: "
      f"pull request {mr_number} {mr_title}: {pr_summary}"
  ) if pr_summary else None
  ```
- Pass `question=question` to `Hippocampus(...)` call
- Remove comment about "No question_field" (lines 173-175)

### 11. Update Supply Chain Auditor

**File:** `src/codespy/agents/reviewer/modules/supply_chain_auditor.py`

- Add `pr_summary: str | None = None`, `mr_number: int | None = None`, `mr_title: str | None = None` to `aforward()` and `forward()` signatures
- Construct per-scope question:
  ```python
  question = (
      f"review supply chain of {scope.repo}: {scope.subroot}: "
      f"pull request {mr_number} {mr_title}: {pr_summary}"
  ) if pr_summary else None
  ```
- Pass `question=question` to `Hippocampus(...)` call

### 12. Update Config: Signature Names

**File:** `src/codespy/config_dspy.py`

Replace `"summarization"` with `"summary"` and `"audit"` in `SIGNATURE_NAMES`:
```python
SIGNATURE_NAMES = {
    "code_review",
    "doc",
    "scope",
    "supply_chain",
    "summary",
    "audit",
}
```

### 13. Update config references to "summarization"

**File:** `src/codespy/agents/reviewer/models.py` — line 123 description mentions `summarization`

**File:** `src/codespy/agents/reviewer/server.py` — line 107 mentions `summarization`

Update these doc references to say `summary` and `audit`.

---

## Edge Cases & Notes

- **Memory disabled for Summary**: pipeline still works — `Summarizer.forward()` runs `ChainOfThought` directly, `pr_summary` is still produced.
- **Summary signature disabled**: `Summarizer.forward()` returns `mr.title` as fallback.
- **Audit signature disabled**: `Auditor.forward()` returns fallback strings.
- **`question=None` fallback**: When `pr_summary` is None (standalone usage outside pipeline), `Hippocampus._make_question()` falls back to `format_inputs()` bounded by `max_question_tokens`.
- **Per-scope questions**: Each Hippocampus instance within the scope loop gets a unique question containing `scope.subroot`, so episodes are identifiable per scope.
- **`scope.repo`**: Already equals `mr.repo_slug` (set in `scope_identifier._convert_assignments_to_results`), so per-scope modules use `scope.repo` directly.
- **Episode question field on `Episode` model**: stores the task-specific question string.
- **`ReviewResult` model unchanged**: `overall_summary` populated from `Summarizer`; `quality_assessment` and `recommendation` from `Auditor`.

## Validation

1. Run the full pipeline on a sample MR and verify:
   - Summary runs before scope identification
   - Each module's episode contains its task-specific question (grep episode JSON files)
   - Questions are compact and identifiable (no huge serialized inputs)
   - Audit produces quality_assessment + recommendation
   - `ReviewResult` output structure unchanged
2. Verify env var overrides work for `"summary"` and `"audit"` (e.g. `SUMMARY_ENABLED=false`, `AUDIT_MODEL=...`)
3. Check episode file sizes are smaller (the original motivation)
