# Plan: Task-Specific Hippocampus Question Formats

## Goal

Replace generic `question=pr_summary` / `question=mr_title` with structured, task-specific question strings per module so episodes are identifiable and semantically meaningful. Introduce `PRContext` dataclass to bundle the shared PR identity fields.

## Target Question Formats

| Module | Question |
|--------|----------|
| Summary | `"summarize {repo_slug}: pull request {mr_number} {mr_title}"` |
| Scope | `"identify scopes of {repo_slug}: pull request {mr_number} {mr_title}: {summary}"` |
| Code Review | `"review code change of {repo_slug}: {scope.subroot}: pull request {mr_number} {mr_title}: {summary}"` |
| Doc | `"review documentation of {repo_slug}: {scope.subroot}: pull request {mr_number} {mr_title}: {summary}"` |
| Supply Chain | `"review supply chain of {repo_slug}: {scope.subroot}: pull request {mr_number} {mr_title}: {summary}"` |
| Audit | `"final audit of {repo_slug}: pull request {mr_number} {mr_title}: {summary}"` |

---

## Tasks

### 1. Create `PRContext` dataclass

**File:** `src/codespy/agents/reviewer/models.py`

```python
class PRContext(BaseModel):
    """Shared PR identity passed to all review modules after summarization.

    Built by the pipeline orchestrator after the Summarizer runs, then
    threaded through scope identification, review modules, and audit.
    Each module constructs its own Hippocampus question from these fields.
    """

    repo_slug: str = Field(description="Host-qualified repo identifier (e.g. github.com/owner/repo)")
    mr_number: int = Field(description="MR/PR number")
    mr_title: str = Field(description="MR/PR title")
    summary: str = Field(description="2-3 sentence PR summary produced by Summarizer")
```

### 2. Summarizer: add `mr_number`, format question

**File:** `src/codespy/agents/reviewer/modules/summarizer.py`

Summarizer is the producer of `summary` — it does NOT receive `PRContext`.

- Add `mr_number: int` parameter to `forward()` (between `mr_description` and `changed_file_paths`)
- Replace `question=mr_title` with:
  ```python
  question = f"summarize {repo_slug}: pull request {mr_number} {mr_title}"
  ```

### 3. Scope Identifier: accept `PRContext`, format question

**File:** `src/codespy/agents/reviewer/modules/scope_identifier.py`

- Replace `pr_summary: str | None = None` param with `pr_context: PRContext | None = None` on `aforward()` and `forward()`
- Construct question:
  ```python
  question = (
      f"identify scopes of {pr_context.repo_slug}: "
      f"pull request {pr_context.mr_number} {pr_context.mr_title}: {pr_context.summary}"
  ) if pr_context else None
  ```

### 4. Code Reviewer: accept `PRContext`, format per-scope question

**File:** `src/codespy/agents/reviewer/modules/code_reviewer.py`

- Replace `pr_summary: str | None = None` param with `pr_context: PRContext | None = None` on `aforward()` and `forward()`
- Construct per-scope question inside the scope loop:
  ```python
  question = (
      f"review code change of {scope.repo}: {scope.subroot}: "
      f"pull request {pr_context.mr_number} {pr_context.mr_title}: {pr_context.summary}"
  ) if pr_context else None
  ```

### 5. Doc Reviewer: accept `PRContext`, format per-scope question

**File:** `src/codespy/agents/reviewer/modules/doc_reviewer.py`

- Replace `pr_summary: str | None = None` param with `pr_context: PRContext | None = None` on `aforward()` and `forward()`
- Construct per-scope question:
  ```python
  question = (
      f"review documentation of {scope.repo}: {scope.subroot}: "
      f"pull request {pr_context.mr_number} {pr_context.mr_title}: {pr_context.summary}"
  ) if pr_context else None
  ```

### 6. Supply Chain Auditor: accept `PRContext`, format per-scope question

**File:** `src/codespy/agents/reviewer/modules/supply_chain_auditor.py`

- Replace `pr_summary: str | None = None` param with `pr_context: PRContext | None = None` on `aforward()` and `forward()`
- Construct per-scope question:
  ```python
  question = (
      f"review supply chain of {scope.repo}: {scope.subroot}: "
      f"pull request {pr_context.mr_number} {pr_context.mr_title}: {pr_context.summary}"
  ) if pr_context else None
  ```

### 7. Auditor: accept `PRContext`, format question

**File:** `src/codespy/agents/reviewer/modules/auditor.py`

- Replace `mr_title: str` and `pr_summary: str` params with `pr_context: PRContext` on `forward()`
- Remove `repo_slug: str` param (now from `pr_context.repo_slug`)
- Construct question:
  ```python
  question = (
      f"final audit of {pr_context.repo_slug}: "
      f"pull request {pr_context.mr_number} {pr_context.mr_title}: {pr_context.summary}"
  )
  ```
- Update signature call to extract fields:
  ```python
  result = auditor/mem.forward(
      mr_title=pr_context.mr_title,
      summary=pr_context.summary,
      changed_files=...,
      all_issues=...,
  )
  ```
- Update episode path: `f"/{pr_context.repo_slug}/root/"`

### 8. `_run_review_modules`: replace `pr_summary` with `pr_context`

**File:** `src/codespy/agents/reviewer/reviewer.py`

- Replace `pr_summary: str | None = None` param with `pr_context: PRContext | None = None`
- Pass `pr_context=pr_context` to each module call (replacing `pr_summary=pr_summary`)

### 9. `ReviewPipeline.forward()`: build `PRContext`, pass it downstream

**File:** `src/codespy/agents/reviewer/reviewer.py`

- Import `PRContext` from models
- Add `mr_number=mr.number` to `self.summarizer(...)` call
- After summarizer runs, build PRContext:
  ```python
  pr_context = PRContext(
      repo_slug=mr.repo_slug,
      mr_number=mr.number,
      mr_title=mr.title,
      summary=pr_summary,
  )
  ```
- Pass `pr_context=pr_context` to scope_identifier, `_run_review_modules`, and auditor
- Auditor call simplifies to:
  ```python
  quality_assessment, recommendation = self.auditor(
      pr_context=pr_context,
      changed_files=scoped_files,
      all_issues=all_issues,
      run_id=run_id,
  )
  ```

### 10. Update `modules/__init__.py` export

**File:** `src/codespy/agents/reviewer/modules/__init__.py`

No change needed — `PRContext` lives in `models.py`, not modules.

---

## Validation

- Run the pipeline on a sample MR. Grep episode JSON `question` fields — each should match the specified format.
- Standalone module usage (without `pr_context`) still works: `if pr_context else None` guard falls back to `Hippocampus._make_question()` using `format_inputs()`.
