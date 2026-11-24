"""Models for the implementation agent."""

from pydantic import BaseModel, Field, computed_field


class SkippedStep(BaseModel):
    """A plan step the agent did not implement."""

    number: int = Field(description="Global plan step number")
    reason: str = Field(description="Why the step was skipped, one sentence")


class FileChange(BaseModel):
    """One file changed by the implementation agent (repo-root-relative)."""

    path: str = Field(description="Path relative to the repository root")
    # Content stays out of JSON output: it can be large and is published as a commit
    content: str | None = Field(
        default=None, exclude=True, description="New UTF-8 content, None when deleted"
    )
    executable: bool = Field(default=False, description="File has the executable bit")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> str:
        """'deleted' or 'written'."""
        return "deleted" if self.content is None else "written"


class ImplementationResult(BaseModel):
    """Result of the implementation agent: edits plus what was done and skipped."""

    summary: str = Field(default="", description="What was implemented")
    completed_steps: list[int] = Field(default_factory=list, description="Completed step numbers")
    skipped_steps: list[SkippedStep] = Field(default_factory=list, description="Skipped steps")
    changes: list[FileChange] = Field(default_factory=list, description="Changed files")
    pull_request_url: str | None = Field(
        default=None, description="Fix PR/MR URL (PR reviews), set when published"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def changed_files(self) -> list[str]:
        """Paths of the changed files."""
        return [c.path for c in self.changes]

    def files_payload(self) -> dict[str, str | None]:
        """``path -> content`` (None = delete), as committed by the git clients."""
        return {c.path: c.content for c in self.changes}

    def executable_paths(self) -> frozenset[str]:
        """Paths written with the executable bit."""
        return frozenset(c.path for c in self.changes if c.content is not None and c.executable)
