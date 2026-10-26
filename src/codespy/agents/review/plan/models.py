"""Models for the implementation planner."""

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in the implementation plan.

    Steps are ordered by dependency first (across scopes), then by severity.
    Related issues are grouped into one step. Each step changes exactly one scope.
    """

    title: str = Field(description="Brief title of the step (<10 words)")
    scope: str = Field(
        default="",
        description="Subroot of the one scope this step changes; must be one of the input scopes",
    )
    files: list[str] = Field(
        default_factory=list,
        description="Files to modify (repo-root-relative paths)"
    )
    change: str = Field(
        description="What to change and how. Imperative, ≤50 words."
    )
    verification: str = Field(
        description="Concrete test or check to run to verify the fix"
    )
    addresses: list[str] = Field(
        default_factory=list,
        description="Titles of the issues this step fixes"
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="1-based positions of earlier steps this step needs, any scope",
    )
    number: int = Field(default=0, description="Global step number, set by codespy")


class ScopePlan(BaseModel):
    """Steps of one scope within the holistic plan."""

    subroot: str = Field(description="Scope subroot path")
    steps: list[PlanStep] = Field(default_factory=list, description="Ordered steps")
