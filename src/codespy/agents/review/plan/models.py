"""Models for the implementation planner."""

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in the implementation plan.

    Steps are ordered by dependency first, then by severity.
    Related issues are grouped into one step.
    """

    title: str = Field(description="Brief title of the step (<10 words)")
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


class ScopePlan(BaseModel):
    """Implementation plan for a single scope."""

    subroot: str = Field(description="Scope subroot path")
    steps: list[PlanStep] = Field(default_factory=list, description="Ordered steps")
