"""Models for the implementation planner."""

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in the implementation plan.

    Steps are ordered by dependency first (across scopes), then by severity.
    Related issues are grouped into one step. Each step changes exactly one scope.
    """

    title: str = Field(description="Brief title of the step (<10 words)")
    scope: str = Field(
        description="Subroot of the one scope this step changes; must be one of the input scopes",
    )
    files: list[str] = Field(
        default_factory=list,
        description="Files to modify (repo-root-relative paths)",
    )
    change: str = Field(
        description="What to change and how. Imperative, ≤50 words.",
    )
    verification: str = Field(
        description="Concrete test or check to run to verify the fix",
    )
    addresses: list[str] = Field(
        default_factory=list,
        description="Titles of the issues this step fixes",
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="1-based positions of earlier steps this step needs, any scope",
    )
    number: int = Field(description="Global step number, set by codespy")


class PlanStepDraft(BaseModel):
    """A draft step output by the RLM, before numbering.

    All PlanStep fields except ``number``.
    """

    title: str = Field(description="Brief title of the step (<10 words)")
    scope: str = Field(
        default="",
        description="Subroot of the one scope this step changes; must be one of the input scopes",
    )
    files: list[str] = Field(
        default_factory=list,
        description="Files to modify (repo-root-relative paths)",
    )
    change: str = Field(
        description="What to change and how. Imperative, ≤50 words.",
    )
    verification: str = Field(
        description="Concrete test or check to run to verify the fix",
    )
    addresses: list[str] = Field(
        default_factory=list,
        description="Titles of the issues this step fixes",
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="1-based positions of earlier steps this step needs, any scope",
    )


class ScopePlan(BaseModel):
    """Steps of one scope within the holistic plan."""

    subroot: str = Field(description="Scope subroot path")
    steps: list[PlanStep] = Field(default_factory=list, description="Ordered steps")


class ImplementationPlan(BaseModel):
    """A structured implementation plan with sections for design review.

    Sections render in this order: Goal → Decisions → Tasks (steps per scope)
    → Risks → Validation. Empty sections are omitted.
    """

    goal: str = Field(
        default="",
        description="End state once every issue is fixed, one or two sentences",
    )
    decisions: list[str] = Field(
        default_factory=list,
        description="Choices between alternatives, each with the reason (≤8 items, ≤25 words each)",
    )
    scopes: list[ScopePlan] = Field(
        default_factory=list,
        description="Steps grouped per scope; steps carry global numbers",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="What could break or regress (≤8 items, ≤25 words each)",
    )
    validation: list[str] = Field(
        default_factory=list,
        description="End-to-end checks to run after all steps (≤8 items, ≤25 words each)",
    )

    @property
    def total_steps(self) -> int:
        """Total number of steps across all scopes."""
        return sum(len(sp.steps) for sp in self.scopes)
