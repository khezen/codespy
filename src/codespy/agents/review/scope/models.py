"""Scope-agent-owned types."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from codespy.tools.git.models import ChangedFile

if TYPE_CHECKING:
    from codespy.agents.memory.hippocampus.context_memory import Topic


class ScopeType(StrEnum):
    """Type of code scope in a repository."""

    LIBRARY = "library"  # Shared code that others import
    SERVICE = "service"  # Isolated microservice with explicit APIs
    APPLICATION = "application"  # Standalone app or frontend
    SCRIPT = "script"  # Build/deployment scripts, tooling


class PackageManifest(BaseModel):
    """Package management file information for a scope."""

    manifest_path: str = Field(description="Path to manifest file (e.g., package.json)")
    lock_file_path: str | None = Field(
        default=None, description="Path to lock file (e.g., package-lock.json)"
    )
    package_manager: str = Field(description="Package manager name (e.g., npm, go, pip)")
    dependencies_changed: bool = Field(
        default=False, description="Whether PR modified this manifest or lock file"
    )
    package_name: str | None = Field(default=None, description="Package identity from manifest")


class ScopeResult(BaseModel):
    """A detected scope/subroot in the repository."""

    repo: str = Field(
        default="", description="Repo identifier: 'owner/repo' (remote) or local dir name"
    )
    subroot: str = Field(description="Path relative to repo root (e.g., packages/auth)")
    scope_type: ScopeType = Field(description="Type of scope (library, service, etc.)")
    has_changes: bool = Field(
        default=False, description="Whether this scope has changed files from PR"
    )

    language: str | None = Field(default=None, description="Primary language detected")
    package_manifest: PackageManifest | None = Field(
        default=None, description="Package manifest info if present"
    )
    changed_files: list[ChangedFile] = Field(
        default_factory=list, description="Changed files belonging to this scope"
    )
    reason: str = Field(description="Explanation for why this scope was identified")
    skills: str | None = Field(
        default=None, description="Project/scope instructions inherited from ancestor directories"
    )
    description: str = Field(
        default="", description="Description of scope's role in the project (max 500 chars)"
    )

    model_config = {"arbitrary_types_allowed": True}

    def scope_path(self) -> str:
        """Return the storage-relative path for this scope.

        Used by Hippocampus memory as the base directory for episode files.
        ``subroot == "."`` (repo root) results in ``/{repo}/``.
        """
        if self.subroot in (".", ""):
            return f"/{self.repo}/"
        return f"/{self.repo}/{self.subroot.strip('/')}/"

    def topic(self, repo_full_name: str) -> "Topic":
        """Build the Topic for this scope.

        Args:
            repo_full_name: Repository full name (owner/repo)

        Returns:
            Topic object with id and description
        """
        from codespy.agents.memory.hippocampus.context_memory import Topic, make_topic_id

        package_name = self.package_manifest.package_name if self.package_manifest else None
        topic_id = make_topic_id(repo_full_name, self.subroot, package_name)
        return Topic(id=topic_id, type="project_scope", description=self.description)
