"""Tests for ContextMemory with Topics."""

from pathlib import Path
from unittest.mock import patch

import pytest

from codespy.agents.memory.hippocampus import (
    ContextMemory,
    Item,
    OpType,
    Operation,
    Topic,
    compute_common_ancestor_topic_id,
    make_topic_id,
)
from codespy.agents.reviewer.modules.manifest_parser import extract_package_name


class TestMakeTopicId:
    """Tests for make_topic_id function."""

    def test_package_name_contains_repo(self):
        """Path 1: package_name contains owner/repo -> use from owner/repo onwards."""
        package_name = "github.com/owner/repo/packages/auth"
        result = make_topic_id("owner/repo", "packages/auth", package_name)
        assert result == "owner/repo/packages/auth"

    def test_package_name_without_repo(self):
        """Path 2: package_name provided but doesn't contain repo -> prepend repo."""
        package_name = "@myorg/auth-service"
        result = make_topic_id("owner/repo", "packages/auth", package_name)
        assert result == "owner/repo/@myorg/auth-service"

    def test_subroot_fallback(self):
        """Path 3: no package_name -> use subroot."""
        result = make_topic_id("owner/repo", "packages/auth", None)
        assert result == "owner/repo/packages/auth"

    def test_root_scope(self):
        """Root scope (subroot='.') returns just repo_full_name."""
        result = make_topic_id("owner/repo", ".", None)
        assert result == "owner/repo"

    def test_empty_subroot(self):
        """Empty subroot returns just repo_full_name."""
        result = make_topic_id("owner/repo", "", None)
        assert result == "owner/repo"

    def test_strips_leading_slashes(self):
        """Leading slashes in subroot are stripped."""
        result = make_topic_id("owner/repo", "/packages/auth/", None)
        assert result == "owner/repo/packages/auth"


class TestComputeCommonAncestorTopicId:
    """Tests for compute_common_ancestor_topic_id function."""

    def test_single_scope_returns_none(self):
        """Single scope returns None (no common ancestor needed)."""
        result = compute_common_ancestor_topic_id("owner/repo", ["packages/auth"])
        assert result is None

    def test_empty_list_returns_none(self):
        """Empty list returns None."""
        result = compute_common_ancestor_topic_id("owner/repo", [])
        assert result is None

    def test_two_scopes_with_common_ancestor(self):
        """Two scopes sharing common ancestor."""
        subroots = ["packages/auth", "packages/api"]
        result = compute_common_ancestor_topic_id("owner/repo", subroots)
        assert result == "owner/repo/packages"

    def test_three_scopes_with_common_ancestor(self):
        """Three scopes with nested common ancestor."""
        subroots = ["packages/auth/src", "packages/auth/tests", "packages/api"]
        result = compute_common_ancestor_topic_id("owner/repo", subroots)
        assert result == "owner/repo/packages"

    def test_disjoint_scopes_returns_root(self):
        """Disjoint scopes with no common ancestor return root."""
        subroots = ["frontend", "backend"]
        result = compute_common_ancestor_topic_id("owner/repo", subroots)
        assert result == "owner/repo"

    def test_one_root_scope(self):
        """Mix of root and nested scopes returns root."""
        subroots = [".", "packages/auth"]
        result = compute_common_ancestor_topic_id("owner/repo", subroots)
        assert result == "owner/repo"

    def test_identical_scopes(self):
        """Identical scopes return that scope."""
        subroots = ["packages/auth", "packages/auth"]
        result = compute_common_ancestor_topic_id("owner/repo", subroots)
        assert result == "owner/repo/packages/auth"


class TestExtractPackageName:
    """Tests for extract_package_name function."""

    def test_extract_from_package_json(self, tmp_path: Path):
        """Extract name from package.json."""
        manifest = tmp_path / "package.json"
        manifest.write_text('{"name": "my-package", "version": "1.0.0"}')
        result = extract_package_name("package.json", tmp_path)
        assert result == "my-package"

    def test_extract_from_composer_json(self, tmp_path: Path):
        """Extract name from composer.json."""
        manifest = tmp_path / "composer.json"
        manifest.write_text('{"name": "vendor/package", "version": "1.0.0"}')
        result = extract_package_name("composer.json", tmp_path)
        assert result == "vendor/package"

    def test_extract_from_go_mod(self, tmp_path: Path):
        """Extract module from go.mod."""
        manifest = tmp_path / "go.mod"
        manifest.write_text("module github.com/owner/repo\n\ngo 1.21\n")
        result = extract_package_name("go.mod", tmp_path)
        assert result == "github.com/owner/repo"

    def test_extract_from_cargo_toml(self, tmp_path: Path):
        """Extract name from Cargo.toml."""
        manifest = tmp_path / "Cargo.toml"
        manifest.write_text('[package]\nname = "my-crate"\nversion = "1.0.0"')
        result = extract_package_name("Cargo.toml", tmp_path)
        assert result == "my-crate"

    def test_extract_from_pubspec_yaml(self, tmp_path: Path):
        """Extract name from pubspec.yaml."""
        manifest = tmp_path / "pubspec.yaml"
        manifest.write_text("name: my_app\ndescription: A Flutter app\n")
        result = extract_package_name("pubspec.yaml", tmp_path)
        assert result == "my_app"

    def test_extract_from_chart_yaml(self, tmp_path: Path):
        """Extract name from Chart.yaml."""
        manifest = tmp_path / "Chart.yaml"
        manifest.write_text("apiVersion: v2\nname: my-chart\nversion: 1.0.0\n")
        result = extract_package_name("Chart.yaml", tmp_path)
        assert result == "my-chart"

    def test_extract_from_setup_cfg(self, tmp_path: Path):
        """Extract name from setup.cfg."""
        manifest = tmp_path / "setup.cfg"
        manifest.write_text("[metadata]\nname = my-package\nversion = 1.0.0\n")
        result = extract_package_name("setup.cfg", tmp_path)
        assert result == "my-package"

    def test_pyproject_toml_project_name(self, tmp_path: Path):
        """Extract name from pyproject.toml [project]."""
        manifest = tmp_path / "pyproject.toml"
        manifest.write_text('[project]\nname = "my-project"\nversion = "1.0.0"\n')
        result = extract_package_name("pyproject.toml", tmp_path)
        assert result == "my-project"

    def test_pyproject_toml_poetry_name(self, tmp_path: Path):
        """Extract name from pyproject.toml [tool.poetry]."""
        manifest = tmp_path / "pyproject.toml"
        manifest.write_text('[tool.poetry]\nname = "poetry-project"\nversion = "1.0.0"\n')
        result = extract_package_name("pyproject.toml", tmp_path)
        assert result == "poetry-project"

    def test_missing_manifest_returns_none(self, tmp_path: Path):
        """Missing manifest file returns None."""
        result = extract_package_name("package.json", tmp_path)
        assert result is None

    def test_malformed_json_returns_none(self, tmp_path: Path):
        """Malformed JSON returns None."""
        manifest = tmp_path / "package.json"
        manifest.write_text("not valid json")
        result = extract_package_name("package.json", tmp_path)
        assert result is None

    def test_missing_name_field_returns_none(self, tmp_path: Path):
        """JSON without name field returns None."""
        manifest = tmp_path / "package.json"
        manifest.write_text('{"version": "1.0.0"}')
        result = extract_package_name("package.json", tmp_path)
        assert result is None

    def test_gradle_from_settings_gradle(self, tmp_path: Path):
        """Extract project name from settings.gradle."""
        manifest = tmp_path / "build.gradle"
        manifest.write_text("")
        settings = tmp_path / "settings.gradle"
        settings.write_text("rootProject.name = 'my-project'")
        result = extract_package_name("build.gradle", tmp_path)
        assert result == "my-project"


class TestContextMemoryRender:
    """Tests for ContextMemory.render() method."""

    def test_topic_grouped_format(self):
        """Render produces topic-grouped format."""
        topics = [
            Topic(id="owner/repo/auth", description="Auth service"),
            Topic(id="owner/repo/api", description="API gateway"),
        ]
        items = [
            Item(id="cu-abc", content="Auth uses JWT", topic_ids=["owner/repo/auth"]),
            Item(id="cu-def", content="API uses rate limiting", topic_ids=["owner/repo/api"]),
        ]
        memory = ContextMemory(
            topics=topics,
            context_understanding=items,
        )
        result = memory.render()

        assert "## TOPIC: owner/repo/auth (Auth service)" in result
        assert "## TOPIC: owner/repo/api (API gateway)" in result
        assert "[cu-abc] Auth uses JWT" in result
        assert "[cu-def] API uses rate limiting" in result

    def test_shared_group_for_multiple_topic_ids(self):
        """Items with 2+ topic_ids go to SHARED group."""
        topics = [
            Topic(id="owner/repo/auth", description="Auth service"),
            Topic(id="owner/repo/api", description="API gateway"),
        ]
        items = [
            Item(id="cu-shared", content="Shared context", topic_ids=["owner/repo/auth", "owner/repo/api"]),
        ]
        memory = ContextMemory(
            topics=topics,
            context_understanding=items,
        )
        result = memory.render()

        assert "## SHARED" in result
        assert "[cu-shared] Shared context" in result

    def test_shared_group_for_empty_topic_ids(self):
        """Items with empty topic_ids go to SHARED group."""
        topics = [Topic(id="owner/repo/auth", description="Auth service")]
        items = [
            Item(id="cu-empty", content="No topic", topic_ids=[]),
        ]
        memory = ContextMemory(
            topics=topics,
            context_understanding=items,
        )
        result = memory.render()

        assert "## SHARED" in result
        assert "[cu-empty] No topic" in result

    def test_shared_group_for_unknown_topic_id(self):
        """Items with unknown topic_id go to SHARED group."""
        topics = [Topic(id="owner/repo/auth", description="Auth service")]
        items = [
            Item(id="cu-unknown", content="Unknown topic", topic_ids=["nonexistent"]),
        ]
        memory = ContextMemory(
            topics=topics,
            context_understanding=items,
        )
        result = memory.render()

        assert "## SHARED" in result
        assert "[cu-unknown] Unknown topic" in result

    def test_empty_memory_returns_empty_string(self):
        """Completely empty memory returns empty string."""
        memory = ContextMemory()
        result = memory.render()
        assert result == ""

    def test_topic_with_no_items_is_hidden(self):
        """Topics with no items are not rendered."""
        topics = [
            Topic(id="owner/repo/auth", description="Auth service"),
            Topic(id="owner/repo/empty", description="Empty scope"),
        ]
        items = [
            Item(id="cu-abc", content="Only auth has items", topic_ids=["owner/repo/auth"]),
        ]
        memory = ContextMemory(
            topics=topics,
            context_understanding=items,
        )
        result = memory.render()

        assert "owner/repo/auth" in result
        assert "owner/repo/empty" not in result

    def test_section_headers_in_render(self):
        """Render includes section headers for non-empty sections."""
        topics = [Topic(id="owner/repo/auth", description="Auth service")]
        memory = ContextMemory(
            topics=topics,
            context_roadmap=[Item(id="cr-1", content="Roadmap item", topic_ids=["owner/repo/auth"])],
            context_understanding=[Item(id="cu-1", content="Understanding item", topic_ids=["owner/repo/auth"])],
        )
        result = memory.render()

        assert "### CONTEXT ROADMAP" in result
        assert "### CONTEXT UNDERSTANDING" in result

    def test_shared_appears_first(self):
        """SHARED group appears before topic groups."""
        topics = [Topic(id="owner/repo/auth", description="Auth service")]
        memory = ContextMemory(
            topics=topics,
            context_understanding=[
                Item(id="cu-1", content="Topic item", topic_ids=["owner/repo/auth"]),
                Item(id="cu-2", content="Shared item", topic_ids=[]),
            ],
        )
        result = memory.render()

        shared_pos = result.find("## SHARED")
        topic_pos = result.find("## TOPIC")
        assert shared_pos < topic_pos


class TestContextMemoryApply:
    """Tests for ContextMemory.apply() method."""

    def test_add_operation_with_topic_ids(self):
        """ADD operation assigns topic_ids to new items."""
        memory = ContextMemory(topics=[Topic(id="t1", description="Test")])
        ops = [Operation(type=OpType.ADD, section="context_understanding", content="New item")]
        new_memory, new_ids = memory.apply(ops, topic_ids=["t1"])

        assert len(new_ids) == 1
        item = new_memory.context_understanding[0]
        assert item.topic_ids == ["t1"]

    def test_replace_preserves_existing_topic_ids(self):
        """REPLACE operation preserves existing topic_ids."""
        memory = ContextMemory(
            context_understanding=[
                Item(id="cu-abc", content="Original", topic_ids=["t1"]),
            ],
        )
        ops = [Operation(type=OpType.REPLACE, item_id="cu-abc", content="Replaced")]
        new_memory, _ = memory.apply(ops, topic_ids=["t2"])

        item = new_memory.context_understanding[0]
        assert item.content == "Replaced"
        assert item.topic_ids == ["t1"]  # Preserved, not overwritten

    def test_add_without_topic_ids(self):
        """ADD without topic_ids creates item with empty topic_ids."""
        memory = ContextMemory()
        ops = [Operation(type=OpType.ADD, section="context_understanding", content="New item")]
        new_memory, _ = memory.apply(ops)

        item = new_memory.context_understanding[0]
        assert item.topic_ids == []

    def test_delete_removes_item(self):
        """DELETE operation removes item."""
        memory = ContextMemory(
            context_understanding=[
                Item(id="cu-abc", content="To delete", topic_ids=["t1"]),
            ],
        )
        ops = [Operation(type=OpType.DELETE, item_id="cu-abc")]
        new_memory, _ = memory.apply(ops)

        assert len(new_memory.context_understanding) == 0


class TestContextMemoryMerge:
    """Tests for ContextMemory.merge() method."""

    def test_merge_deduplicates_topics(self):
        """Merge deduplicates topics by ID."""
        mem1 = ContextMemory(topics=[Topic(id="t1", description="First")])
        mem2 = ContextMemory(topics=[Topic(id="t1", description="Second")])
        merged = ContextMemory.merge(mem1, mem2)

        assert len(merged.topics) == 1

    def test_merge_later_description_wins(self):
        """Later non-empty description wins in topic merge."""
        mem1 = ContextMemory(topics=[Topic(id="t1", description="")])
        mem2 = ContextMemory(topics=[Topic(id="t1", description="Better description")])
        merged = ContextMemory.merge(mem1, mem2)

        assert merged.topics[0].description == "Better description"

    def test_merge_items_by_id(self):
        """Merge replaces items with same ID (later wins)."""
        mem1 = ContextMemory(
            context_understanding=[Item(id="cu-abc", content="First", topic_ids=["t1"])],
        )
        mem2 = ContextMemory(
            context_understanding=[Item(id="cu-abc", content="Second", topic_ids=["t2"])],
        )
        merged = ContextMemory.merge(mem1, mem2)

        assert len(merged.context_understanding) == 1
        assert merged.context_understanding[0].content == "Second"

    def test_merge_multiple_memories(self):
        """Merge can handle multiple memories."""
        mem1 = ContextMemory(
            topics=[Topic(id="t1", description="T1")],
            context_understanding=[Item(id="cu-1", content="Item 1", topic_ids=["t1"])],
        )
        mem2 = ContextMemory(
            topics=[Topic(id="t2", description="T2")],
            context_understanding=[Item(id="cu-2", content="Item 2", topic_ids=["t2"])],
        )
        mem3 = ContextMemory(
            topics=[Topic(id="t3", description="T3")],
            domain_constants=[Item(id="dc-1", content="Constant", topic_ids=["t3"])],
        )
        merged = ContextMemory.merge(mem1, mem2, mem3)

        assert len(merged.topics) == 3
        assert len(merged.context_understanding) == 2
        assert len(merged.domain_constants) == 1


class TestTopicDependencies:
    """Tests for Topic.dependencies field."""

    def test_topic_with_dependencies(self):
        """Topic can have dependencies."""
        topic = Topic(
            id="owner/repo/auth",
            description="Auth service",
            dependencies=["owner/repo/core", "PyPI/passlib"]
        )
        assert topic.dependencies == ["owner/repo/core", "PyPI/passlib"]

    def test_topic_dependencies_default_empty(self):
        """Topic dependencies default to empty list."""
        topic = Topic(id="owner/repo/auth", description="Auth service")
        assert topic.dependencies == []

    def test_topic_deserialization_without_dependencies(self):
        """Old episodes without dependencies deserialize to empty list."""
        import json
        old_data = '{"id": "owner/repo/auth", "description": "Auth service"}'
        topic = Topic.model_validate_json(old_data)
        assert topic.dependencies == []


class TestScopeResultTopicHelper:
    """Tests for ScopeResult.topic() helper method."""

    def test_topic_with_package_manifest(self):
        """Topic uses package_name from manifest when available."""
        from codespy.agents.reviewer.models import PackageManifest, ScopeResult, ScopeType

        scope = ScopeResult(
            repo="owner/repo",
            subroot="packages/auth",
            scope_type=ScopeType.SERVICE,
            reason="test",
            description="Auth service",
            package_manifest=PackageManifest(
                manifest_path="packages/auth/package.json",
                package_manager="npm",
                package_name="@myorg/auth",
            ),
        )
        topic = scope.topic("owner/repo")

        assert topic.id == "owner/repo/@myorg/auth"
        assert topic.description == "Auth service"

    def test_topic_without_package_manifest(self):
        """Topic uses subroot when no package manifest."""
        from codespy.agents.reviewer.models import ScopeResult, ScopeType

        scope = ScopeResult(
            repo="owner/repo",
            subroot="services/api",
            scope_type=ScopeType.SERVICE,
            reason="test",
            description="API service",
        )
        topic = scope.topic("owner/repo")

        assert topic.id == "owner/repo/services/api"
        assert topic.description == "API service"

    def test_topic_for_root_scope(self):
        """Root scope topic uses just repo_full_name."""
        from codespy.agents.reviewer.models import ScopeResult, ScopeType

        scope = ScopeResult(
            repo="owner/repo",
            subroot=".",
            scope_type=ScopeType.APPLICATION,
            reason="test",
            description="Repository root",
        )
        topic = scope.topic("owner/repo")

        assert topic.id == "owner/repo"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
