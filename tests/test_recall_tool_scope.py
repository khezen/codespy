"""Tests for recall_memory tool scope handling in code_review and supply_chain agents.

This tests the fix for: `recall_memory` with `reach=local` returned nothing for
code_review and supply_chain because they passed `scope_topic_ids=None` to
get_prefrontal(). The fix ensures they pass `scope_topic_ids=[scope.topic(repo).id]`
so local_tags() returns the project_scope tag.
"""

import pytest


class TestLocalTagsWithScope:
    """Unit tests for local_tags returning project_scope tag."""

    def test_local_tags_returns_project_scope(self):
        """local_tags with scope_id returns project_scope tag."""
        # Skip if prefrontal.reach not available (Hindsight not installed)
        try:
            from codespy.agents.memory.prefrontal.reach import local_tags
        except ImportError:
            pytest.skip("prefrontal.reach module not available")

        result = local_tags("github.com/o/r", ["scope123"], False)
        assert "project_scope:scope123" in result

    def test_local_tags_without_scope_empty(self):
        """local_tags with empty scope_ids returns empty list for non-repo scope."""
        try:
            from codespy.agents.memory.prefrontal.reach import local_tags
        except ImportError:
            pytest.skip("prefrontal.reach module not available")

        result = local_tags("github.com/o/r", [], False)
        assert result == []


class TestRootScopeReach:
    """include_repo means PR scopes + repo-root scope, never every repo: fact."""

    def test_local_tags_include_repo_uses_root_scope(self):
        from codespy.agents.memory.prefrontal.reach import local_tags

        result = local_tags("o/r", ["o/r/svc"], True)
        assert result == ["project_scope:o/r/svc", "project_scope:o/r"]
        assert not any(t.startswith("repo:") for t in result)

    def test_local_tags_root_deduped(self):
        from codespy.agents.memory.prefrontal.reach import local_tags

        assert local_tags("o/r", ["o/r", "o/r/svc"], True) == [
            "project_scope:o/r",
            "project_scope:o/r/svc",
        ]

    def test_local_tags_include_repo_without_scopes(self):
        from codespy.agents.memory.prefrontal.reach import local_tags

        assert local_tags("o/r", None, True) == ["project_scope:o/r"]

    def test_mental_model_ids_never_include_bare_repo_briefing(self):
        from codespy.agents.memory.prefrontal.reach import (
            mental_model_id,
            mental_model_ids,
            repo_tags,
            scope_tags,
        )

        ids = mental_model_ids("o/r", ["o/r/svc"], True)
        assert ids == [
            mental_model_id(scope_tags("o/r", "o/r/svc")),
            mental_model_id(scope_tags("o/r", "o/r")),
        ]
        assert mental_model_id(repo_tags("o/r")) not in ids

    def test_reflect_groups_org_reach_keeps_org_tag(self):
        pytest.importorskip("hindsight_api")
        from codespy.agents.memory.prefrontal.reach import build_reflect_groups

        groups = build_reflect_groups("o/r", ["o/r/svc"], True, "org")
        assert groups[0].tags == ["project_scope:o/r/svc", "project_scope:o/r", "org:o"]


class TestScopeAgentRootTopic:
    def test_scope_episode_bound_to_root_scope_only(self):
        """The scope agent binds its episode to the repo-root topic, not every PR scope."""
        with open("src/codespy/agents/review/scope/agent.py") as f:
            source = f.read()
        assert "root_topic_id = root_scope_id(pr.repo_full_name)" in source
        assert "hippo.bind_topics(scope_topics, [root_topic_id, pr_ctx.pr_url])" in source
        assert "compute_common_ancestor_topic_id" not in source


class TestAgentSourceCode:
    """Verify agents pass scope_topic_ids to get_prefrontal by inspecting source.

    This checks that the agent code was updated to pass scope_topic_ids.
    The actual code flow is:
    1. code_review agent: calls scope.topic(repo_full_name).id
    2. passes scope_topic_ids=[topic_id] to get_prefrontal()
    3. get_prefrontal -> Prefrontal() -> local_tags(scope_topic_ids)
    4. local_tags returns ["project_scope:<id>"]
    """

    def test_code_review_agent_has_scope_topic_id_computation(self):
        """Verify code_review/agent.py computes scope_topic_id before get_prefrontal."""
        with open("src/codespy/agents/review/code_review/agent.py") as f:
            source = f.read()
        # The fix adds: scope_topic_id = scope.topic(repo_full_name).id
        assert "scope_topic_id = scope.topic(repo_full_name).id" in source
        # And passes: scope_topic_ids=[scope_topic_id]
        assert "scope_topic_ids=[scope_topic_id]" in source

    def test_supply_chain_agent_has_scope_topic_id_computation(self):
        """Verify supply_chain/agent.py computes scope_topic_id before get_prefrontal."""
        with open("src/codespy/agents/review/supply_chain/agent.py") as f:
            source = f.read()
        # The fix adds: scope_topic_id = scope.topic(repo_full_name).id
        assert "scope_topic_id = scope.topic(repo_full_name).id" in source
        # And passes: scope_topic_ids=[scope_topic_id]
        assert "scope_topic_ids=[scope_topic_id]" in source
