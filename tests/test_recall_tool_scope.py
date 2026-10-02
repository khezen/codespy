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
