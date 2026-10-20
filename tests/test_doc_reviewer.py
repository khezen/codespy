"""Regression tests for DocReviewer's RLM call kwargs."""

from unittest.mock import MagicMock, patch

import dspy
import pytest

from codespy.agents.memory.prefrontal import with_prefrontal_memory
from codespy.agents.review.doc.agent import DocReviewer, DocReviewSignature
from codespy.agents.review.models import PRContext, ReviewContext, ReviewMetadata
from codespy.agents.review.scope.models import ScopeResult, ScopeType
from codespy.tools.git.models import ChangedFile, FileStatus


def _run_doc_review(tmp_path, prefrontal_memory: str) -> dict:
    """Run DocReviewer on a single scope and return the kwargs passed to run_rlm."""
    mock_settings = MagicMock()
    mock_settings.is_signature_enabled.return_value = True
    mock_settings.memory.enabled = False
    mock_settings.get_max_iters.return_value = 5
    mock_settings.get_max_llm_calls.return_value = 16
    mock_settings.get_rlm_threshold.return_value = 0.30
    mock_settings.review.min_confidence = 0.0

    captured_kwargs: dict = {}

    async def capture_run_rlm(agent, **kwargs):
        captured_kwargs.update(kwargs)
        return dspy.Prediction(issues=[])

    scope = ScopeResult(
        repo="o/r",
        subroot="svc/a",
        scope_type=ScopeType.SERVICE,
        has_changes=True,
        changed_files=[
            ChangedFile(
                filename="svc/a/main.py",
                status=FileStatus.MODIFIED,
                patch="@@ -1 +1 @@\n-a\n+b",
            )
        ],
        reason="test",
    )
    review_context = ReviewContext(
        pr_context=PRContext(
            repo_slug="github.com/o/r",
            pr_number=1,
            pr_title="t",
            pr_url="https://github.com/o/r/pull/1",
            summary="s",
        ),
        metadata=ReviewMetadata(repo_path=tmp_path, run_id="test-run"),
        prefrontal_memory=prefrontal_memory,
    )

    mod = "codespy.agents.review.doc.agent"
    with (
        patch(f"{mod}.get_settings", return_value=mock_settings),
        patch(f"{mod}.get_cost_tracker", return_value=MagicMock()),
        patch(f"{mod}.build_rlm_agent", return_value=MagicMock()),
        patch(f"{mod}.run_rlm", side_effect=capture_run_rlm),
        patch(f"{mod}.log_rlm_outcome"),
        patch(f"{mod}.get_prefrontal", return_value=None),
        patch(f"{mod}.extract_documentation", return_value="=== README.md ===\nx"),
        patch(f"{mod}.resolve_scope_root", return_value=tmp_path),
    ):
        issues, _ = DocReviewer().forward(scopes=[scope], review_context=review_context)

    assert issues == []
    assert captured_kwargs, "run_rlm was not called"
    return captured_kwargs


class TestDocReviewerRlmInputs:
    """run_rlm must receive exactly the signature's declared inputs."""

    def test_kwargs_match_signature_without_prefrontal(self, tmp_path):
        kwargs = _run_doc_review(tmp_path, prefrontal_memory="")
        assert set(kwargs) == set(DocReviewSignature.input_fields)
        assert "--- main.py ---" in kwargs["patches"]
        assert kwargs["documentation"] == "=== README.md ===\nx"

    def test_kwargs_match_signature_with_prefrontal(self, tmp_path):
        kwargs = _run_doc_review(tmp_path, prefrontal_memory="m")
        assert set(kwargs) == set(with_prefrontal_memory(DocReviewSignature).input_fields)
        assert kwargs["prefrontal_memory"] == "m"


if __name__ == "__main__":
    pytest.main([__file__])
