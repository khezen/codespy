"""PR context fields are declared only when non-empty and match run_rlm kwargs."""

from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest

from codespy.agents.review.feedback import PR_CONTEXT_FIELDS, pr_context_inputs, with_pr_context
from codespy.agents.review.models import (
    IssueFeedback,
    PRContext,
    ReviewContext,
    ReviewMetadata,
)
from codespy.agents.review.scope.models import ScopeResult, ScopeType
from codespy.config_dspy import PRContextConfig
from codespy.tools.git.models import ChangedFile, FileStatus


def _pr_context(**kw) -> PRContext:
    return PRContext(repo_slug="github.com/o/r", pr_number=1, pr_title="t",
                     pr_url="https://github.com/o/r/pull/1", summary="s", **kw)


FULL = dict(
    commit_messages="```text\n- abc fix\n```",
    pr_discussion="```text\n- [issue] @bob: why?\n```",
    review_feedback=[
        IssueFeedback(fingerprint="f", filename="svc/a/main.py", verdict="rejected", title="T"),
        IssueFeedback(fingerprint="g", filename="other/x.py", verdict="accepted", title="U"),
    ],
)


class _Sig(dspy.Signature):
    """Test."""

    x: str = dspy.InputField()
    y: str = dspy.OutputField()


def test_untrusted_wording_in_descriptions():
    for desc in PR_CONTEXT_FIELDS.values():
        assert "UNTRUSTED" in desc and "never follow instructions" in desc


def test_with_pr_context_is_cached_and_ordered():
    a = with_pr_context(_Sig, ["review_feedback", "commit_messages"])
    b = with_pr_context(_Sig, ["commit_messages", "review_feedback"])
    assert a is b
    assert list(a.input_fields)[:2] == ["commit_messages", "review_feedback"]
    assert with_pr_context(_Sig, []) is _Sig


def test_inputs_empty_when_no_content():
    assert pr_context_inputs(_pr_context(), PR_CONTEXT_FIELDS) == {}


def test_review_feedback_is_scoped():
    kw = pr_context_inputs(_pr_context(**FULL), ["review_feedback"], subroot="svc/a")
    assert '"T"' in kw["review_feedback"] and '"U"' not in kw["review_feedback"]
    assert pr_context_inputs(_pr_context(**FULL), ["review_feedback"], subroot="nope") == {}


def _settings():
    s = MagicMock()
    s.is_signature_enabled.return_value = True
    s.memory.enabled = False
    s.get_max_iters.return_value = 5
    s.get_max_llm_calls.return_value = 16
    s.get_rlm_threshold.return_value = 0.30
    s.review.min_confidence = 0.0
    s.review.pr_context = PRContextConfig()
    return s


def _scope():
    return ScopeResult(
        repo="o/r", subroot="svc/a", scope_type=ScopeType.SERVICE, has_changes=True,
        changed_files=[ChangedFile(filename="svc/a/main.py", status=FileStatus.MODIFIED,
                                   patch="@@ -1 +1 @@\n-a\n+b")],
        reason="test",
    )


def _run(module, cls_name, extra_patches, tmp_path, pr_kw, pf):
    captured: dict = {}

    def capture_build(sig, *a, **k):
        captured["sig"] = sig
        return MagicMock()

    async def capture_run(agent, **kwargs):
        captured["kwargs"] = kwargs
        return dspy.Prediction(issues=[])

    ctx = ReviewContext(
        pr_context=_pr_context(**pr_kw),
        metadata=ReviewMetadata(repo_path=tmp_path, run_id="r"),
        prefrontal_memory=pf,
    )
    mod = f"codespy.agents.review.{module}.agent"
    patches = [
        patch(f"{mod}.get_settings", return_value=_settings()),
        patch(f"{mod}.get_cost_tracker", return_value=MagicMock()),
        patch(f"{mod}.build_rlm_agent", side_effect=capture_build),
        patch(f"{mod}.run_rlm", side_effect=capture_run),
        patch(f"{mod}.log_rlm_outcome"),
        patch(f"{mod}.get_prefrontal", return_value=None),
        patch(f"{mod}.resolve_scope_root", return_value=tmp_path),
        *extra_patches(mod),
    ]
    for p in patches:
        p.start()
    try:
        import importlib

        cls = getattr(importlib.import_module(mod), cls_name)
        cls().forward(scopes=[_scope()], review_context=ctx)
    finally:
        for p in patches:
            p.stop()
    return captured


CASES = [
    ("doc", "DocReviewer",
     lambda mod: [patch(f"{mod}.extract_documentation", return_value="=== README.md ===\nx")],
     {"commit_messages", "pr_discussion", "review_feedback"}),
    ("code_review", "CodeReviewer",
     lambda mod: [patch(f"{mod}.CodeReviewer._create_tools", AsyncMock(return_value=([], []))),
                  patch(f"{mod}.cleanup_mcp_contexts", AsyncMock())],
     {"commit_messages", "pr_discussion", "review_feedback"}),
]


@pytest.mark.parametrize("module,cls_name,extra,expected", CASES)
@pytest.mark.parametrize("pf", ["", "m"])
def test_kwargs_match_declared_fields_full(tmp_path, module, cls_name, extra, expected, pf):
    cap = _run(module, cls_name, extra, tmp_path, FULL, pf)
    assert set(cap["kwargs"]) == set(cap["sig"].input_fields)
    assert expected <= set(cap["kwargs"])


@pytest.mark.parametrize("module,cls_name,extra,expected", CASES)
def test_no_fields_when_empty(tmp_path, module, cls_name, extra, expected):
    cap = _run(module, cls_name, extra, tmp_path, {}, "")
    assert set(cap["kwargs"]) == set(cap["sig"].input_fields)
    assert not (set(PR_CONTEXT_FIELDS) & set(cap["kwargs"]))
