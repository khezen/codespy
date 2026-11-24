"""Coverage of changed files by the root RLM: helper, SUBMIT gate, plumbing."""

import asyncio
import logging
from unittest.mock import MagicMock, patch

import dspy  # type: ignore[import-untyped]
import pytest
from dspy.predict.rlm import _LLMCallBudget  # type: ignore[import-untyped]
from dspy.primitives.code_interpreter import FinalOutput  # type: ignore[import-untyped]
from dspy.primitives.repl_types import REPLEntry, REPLHistory  # type: ignore[import-untyped]
from dspy.utils.dummies import DummyLM  # type: ignore[import-untyped]

from codespy.agents.context_safe import COVERAGE_REJECTION_PREFIX, CodespyRLM, ContextSafe
from codespy.agents.review.code_review.agent import (
    CodeReviewSignature,
    coverage_check,
    with_output_cap,
)
from codespy.agents.rlm_tools import build_rlm_agent, log_rlm_outcome, unseen_files
from codespy.tools.git.models import ChangedFile, FileStatus


def _patch(prefix: str, n: int) -> str:
    body = "\n".join(f"+    {prefix}_value_{i} = compute_{prefix}({i})" for i in range(n))
    return f"@@ -1,{n} +1,{n} @@\n{body}"


PATCH_A = _patch("alpha", 20)
PATCH_B = _patch("beta", 20)


def _files(**patches: str | None) -> list[ChangedFile]:
    return [
        ChangedFile(filename=f"{name}.py", status=FileStatus.MODIFIED, patch=p)
        for name, p in patches.items()
    ]


def _history(*outputs: str, code: str = "print(x)") -> REPLHistory:
    h = REPLHistory(max_output_chars=10_000)
    for out in outputs:
        h = h.append(reasoning="r", code=code, output=out)
    return h


# --------------------------------------------------------------------------
# unseen_files
# --------------------------------------------------------------------------


class TestUnseenFiles:
    def test_full_print_is_seen(self):
        files = _files(a=PATCH_A, b=PATCH_B)
        assert unseen_files(files, _history(PATCH_A), 10_000) == ["b.py"]
        assert unseen_files(files, _history(PATCH_A + "\n" + PATCH_B), 10_000) == []

    def test_patch_only_in_omitted_middle_is_unseen(self):
        filler = "x" * 6_000
        output = filler + "\n" + PATCH_A + "\n" + filler
        assert "characters omitted" in REPLEntry.format_output(output, 10_000)
        assert unseen_files(_files(a=PATCH_A), _history(output), 10_000) == ["a.py"]

    def test_larger_cap_makes_same_output_seen(self):
        filler = "x" * 6_000
        output = filler + "\n" + PATCH_A + "\n" + filler
        assert unseen_files(_files(a=PATCH_A), _history(output), 40_000) == []

    def test_slices_over_several_steps(self):
        half = len(PATCH_A) // 2
        # Cut on a line boundary so each slice holds whole lines
        cut = PATCH_A.index("\n", half)
        history = _history(PATCH_A[:cut], PATCH_A[cut:])
        assert unseen_files(_files(a=PATCH_A), history, 10_000) == []
        assert unseen_files(_files(a=PATCH_A), _history(PATCH_A[:cut]), 10_000) == ["a.py"]

    def test_no_patch_is_seen(self):
        assert unseen_files(_files(a=None, b=""), _history(), 10_000) == []

    def test_lines_pasted_in_code_are_seen(self):
        assert unseen_files(_files(a=PATCH_A), _history("ok", code=PATCH_A), 10_000) == []

    def test_threshold_90_percent(self):
        lines = PATCH_A.splitlines()[1:]  # 20 significant lines
        seen_18 = "\n".join(lines[:18])
        seen_17 = "\n".join(lines[:17])
        assert unseen_files(_files(a=PATCH_A), _history(seen_18), 10_000) == []
        assert unseen_files(_files(a=PATCH_A), _history(seen_17), 10_000) == ["a.py"]

    def test_repr_escaping_tolerated(self):
        # read_file-style dict output: the patch shows up inside a repr
        patch_q = "@@ -1 +1 @@\n" + "\n".join(f"+name_{i} = 'v{i}'" for i in range(20))
        output = repr({"content": patch_q})
        assert unseen_files(_files(a=patch_q), _history(output), 10_000) == []

    def test_short_lines_and_headers_ignored(self):
        patch_s = "@@ -1,3 +1,3 @@\n+}\n-)\n+ok"
        assert unseen_files(_files(a=patch_s), _history(), 10_000) == []

    def test_trajectory_dicts(self):
        trajectory = [{"reasoning": "r", "code": "c", "output": PATCH_A}]
        assert unseen_files(_files(a=PATCH_A, b=PATCH_B), trajectory, 10_000) == ["b.py"]


# --------------------------------------------------------------------------
# SUBMIT gate
# --------------------------------------------------------------------------


class _Sig(dspy.Signature):
    """Review."""

    changes: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ScriptedInterpreter:
    """CodeInterpreter returning scripted results keyed by code (no deno)."""

    def __init__(self, script):
        self._tools: dict = {}
        self._script = script

    @property
    def tools(self):
        return self._tools

    def start(self):
        pass

    def execute(self, code, variables=None):
        action = self._script.get(code.strip())
        if action is None:
            return None
        return action(self._tools) if callable(action) else action

    def shutdown(self):
        pass


SCRIPT = {
    "print(a)": PATCH_A,
    "print(b)": PATCH_B,
    "SUBMIT(answer='done')": FinalOutput({"answer": "done"}),
    "SUBMIT()": FinalOutput({}),
    "burn()": lambda tools: tools["llm_query"]("hello"),
}


def _actions(*codes: str) -> list[dict]:
    return [{"reasoning": "r", "code": c} for c in codes]


def _gate(files, cap=10_000):
    return coverage_check(files, cap)


def _run(rlm, answers):
    lm = DummyLM(answers)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        return rlm(changes="x", interpreter_factory=lambda: ScriptedInterpreter(SCRIPT))


def _outputs(pred) -> list[str]:
    return [e["output"] for e in pred.trajectory]


class TestSubmitGate:
    def test_pin_dspy_private_budget_attrs(self):
        # CodespyRLM._llm_budget_spent reads these private attrs (dspy 3.4.1)
        budget = _LLMCallBudget(3)
        assert (budget._count, budget._limit) == (0, 3)
        budget.reserve(3)
        assert budget._count == 3

    def test_early_submit_rejected_then_accepted_once_covered(self):
        rlm = CodespyRLM(_Sig, max_iters=5, submit_check=_gate(_files(a=PATCH_A)))
        pred = _run(rlm, _actions("SUBMIT(answer='done')", "print(a)", "SUBMIT(answer='done')"))
        assert pred.answer == "done"
        assert rlm.coverage_rejections == 1
        outputs = _outputs(pred)
        assert outputs[0].startswith(COVERAGE_REJECTION_PREFIX) and "a.py" in outputs[0]
        assert outputs[-1].startswith("FINAL")

    def test_rejected_repeatedly_no_cap(self):
        rlm = CodespyRLM(_Sig, max_iters=6, submit_check=_gate(_files(a=PATCH_A)))
        pred = _run(
            rlm,
            _actions(*["SUBMIT(answer='done')"] * 3, "print(a)", "SUBMIT(answer='done')"),
        )
        assert pred.answer == "done"
        assert rlm.coverage_rejections == 3

    def test_counter_reset_per_forward(self):
        rlm = CodespyRLM(_Sig, max_iters=5, submit_check=_gate(_files(a=PATCH_A)))
        _run(rlm, _actions("SUBMIT(answer='done')", "print(a)", "SUBMIT(answer='done')"))
        _run(rlm, _actions("print(a)", "SUBMIT(answer='done')"))
        assert rlm.coverage_rejections == 0

    def test_accepted_on_last_iteration_with_warning(self, caplog):
        rlm = CodespyRLM(_Sig, max_iters=2, submit_check=_gate(_files(a=PATCH_A, b=PATCH_B)))
        with caplog.at_level(logging.WARNING, logger="codespy.agents.context_safe"):
            pred = _run(rlm, _actions("print(a)", "SUBMIT(answer='done')"))
        assert pred.answer == "done"
        assert rlm.coverage_rejections == 0
        assert "budget spent" in caplog.text and "b.py" in caplog.text

    def test_accepted_once_llm_calls_exhausted(self):
        rlm = CodespyRLM(
            _Sig, max_iters=5, max_llm_calls=1,
            sub_lm=lambda prompt: ["ok"],
            submit_check=_gate(_files(a=PATCH_A)),
        )
        pred = _run(rlm, _actions("burn()", "SUBMIT(answer='done')"))
        assert pred.answer == "done"
        assert rlm.coverage_rejections == 0

    def test_iterations_run_out_extract_fallback(self):
        files = _files(a=PATCH_A, b=PATCH_B)
        rlm = CodespyRLM(_Sig, max_iters=2, submit_check=_gate(files))
        pred = _run(rlm, [*_actions("print(b)", "print(b)"), {"answer": "found"}])
        assert pred.answer == "found"
        assert pred.final_reasoning == "Extract forced final output"
        assert log_rlm_outcome(pred, "code_review", "s", changed_files=files,
                               max_output_chars=10_000, max_iters=2) == ["a.py"]

    def test_no_submit_check_is_stock(self):
        rlm = CodespyRLM(_Sig, max_iters=5)
        pred = _run(rlm, _actions("SUBMIT(answer='done')"))
        assert pred.answer == "done"
        assert len(pred.trajectory) == 1

    def test_invalid_fields_take_error_path_not_gate(self):
        rlm = CodespyRLM(_Sig, max_iters=5, submit_check=_gate(_files(a=PATCH_A)))
        pred = _run(rlm, _actions("SUBMIT()", "print(a)", "SUBMIT(answer='done')"))
        assert pred.answer == "done"
        assert rlm.coverage_rejections == 0
        assert _outputs(pred)[0].startswith("[Error] Missing output fields")

    def test_submit_check_failure_accepts(self):
        def boom(history):
            raise RuntimeError("x")

        rlm = CodespyRLM(_Sig, max_iters=5, submit_check=boom)
        assert _run(rlm, _actions("SUBMIT(answer='done')")).answer == "done"

    def test_async_path_gated(self):
        rlm = CodespyRLM(_Sig, max_iters=5, submit_check=_gate(_files(a=PATCH_A)))
        lm = DummyLM(_actions("SUBMIT(answer='done')", "print(a)", "SUBMIT(answer='done')"))

        async def run():
            with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
                return await rlm.acall(
                    changes="x", interpreter_factory=lambda: ScriptedInterpreter(SCRIPT)
                )

        pred = asyncio.run(run())
        assert pred.answer == "done"
        assert rlm.coverage_rejections == 1


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


class TestPlumbing:
    @patch("codespy.config.get_settings")
    def test_build_rlm_agent_passes_configured_cap(self, mock_get_settings):
        settings = MagicMock()
        settings.llm.timeout = 240.0
        settings.get_max_output_chars.return_value = 12_345
        mock_get_settings.return_value = settings
        check = _gate(_files(a=PATCH_A))

        async def build(**kw):
            return build_rlm_agent(
                _Sig, [], name="code_review", max_iters=5, max_llm_calls=8,
                rlm_threshold=0.3, **kw,
            )

        agent = asyncio.run(build(submit_check=check))
        settings.get_max_output_chars.assert_called_with("code_review")
        assert agent.module.max_output_chars == 12_345
        assert agent._max_output_chars == 12_345
        assert agent.module._submit_check is check and agent._submit_check is check

        explicit = asyncio.run(build(max_output_chars=20_000))
        assert explicit.module.max_output_chars == 20_000
        assert explicit.module._submit_check is None

    def test_fallback_gets_same_cap_and_check(self):
        check = _gate(_files(a=PATCH_A))
        cs = ContextSafe(
            dspy.Predict(_Sig), _Sig, tools=[], name="t", max_iters=2, max_llm_calls=5,
            max_output_chars=15_000, submit_check=check,
        )
        fallback = cs._create_rlm_fallback()
        assert isinstance(fallback, CodespyRLM)
        assert fallback.max_output_chars == 15_000
        assert fallback._submit_check is check

    def test_fallback_default_cap_is_dspy_default(self):
        cs = ContextSafe(dspy.Predict(_Sig), _Sig, tools=[], name="t", max_iters=2, max_llm_calls=5)
        assert cs._create_rlm_fallback().max_output_chars == 10_000


class TestCodeReviewCoverage:
    def test_prompt_cap_filled(self):
        sig = with_output_cap(CodeReviewSignature, 40_000)
        assert "{max_output_chars}" not in sig.instructions
        assert "40,000" in sig.instructions
        assert "very little time" not in sig.instructions
        assert "COVERAGE" in sig.instructions
        assert set(sig.input_fields) == set(CodeReviewSignature.input_fields)

    def test_coverage_check_message(self):
        check = coverage_check(_files(a=PATCH_A, b=PATCH_B), 40_000)
        message = check(_history(PATCH_A))
        assert message.startswith(COVERAGE_REJECTION_PREFIX)
        assert "b.py" in message and "a.py" not in message and "40,000" in message
        assert check(_history(PATCH_A, PATCH_B)) is None


class TestTelemetry:
    def test_logs_iterations_rejections_truncation_unseen(self, caplog):
        files = _files(a=PATCH_A, b=PATCH_B)
        pred = dspy.Prediction(
            answer="x",
            final_reasoning="done",
            trajectory=[
                {"reasoning": "r", "code": "c", "output": f"{COVERAGE_REJECTION_PREFIX} Not reviewed"},
                {"reasoning": "r", "code": "c", "output": "y" * 20_000},
                {"reasoning": "r", "code": "c", "output": PATCH_A},
            ],
        )
        with caplog.at_level(logging.INFO, logger="codespy.agents.rlm_tools"):
            unseen = log_rlm_outcome(
                pred, "code_review", "svc", changed_files=files, max_output_chars=10_000, max_iters=5
            )
        assert unseen == ["b.py"]
        text = caplog.text
        assert "iterations=3/5" in text and "end=SUBMIT" in text
        assert "coverage_rejections=1" in text and "truncated_outputs=1" in text
        assert "unseen_files=1: b.py" in text

    def test_no_telemetry_without_coverage_args(self, caplog):
        pred = dspy.Prediction(answer="x", final_reasoning="done", trajectory=[])
        with caplog.at_level(logging.INFO, logger="codespy.agents.rlm_tools"):
            assert log_rlm_outcome(pred, "doc", "svc") == []
        assert "iterations=" not in caplog.text


@pytest.mark.parametrize("reason,expected", [
    ("Extract forced final output", "extract"),
    ("Extract after content_filter refusal", "refusal"),
    ("anything", "SUBMIT"),
])
def test_end_reason(reason, expected, caplog):
    pred = dspy.Prediction(answer="x", final_reasoning=reason, trajectory=[])
    with caplog.at_level(logging.INFO, logger="codespy.agents.rlm_tools"):
        log_rlm_outcome(pred, "code_review", "svc", changed_files=[], max_output_chars=10_000)
    assert f"end={expected}" in caplog.text
