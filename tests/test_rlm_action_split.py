"""RLM action steps are split locally so the extraction model can't pollute `reasoning`."""

import asyncio
from typing import Any

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from codespy.agents.diagnostic_adapter import (
    NO_CODE_FEEDBACK,
    DiagnosticTwoStepAdapter,
    split_rlm_action,
)

COMPLETION = "Inspect changed files first.\n\n```python\nprint(len(scope.changed_files))\n```"

ACTION_SIG = dspy.Signature("variables_info, repl_history, iteration -> reasoning, code")


class ExplodingLM(DummyLM):
    """Extraction LM that fails the test if it is called."""

    def __init__(self) -> None:
        super().__init__([{"reasoning": "unused", "code": "unused"}])

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("extraction model must not be called for RLM action steps")

    async def aforward(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("extraction model must not be called for RLM action steps")


class TextLM(DummyLM):
    """Main LM returning a fixed free-form completion."""

    def __init__(self, text: str) -> None:
        super().__init__([{"x": "x"}])
        self._text = text

    def __call__(self, *args: Any, **kwargs: Any) -> list[str]:
        return [self._text]

    async def acall(self, *args: Any, **kwargs: Any) -> list[str]:
        return [self._text]


class TestSplitRlmAction:
    def test_plan_and_code(self) -> None:
        assert split_rlm_action(COMPLETION) == {
            "reasoning": "Inspect changed files first.",
            "code": "```python\nprint(len(scope.changed_files))\n```",
        }

    def test_code_only(self) -> None:
        assert split_rlm_action("```python\nprint(1)\n```") == {
            "reasoning": "",
            "code": "```python\nprint(1)\n```",
        }

    def test_no_fence_bare_python_is_code(self) -> None:
        assert split_rlm_action("print(1)") == {"reasoning": "", "code": "print(1)"}

    def test_empty_defers_to_extraction(self) -> None:
        assert split_rlm_action("  ") is None

    def test_think_preamble_dropped(self) -> None:
        out = split_rlm_action("I should look.\n</think>\nPlan.\n```python\nprint(2)\n```")
        assert out == {"reasoning": "Plan.", "code": "```python\nprint(2)\n```"}

    def test_json_action(self) -> None:
        text = '{\n  "reasoning": "Look first.",\n  "code": "print(\\"x\\")"\n}'
        assert split_rlm_action(text) == {"reasoning": "Look first.", "code": 'print("x")'}

    def test_json_without_code_is_prose(self) -> None:
        out = split_rlm_action('{"summary": "done"}')
        assert out is not None and NO_CODE_FEEDBACK in out["code"]

    def test_prose_answer_gets_feedback(self) -> None:
        # nemotron-super on mono#10519: the summary in prose, no code, no SUBMIT
        prose = "The PR removes a filter that excluded draft requests. No submit."
        out = split_rlm_action(prose)
        assert out["reasoning"] == prose
        assert out["code"] == f"print({NO_CODE_FEEDBACK!r})"
        compile(out["code"], "<t>", "exec")

    def test_lone_word_is_prose(self) -> None:
        assert NO_CODE_FEEDBACK in split_rlm_action("Done")["code"]

    @pytest.mark.parametrize("text", ["```", "Plan.\n```python\n```", "```python\n\n```\n```"])
    def test_empty_fence_gets_feedback(self, text: str) -> None:
        assert NO_CODE_FEEDBACK in split_rlm_action(text)["code"]

    def test_second_empty_fence_kept(self) -> None:
        # nemotron sometimes appends an empty block; dspy runs the first one
        out = split_rlm_action("```python\nprint(1)\n```\n```python\n```")
        assert out["code"].startswith("```python\nprint(1)")


@pytest.fixture
def adapter() -> DiagnosticTwoStepAdapter:
    return DiagnosticTwoStepAdapter(ExplodingLM())


INPUTS = {"variables_info": "v", "repl_history": "", "iteration": "1/5"}


class TestAdapterRouting:
    def test_sync_rlm_action_skips_extraction(self, adapter: DiagnosticTwoStepAdapter) -> None:
        out = adapter(TextLM(COMPLETION), {}, ACTION_SIG, [], INPUTS)
        assert out[0]["reasoning"] == "Inspect changed files first."
        assert out[0]["code"].startswith("```python")

    def test_async_rlm_action_skips_extraction(self, adapter: DiagnosticTwoStepAdapter) -> None:
        out = asyncio.run(adapter.acall(TextLM(COMPLETION), {}, ACTION_SIG, [], INPUTS))
        assert out[0]["reasoning"] == "Inspect changed files first."
        assert out[0]["code"].startswith("```python")

    def test_other_signatures_still_use_extraction(self, adapter: DiagnosticTwoStepAdapter) -> None:
        sig = dspy.Signature("question -> answer")
        with pytest.raises(AssertionError, match="extraction model must not be called"):
            adapter(TextLM("42"), {}, sig, [], {"question": "q"})
