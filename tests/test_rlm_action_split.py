"""RLM action steps are split locally so the extraction model can't pollute `reasoning`."""

import asyncio
from typing import Any

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from codespy.agents.diagnostic_adapter import DiagnosticTwoStepAdapter, split_rlm_action

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

    def test_no_fence_defers_to_extraction(self) -> None:
        assert split_rlm_action("print(1)") is None


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
