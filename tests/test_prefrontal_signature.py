"""Tests for prefrontal_memory signature handling in audit and summary agents."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest

from codespy.agents.review.models import PRContext, ReviewContext


class TestSummarizerPrefrontalMemory:
    """Tests for Summarizer handling of prefrontal_memory kwarg and signature field."""

    def test_empty_prefrontal_memory_no_kwarg(self):
        """Empty prefrontal_memory: kwarg absent from call."""
        pr_context = PRContext(
            repo_slug="github.com/o/r",
            pr_number=1,
            pr_title="t",
            pr_url="https://github.com/o/r/pull/1",
            summary="s",
        )
        mock_settings = MagicMock()
        mock_settings.is_signature_enabled.return_value = True
        mock_settings.memory.enabled = False  # Disable memory to simplify
        mock_settings.get_max_iters.return_value = 5
        mock_settings.get_max_llm_calls.return_value = 16
        mock_settings.get_rlm_threshold.return_value = 0.30

        captured_kwargs = {}

        async def capture_run_rlm(agent, **kwargs):
            captured_kwargs.update(kwargs)
            return dspy.Prediction(summary="test summary")

        mock_agent = MagicMock()

        def mock_build_rlm_agent(sig, tools, *, name, max_iters, max_llm_calls, rlm_threshold):
            return mock_agent

        with patch("codespy.agents.review.summary.agent.get_settings", return_value=mock_settings):
            with patch("codespy.agents.review.summary.agent.get_cost_tracker", return_value=MagicMock()):
                with patch("codespy.agents.review.summary.agent.build_rlm_agent", side_effect=mock_build_rlm_agent):
                    with patch("codespy.agents.review.summary.agent.run_rlm", side_effect=capture_run_rlm):
                        with patch("codespy.agents.review.summary.agent.log_rlm_outcome"):
                            with patch("codespy.agents.review.summary.agent.get_prefrontal", return_value=None):
                                from codespy.agents.review.summary.agent import Summarizer

                                summarizer = Summarizer()
                                result = summarizer.forward(
                                    pr_context=pr_context,
                                    changed_file_paths=["a.py"],
                                    patches="diff",
                                    run_id="test-run",
                                    scopes=None,
                                    topics=None,
                                    prefrontal_memory="",
                                )

        assert result is not None
        # Empty string means no prefrontal_memory kwarg
        assert "prefrontal_memory" not in captured_kwargs

    def test_with_prefrontal_memory_kwarg_present(self):
        """With prefrontal_memory: kwarg equals text."""
        pr_context = PRContext(
            repo_slug="github.com/o/r",
            pr_number=1,
            pr_title="t",
            pr_url="https://github.com/o/r/pull/1",
            summary="s",
        )
        mock_settings = MagicMock()
        mock_settings.is_signature_enabled.return_value = True
        mock_settings.memory.enabled = False  # Disable memory to simplify
        mock_settings.get_max_iters.return_value = 5
        mock_settings.get_max_llm_calls.return_value = 16
        mock_settings.get_rlm_threshold.return_value = 0.30

        captured_kwargs = {}

        async def capture_run_rlm(agent, **kwargs):
            captured_kwargs.update(kwargs)
            return dspy.Prediction(summary="test summary")

        mock_agent = MagicMock()

        def mock_build_rlm_agent(sig, tools, *, name, max_iters, max_llm_calls, rlm_threshold):
            return mock_agent

        with patch("codespy.agents.review.summary.agent.get_settings", return_value=mock_settings):
            with patch("codespy.agents.review.summary.agent.get_cost_tracker", return_value=MagicMock()):
                with patch("codespy.agents.review.summary.agent.build_rlm_agent", side_effect=mock_build_rlm_agent):
                    with patch("codespy.agents.review.summary.agent.run_rlm", side_effect=capture_run_rlm):
                        with patch("codespy.agents.review.summary.agent.log_rlm_outcome"):
                            with patch("codespy.agents.review.summary.agent.get_prefrontal", return_value=None):
                                from codespy.agents.review.summary.agent import Summarizer

                                summarizer = Summarizer()
                                result = summarizer.forward(
                                    pr_context=pr_context,
                                    changed_file_paths=["a.py"],
                                    patches="diff",
                                    run_id="test-run",
                                    scopes=None,
                                    topics=None,
                                    prefrontal_memory="some recall text",
                                )

        assert result is not None
        # Verify prefrontal_memory is in call kwargs
        assert "prefrontal_memory" in captured_kwargs
        assert captured_kwargs["prefrontal_memory"] == "some recall text"

    def test_build_rlm_agent_called_with_react_threshold(self):
        """Verify build_rlm_agent is called with 'react' threshold."""
        pr_context = PRContext(
            repo_slug="github.com/o/r",
            pr_number=1,
            pr_title="t",
            pr_url="https://github.com/o/r/pull/1",
            summary="s",
        )
        mock_settings = MagicMock()
        mock_settings.is_signature_enabled.return_value = True
        mock_settings.memory.enabled = False
        mock_settings.get_max_iters.return_value = 5
        mock_settings.get_max_llm_calls.return_value = 16
        mock_settings.get_rlm_threshold.return_value = 0.30

        build_calls = []

        def capture_build_rlm_agent(sig, tools, *, name, max_iters, max_llm_calls, rlm_threshold):
            build_calls.append({
                "name": name,
                "max_iters": max_iters,
                "max_llm_calls": max_llm_calls,
                "rlm_threshold": rlm_threshold,
            })
            return MagicMock()

        async def mock_run_rlm(agent, **kwargs):
            return dspy.Prediction(summary="test summary")

        with patch("codespy.agents.review.summary.agent.get_settings", return_value=mock_settings):
            with patch("codespy.agents.review.summary.agent.get_cost_tracker", return_value=MagicMock()):
                with patch("codespy.agents.review.summary.agent.build_rlm_agent", side_effect=capture_build_rlm_agent):
                    with patch("codespy.agents.review.summary.agent.run_rlm", side_effect=mock_run_rlm):
                        with patch("codespy.agents.review.summary.agent.log_rlm_outcome"):
                            with patch("codespy.agents.review.summary.agent.get_prefrontal", return_value=None):
                                from codespy.agents.review.summary.agent import Summarizer

                                summarizer = Summarizer()
                                summarizer.forward(
                                    pr_context=pr_context,
                                    changed_file_paths=["a.py"],
                                    patches="diff",
                                    run_id="test-run",
                                    scopes=None,
                                    topics=None,
                                    prefrontal_memory="",
                                )

        assert len(build_calls) == 1
        assert build_calls[0]["name"] == "summary"
        assert build_calls[0]["max_iters"] == 5
        assert build_calls[0]["max_llm_calls"] == 16
        assert build_calls[0]["rlm_threshold"] == 0.30


class TestAuditorPrefrontalMemory:
    """Tests for Auditor handling of prefrontal_memory.

    The critical test: no UnboundLocalError when prefrontal_memory is used
    before being assigned (bug fixed by moving assignment before use).
    """

    def test_empty_prefrontal_memory_no_exception(self):
        """Empty prefrontal_memory: no exception raised (was UnboundLocalError before fix)."""
        pr_context = PRContext(
            repo_slug="github.com/o/r",
            pr_number=1,
            pr_title="t",
            pr_url="https://github.com/o/r/pull/1",
            summary="s",
        )
        review_context = ReviewContext(pr_context=pr_context, prefrontal_memory="")
        mock_settings = MagicMock()
        mock_settings.is_signature_enabled.return_value = True
        mock_settings.memory.enabled = False  # Disable memory to simplify
        mock_settings.get_max_iters.return_value = 5
        mock_settings.get_max_llm_calls.return_value = 16
        mock_settings.get_rlm_threshold.return_value = 0.30

        mock_agent = MagicMock()

        def mock_build_rlm_agent(sig, tools, *, name, max_iters, max_llm_calls, rlm_threshold):
            return mock_agent

        async def mock_run_rlm(agent, **kwargs):
            return dspy.Prediction(quality_assessment="ok", recommendation="APPROVE")

        with patch("codespy.agents.review.audit.agent.get_settings", return_value=mock_settings):
            with patch("codespy.agents.review.audit.agent.get_cost_tracker", return_value=MagicMock()):
                with patch("codespy.agents.review.audit.agent.build_rlm_agent", side_effect=mock_build_rlm_agent):
                    with patch("codespy.agents.review.audit.agent.run_rlm", side_effect=mock_run_rlm):
                        with patch("codespy.agents.review.audit.agent.log_rlm_outcome"):
                            with patch("codespy.agents.review.audit.agent.SignatureContext"):
                                with patch("codespy.agents.review.audit.agent.get_cerebral", return_value=None):
                                    with patch("codespy.agents.review.audit.agent.get_episode_store", return_value=None):
                                        with patch("codespy.agents.review.audit.agent.get_prefrontal", return_value=None):
                                            from codespy.agents.review.audit.agent import Auditor

                                            auditor = Auditor()
                                            # This would raise UnboundLocalError before the fix
                                            result = auditor.forward(
                                                review_context=review_context,
                                                all_issues=[],
                                                run_id="test-run",
                                                scopes=None,
                                                topics=None,
                                            )

        # Should return a tuple (quality_assessment, recommendation)
        assert result is not None
        assert len(result) == 2
        assert result[0] == "ok"
        assert result[1] == "APPROVE"

    def test_with_prefrontal_memory_kwarg_present(self):
        """With prefrontal_memory: kwarg passed to auditor call."""
        pr_context = PRContext(
            repo_slug="github.com/o/r",
            pr_number=1,
            pr_title="t",
            pr_url="https://github.com/o/r/pull/1",
            summary="s",
        )
        review_context = ReviewContext(pr_context=pr_context, prefrontal_memory="some text")
        mock_settings = MagicMock()
        mock_settings.is_signature_enabled.return_value = True
        mock_settings.memory.enabled = False  # Disable memory to simplify
        mock_settings.get_max_iters.return_value = 5
        mock_settings.get_max_llm_calls.return_value = 16
        mock_settings.get_rlm_threshold.return_value = 0.30

        captured_kwargs = {}

        mock_agent = MagicMock()

        def mock_build_rlm_agent(sig, tools, *, name, max_iters, max_llm_calls, rlm_threshold):
            return mock_agent

        async def capture_run_rlm(agent, **kwargs):
            captured_kwargs.update(kwargs)
            return dspy.Prediction(quality_assessment="ok", recommendation="APPROVE")

        with patch("codespy.agents.review.audit.agent.get_settings", return_value=mock_settings):
            with patch("codespy.agents.review.audit.agent.get_cost_tracker", return_value=MagicMock()):
                with patch("codespy.agents.review.audit.agent.build_rlm_agent", side_effect=mock_build_rlm_agent):
                    with patch("codespy.agents.review.audit.agent.run_rlm", side_effect=capture_run_rlm):
                        with patch("codespy.agents.review.audit.agent.log_rlm_outcome"):
                            with patch("codespy.agents.review.audit.agent.SignatureContext"):
                                with patch("codespy.agents.review.audit.agent.get_cerebral", return_value=None):
                                    with patch("codespy.agents.review.audit.agent.get_episode_store", return_value=None):
                                        with patch("codespy.agents.review.audit.agent.get_prefrontal", return_value=None):
                                            from codespy.agents.review.audit.agent import Auditor

                                            auditor = Auditor()
                                            result = auditor.forward(
                                                review_context=review_context,
                                                all_issues=[],
                                                run_id="test-run",
                                                scopes=None,
                                                topics=None,
                                            )

        assert result is not None
        # Verify prefrontal_memory is in call kwargs
        assert "prefrontal_memory" in captured_kwargs
        assert captured_kwargs["prefrontal_memory"] == "some text"

    def test_build_rlm_agent_called_with_react_threshold(self):
        """Verify build_rlm_agent is called with 'react' threshold."""
        pr_context = PRContext(
            repo_slug="github.com/o/r",
            pr_number=1,
            pr_title="t",
            pr_url="https://github.com/o/r/pull/1",
            summary="s",
        )
        review_context = ReviewContext(pr_context=pr_context, prefrontal_memory="")
        mock_settings = MagicMock()
        mock_settings.is_signature_enabled.return_value = True
        mock_settings.memory.enabled = False
        mock_settings.get_max_iters.return_value = 5
        mock_settings.get_max_llm_calls.return_value = 16
        mock_settings.get_rlm_threshold.return_value = 0.30

        build_calls = []

        def capture_build_rlm_agent(sig, tools, *, name, max_iters, max_llm_calls, rlm_threshold):
            build_calls.append({
                "name": name,
                "max_iters": max_iters,
                "max_llm_calls": max_llm_calls,
                "rlm_threshold": rlm_threshold,
            })
            return MagicMock()

        async def mock_run_rlm(agent, **kwargs):
            return dspy.Prediction(quality_assessment="ok", recommendation="APPROVE")

        with patch("codespy.agents.review.audit.agent.get_settings", return_value=mock_settings):
            with patch("codespy.agents.review.audit.agent.get_cost_tracker", return_value=MagicMock()):
                with patch("codespy.agents.review.audit.agent.build_rlm_agent", side_effect=capture_build_rlm_agent):
                    with patch("codespy.agents.review.audit.agent.run_rlm", side_effect=mock_run_rlm):
                        with patch("codespy.agents.review.audit.agent.log_rlm_outcome"):
                            with patch("codespy.agents.review.audit.agent.SignatureContext"):
                                with patch("codespy.agents.review.audit.agent.get_cerebral", return_value=None):
                                    with patch("codespy.agents.review.audit.agent.get_episode_store", return_value=None):
                                        with patch("codespy.agents.review.audit.agent.get_prefrontal", return_value=None):
                                            from codespy.agents.review.audit.agent import Auditor

                                            auditor = Auditor()
                                            auditor.forward(
                                                review_context=review_context,
                                                all_issues=[],
                                                run_id="test-run",
                                                scopes=None,
                                                topics=None,
                                            )

        assert len(build_calls) == 1
        assert build_calls[0]["name"] == "audit"
        assert build_calls[0]["max_iters"] == 5
        assert build_calls[0]["max_llm_calls"] == 16
        assert build_calls[0]["rlm_threshold"] == 0.30
