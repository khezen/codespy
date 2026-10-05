"""Tests for RLM tool bridge functionality."""

import asyncio
import shutil
import threading
from unittest.mock import MagicMock, patch

import pytest

from codespy.agents.rlm_tools import (
    _is_coroutine_function,
    bridge_tools_to_loop,
    build_rlm_agent,
    log_rlm_outcome,
    run_rlm,
)


# Skip tests that require Deno if it's not installed
requires_deno = pytest.mark.skipif(
    shutil.which("deno") is None,
    reason="Deno not installed"
)


class TestIsCoroutineFunction:
    """Test the _is_coroutine_function helper."""

    def test_sync_function_returns_false(self):
        """Sync functions should return False."""
        def sync_func():
            return "sync"
        assert _is_coroutine_function(sync_func) is False

    def test_async_function_returns_true(self):
        """Async functions should return True."""
        async def async_func():
            return "async"
        assert _is_coroutine_function(async_func) is True

    def test_dspy_tool_with_async_func(self):
        """dspy.Tool wrapping async func should be detected."""
        try:
            import dspy
            async def async_func():
                return "async"
            tool = dspy.Tool(async_func)
            assert _is_coroutine_function(tool) is True
        except ImportError:
            pytest.skip("dspy not installed")


class TestBridgeToolsToLoop:
    """Test the bridge_tools_to_loop function."""

    def test_sync_tools_pass_through(self):
        """Sync tools should be passed through unchanged."""
        def sync_tool(x: int) -> int:
            """Sync tool docstring."""
            return x * 2

        loop = asyncio.new_event_loop()
        try:
            bridged = bridge_tools_to_loop([sync_tool], loop, timeout=30.0)
            assert len(bridged) == 1
            # The tool should be wrapped in dspy.Tool but remain sync
            result = bridged[0].func(x=5)
            assert result == 10
        finally:
            loop.close()

    def test_async_tools_get_sync_wrapper(self):
        """Async tools should get a sync wrapper that schedules on the loop."""
        async def async_echo(x: int) -> int:
            """Echo async tool."""
            return x

        async def test_in_loop():
            loop = asyncio.get_running_loop()
            bridged = bridge_tools_to_loop([async_echo], loop, timeout=30.0)
            assert len(bridged) == 1

            # The wrapper should be sync
            bridged_tool = bridged[0]
            assert not asyncio.iscoroutinefunction(bridged_tool.func)

            # Calling on a different thread should work
            def call_tool():
                return bridged_tool.func(x=42)

            # Use to_thread to run off the event loop
            result = await asyncio.to_thread(call_tool)
            assert result == 42

        asyncio.run(test_in_loop())

    def test_bridge_preserves_tool_metadata(self):
        """Bridging should preserve name, desc, args, arg_types, arg_desc."""
        async def async_echo(x: int) -> str:
            """Echo the input."""
            return str(x)

        try:
            import dspy
            tool = dspy.Tool(
                async_echo,
                name="custom_echo",
                desc="Custom echo tool",
                args={"x": "integer"},
                arg_types={"x": int},
                arg_desc={"x": "The number to echo"},
            )

            async def test_in_loop():
                loop = asyncio.get_running_loop()
                bridged = bridge_tools_to_loop([tool], loop, timeout=30.0)
                assert len(bridged) == 1

                bridged_tool = bridged[0]
                assert bridged_tool.name == "custom_echo"
                assert bridged_tool.desc == "Custom echo tool"
                assert bridged_tool.args == {"x": "integer"}
                assert bridged_tool.arg_types == {"x": int}
                assert bridged_tool.arg_desc == {"x": "The number to echo"}

            asyncio.run(test_in_loop())
        except ImportError:
            pytest.skip("dspy not installed")

    def test_bridge_deadlock_guard(self):
        """Calling bridged tool on the bridge thread should raise RuntimeError."""
        async def async_echo(x: int) -> int:
            return x

        async def test_in_loop():
            loop = asyncio.get_running_loop()
            bridged = bridge_tools_to_loop([async_echo], loop, timeout=30.0)
            bridged_tool = bridged[0]

            # Calling directly on the event loop (same thread as bridge)
            # should raise RuntimeError
            with pytest.raises(RuntimeError, match="RLM tools must run off the event loop"):
                bridged_tool.func(x=1)

        asyncio.run(test_in_loop())

    def test_bridge_timeout(self):
        """Slow tools should timeout and cancel the future."""
        async def slow_tool():
            await asyncio.sleep(10)  # Very slow
            return "done"

        async def test_in_loop():
            loop = asyncio.get_running_loop()
            bridged = bridge_tools_to_loop([slow_tool], loop, timeout=0.05)
            bridged_tool = bridged[0]

            def call_slow():
                return bridged_tool.func()

            # Should timeout
            with pytest.raises(TimeoutError, match="slow_tool"):
                await asyncio.to_thread(call_slow)

        asyncio.run(test_in_loop())


class TestBuildRlmAgent:
    """Test the build_rlm_agent function."""

    @patch("codespy.config.get_settings")
    def test_build_rlm_agent_creates_contextsafe(self, mock_get_settings):
        """build_rlm_agent should return a ContextSafe-wrapped RLM."""
        mock_settings = MagicMock()
        mock_settings.llm.timeout = 240.0
        mock_get_settings.return_value = mock_settings

        try:
            import dspy

            class TestSignature(dspy.Signature):
                """Test signature."""
                question: str = dspy.InputField()
                answer: str = dspy.OutputField()

            async def test_in_loop():
                agent = build_rlm_agent(
                    TestSignature,
                    [],
                    name="test_agent",
                    max_iters=5,
                    max_llm_calls=8,
                    rlm_threshold=0.3,
                )
                # Should be a ContextSafe instance
                from codespy.agents.context_safe import ContextSafe
                assert isinstance(agent, ContextSafe)
                assert agent._name == "test_agent"
                assert agent._max_iters == 5
                assert agent._max_llm_calls == 8
                assert agent._rlm_threshold == 0.3

            asyncio.run(test_in_loop())
        except ImportError:
            pytest.skip("dspy not installed")


class TestRunRlm:
    """Test the run_rlm function."""

    @patch("codespy.config.get_settings")
    def test_run_rlm_calls_agent_off_loop(self, mock_get_settings):
        """run_rlm should execute the agent off the event loop."""
        mock_settings = MagicMock()
        mock_settings.llm.timeout = 240.0
        mock_get_settings.return_value = mock_settings

        try:
            import dspy
            from codespy.agents.context_safe import ContextSafe

            class TestSignature(dspy.Signature):
                """Test signature."""
                question: str = dspy.InputField()
                answer: str = dspy.OutputField()

            # Create a mock agent
            mock_agent = MagicMock(spec=ContextSafe)
            mock_prediction = MagicMock()
            mock_prediction.answer = "test answer"
            mock_agent.return_value = mock_prediction

            async def test_in_loop():
                result = await run_rlm(mock_agent, question="test")
                assert result.answer == "test answer"
                # Verify the agent was called (mock_agent is called as a function via __call__)
                assert mock_agent.called
                # Check the call args
                call_args = mock_agent.call_args
                assert call_args.kwargs.get("question") == "test"

            asyncio.run(test_in_loop())
        except ImportError:
            pytest.skip("dspy not installed")


class TestLogRlmOutcome:
    """Test the log_rlm_outcome function."""

    def test_log_rlm_outcome_no_op_without_extract_reasoning(self):
        """Should not log when final_reasoning is not extract fallback."""
        mock_result = MagicMock()
        mock_result.final_reasoning = "Natural completion"

        # Should not raise and should not log
        log_rlm_outcome(mock_result, "test_module", "test_scope")

    def test_log_rlm_outcome_logs_errors(self, caplog):
        """Should log warning when extract fallback has tool errors."""
        mock_result = MagicMock()
        mock_result.final_reasoning = "Extract forced final output"

        # Create mock trajectory with errors
        entry_with_error = MagicMock()
        entry_with_error.output = "Error: something went wrong"

        entry_ok = MagicMock()
        entry_ok.output = "Normal output"

        mock_result.trajectory = [entry_ok, entry_with_error, entry_with_error]

        with caplog.at_level("WARNING"):
            log_rlm_outcome(mock_result, "code_review", "scope1")

        assert "RLM[code_review] scope scope1" in caplog.text
        assert "2 tool errors" in caplog.text
        # The log message doesn't contain "Extract forced final output", it contains the diagnostic info
        assert "extract fallback" in caplog.text

    def test_log_rlm_outcome_dict_trajectory(self, caplog):
        """Should handle trajectory entries as dicts."""
        mock_result = MagicMock()
        mock_result.final_reasoning = "Extract forced final output"
        mock_result.trajectory = [
            {"reasoning": "r1", "code": "c1", "output": "Normal"},
            {"reasoning": "r2", "code": "c2", "output": "Error: timeout"},
        ]

        with caplog.at_level("WARNING"):
            log_rlm_outcome(mock_result, "supply_chain", "scope2")

        assert "1 tool errors" in caplog.text


@requires_deno
class TestDenoIntegration:
    """Integration tests that require Deno to be installed."""

    @pytest.mark.skip(reason="Requires full DSPy + MCP setup - run manually")
    def test_rlm_with_bridged_mcp_tool(self):
        """Full integration: RLM agent with bridged MCP tool via Deno.

        This test would require:
        1. A running MCP server via Deno
        2. DSPy RLM with bridged tools
        3. Executing on the event loop thread

        Run manually when Deno is available.
        """
        pass
