"""Tests for episode save threading and at-exit behavior.

These tests verify that background episode saves complete before the process
exits, preventing "cannot schedule new futures after shutdown" errors from
Cerebral consolidation embedding calls.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from codespy.agents.memory.hippocampus.episode import (
    join_episode_saves,
    submit_episode_save,
)


class TestJoinEpisodeSaves:
    """Tests for join_episode_saves behavior."""

    def test_join_no_timeout_waits_for_long_thread(self) -> None:
        """join_episode_saves with no timeout should wait for slow threads."""
        result: list[str] = []

        def slow_save() -> None:
            # Sleep longer than the old default timeout (120s)
            time.sleep(0.3)
            result.append("done")

        submit_episode_save(slow_save, name="slow-save")
        join_episode_saves(timeout_per_thread=None)

        assert result == ["done"]

    def test_join_with_timeout_returns_early(self) -> None:
        """join_episode_saves with explicit timeout should return early and warn."""
        result: list[str] = []

        def slow_save() -> None:
            time.sleep(1.0)  # Longer than our timeout
            result.append("done")

        submit_episode_save(slow_save, name="slow-timeout-save")

        # Should return immediately after timeout with warning
        join_episode_saves(timeout_per_thread=0.1)

        # Result should not be set yet (thread still running)
        assert result == []

    def test_join_catches_saves_submitted_during_join(self) -> None:
        """Saves submitted during join should also be joined."""
        results: list[str] = []

        def first_save() -> None:
            # This save submits another save
            def second_save() -> None:
                time.sleep(0.05)
                results.append("second")

            submit_episode_save(second_save, name="second-save")
            time.sleep(0.1)
            results.append("first")

        submit_episode_save(first_save, name="first-save")
        join_episode_saves(timeout_per_thread=None)

        assert "first" in results
        assert "second" in results


class TestSubprocessRegression:
    """Subprocess regression test for the executor shutdown race condition."""

    def test_executor_not_shutdown_before_save_completes(self) -> None:
        """Verify that ThreadPoolExecutor works during episode save at exit.

        This test runs in a subprocess to simulate the actual interpreter exit
        scenario. It creates a daemon event loop thread (like Cerebral does),
        submits an episode save that uses run_in_executor, and exits without
        explicitly joining. The subprocess should complete successfully without
        raising "cannot schedule new futures after shutdown".
        """
        test_code = '''
import asyncio
import sys
import threading
import time

# Import episode module to trigger the at-exit hook registration
from codespy.agents.memory.hippocampus.episode import submit_episode_save

# Create a daemon event loop thread like Cerebral does
loop = asyncio.new_event_loop()

def run_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()

loop_thread = threading.Thread(target=run_loop, daemon=True)
loop_thread.start()

result = []

def save_target():
    async def async_task():
        # This uses the loop's default executor (ThreadPoolExecutor)
        # This is what litellm.aembedding does internally
        return await loop.run_in_executor(None, lambda: 42)

    # Submit to the loop and wait for result
    future = asyncio.run_coroutine_threadsafe(async_task(), loop)
    val = future.result(timeout=5.0)
    result.append(val)
    print("OK", flush=True)

submit_episode_save(save_target, name="test-save")

# Main thread exits WITHOUT calling join_episode_saves()
# The at-exit hook should ensure the save completes
'''
        # Run the test code in a subprocess
        proc = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=30,  # Overall timeout for the subprocess
        )

        stdout = proc.stdout
        stderr = proc.stderr

        # Should succeed
        assert proc.returncode == 0, (
            f"Subprocess failed with return code {proc.returncode}.\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )

        # Should print OK
        assert "OK" in stdout, f"Expected 'OK' in stdout, got: {stdout}"

        # Should NOT have the executor shutdown error
        assert "cannot schedule new futures after shutdown" not in stderr, (
            f"Got executor shutdown error: {stderr}"
        )
        assert "cannot schedule new futures after shutdown" not in stdout, (
            f"Got executor shutdown error in stdout: {stdout}"
        )


class TestThreadPoolExecutorDuringSave:
    """Tests verifying ThreadPoolExecutor behavior during episode saves."""

    def test_loop_default_executor_works_in_save(self) -> None:
        """ThreadPoolExecutor calls should work inside episode save threads."""
        results: list[int] = []

        # Create an event loop in a daemon thread (simulating Cerebral's loop)
        loop = asyncio.new_event_loop()

        def run_loop() -> None:
            asyncio.set_event_loop(loop)
            # Set up a default executor
            executor = ThreadPoolExecutor(max_workers=2)
            loop.set_default_executor(executor)
            loop.run_forever()
            # Cleanup
            executor.shutdown(wait=True)

        loop_thread = threading.Thread(target=run_loop, daemon=True)
        loop_thread.start()

        def save_with_executor() -> None:
            async def async_work() -> int:
                # This should work - uses the loop's default executor
                return await loop.run_in_executor(None, lambda: 42)

            future = asyncio.run_coroutine_threadsafe(async_work(), loop)
            val = future.result(timeout=5.0)
            results.append(val)

        submit_episode_save(save_with_executor, name="executor-save")
        join_episode_saves(timeout_per_thread=None)

        # Stop the loop
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2.0)

        assert results == [42]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
