"""Filesystem MCP server: agent-caused failures reach the agent as ToolError."""

import asyncio
import logging

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import codespy.tools.storage.filesystem.server as fs_server
from codespy.tools.storage.filesystem.client import FileSystem


@pytest.fixture
def server(tmp_path, monkeypatch):
    (tmp_path / "go.mod").write_text("module x")
    monkeypatch.setattr(fs_server, "_fs", FileSystem(str(tmp_path)))
    for cached in (fs_server._list_directory_cached, fs_server._get_tree_cached,
                   fs_server._get_file_info_cached, fs_server._file_exists_cached):
        cached.cache_clear()
    fs_server._read_file_cache.clear()
    return fs_server.mcp


@pytest.mark.parametrize(
    ("tool", "args", "reason"),
    [
        ("list_directory", {"path": "deploy"}, "Directory not found: deploy"),
        ("list_directory", {"path": "go.mod"}, "Not a directory: go.mod"),
        ("get_tree", {"path": "infra"}, "Directory not found: infra"),
        ("get_file_info", {"path": "nope"}, "Path not found: nope"),
        ("list_directory", {"path": "../.."}, "Path escapes root directory"),
    ],
)
def test_expected_errors_keep_reason_without_crash_log(server, caplog, tool, args, reason):
    with caplog.at_level(logging.ERROR), pytest.raises(ToolError) as exc:
        asyncio.run(server.call_tool(tool, args))
    assert reason in str(exc.value)
    assert "unexpected exception" not in caplog.text


def test_success_unchanged(server):
    result = asyncio.run(server.call_tool("list_directory", {}))
    assert "go.mod" in str(result)
