"""Filesystem MCP server: agent-caused failures reach the agent as ToolError."""

import asyncio
import logging

import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

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


def test_invalid_path_is_readable_tool_error(server, caplog):
    """A NUL byte makes resolve() fail: the agent gets the reason, not a crash."""
    with caplog.at_level(logging.ERROR), pytest.raises(ToolError) as exc:
        asyncio.run(server.call_tool("list_directory", {"path": "a\x00b"}))
    assert "Invalid path" in str(exc.value)
    assert not isinstance(exc.value, UnexpectedToolError)
    assert "unexpected exception" not in caplog.text


def test_plain_value_error_stays_a_crash(server, monkeypatch):
    """Only InvalidPathError is mapped: other ValueErrors are bugs, not agent mistakes."""

    def boom(*args):
        raise ValueError("boom")

    monkeypatch.setattr(fs_server, "_list_directory_cached", boom)
    with pytest.raises(UnexpectedToolError) as exc:
        asyncio.run(server.call_tool("list_directory", {"path": ""}))
    assert "boom" not in str(exc.value)  # the reason is withheld from the agent
    assert isinstance(exc.value.__cause__, ValueError)


def test_invalid_path_error_is_a_value_error(tmp_path):
    """Existing `except ValueError` callers (exists, write_file...) keep working."""
    from codespy.tools.storage.filesystem.client import InvalidPathError

    assert issubclass(InvalidPathError, ValueError)
    fs = FileSystem(str(tmp_path))
    assert fs.exists("../..") is False
    assert fs.exists("a\x00b") is False
    assert fs.write_file("../x", "c").success is False
    with pytest.raises(InvalidPathError, match="escapes root"):
        fs.resolve_path("../..")


def test_success_unchanged(server):
    result = asyncio.run(server.call_tool("list_directory", {}))
    assert "go.mod" in str(result)


# --- write tools (--writable) -------------------------------------------------------


def _tool_names(server) -> set[str]:
    return {t.name for t in asyncio.run(server.list_tools())}


def test_write_tools_absent_by_default():
    """Read-only unless --writable: importing the server registers no write tool."""
    import subprocess
    import sys

    code = (
        "import asyncio, codespy.tools.storage.filesystem.server as s;"
        "print(sorted(t.name for t in asyncio.run(s.mcp.list_tools())))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    for name in ("write_file", "replace_in_file", "delete_file"):
        assert name not in out.stdout
    assert "read_file" in out.stdout


@pytest.fixture
def writable(server):
    fs_server.register_write_tools()
    return server


def _call(server, tool, args):
    return asyncio.run(server.call_tool(tool, args))


def test_write_replace_delete(writable, tmp_path):
    assert {"write_file", "replace_in_file", "delete_file"} <= _tool_names(writable)
    _call(writable, "write_file", {"path": "pkg/new.py", "content": "a = 1\n"})
    assert (tmp_path / "pkg/new.py").read_text() == "a = 1\n"
    _call(writable, "replace_in_file", {"path": "pkg/new.py", "old": "a = 1", "new": "a = 2"})
    assert (tmp_path / "pkg/new.py").read_text() == "a = 2\n"
    _call(writable, "delete_file", {"path": "pkg/new.py"})
    assert not (tmp_path / "pkg/new.py").exists()


@pytest.mark.parametrize(("content", "reason"), [("x\n", "not found"), ("a\na\n", "2 times")])
def test_replace_needs_exactly_one_match(writable, tmp_path, content, reason):
    (tmp_path / "f.txt").write_text(content)
    with pytest.raises(ToolError, match=reason):
        _call(writable, "replace_in_file", {"path": "f.txt", "old": "a", "new": "b"})


@pytest.mark.parametrize(
    "path",
    [".git/config", "sub/.git/HEAD", ".github/workflows/ci.yml", ".codespy/reviews/x.md", "../x"],
)
def test_protected_and_escaping_paths_refused(writable, tmp_path, path):
    with pytest.raises(ToolError):
        _call(writable, "write_file", {"path": path, "content": "x"})
    assert not (tmp_path / ".git").exists()


def test_symlinks_refused(writable, tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    (tmp_path / "flink").symlink_to(tmp_path / "go.mod")
    with pytest.raises(ToolError, match="symlink"):
        _call(writable, "write_file", {"path": "link/a.txt", "content": "x"})
    with pytest.raises(ToolError, match="symlink"):
        _call(writable, "delete_file", {"path": "flink"})
    assert (tmp_path / "go.mod").exists()


def test_reads_after_write_see_new_content(writable):
    assert "module x" in str(_call(writable, "read_file", {"path": "go.mod"}))
    assert "new.txt" not in str(_call(writable, "list_directory", {}))
    _call(writable, "write_file", {"path": "go.mod", "content": "module y"})
    _call(writable, "write_file", {"path": "new.txt", "content": "n"})
    assert "module y" in str(_call(writable, "read_file", {"path": "go.mod"}))
    assert "new.txt" in str(_call(writable, "list_directory", {}))
