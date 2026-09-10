"""T5 内置工具测试。

checklist 对应项：
- bash：Windows 走 PowerShell（反证：不直调 bash/cmd）
- read：中文 UTF-8 无乱码
- edit：目标出现 2 次时报错（含次数与唯一性提示）；恰好 1 次时替换成功且 diff 含新旧
- bash 输出截断：超限内容含截断标记
- grep/find/ls：忽略规则（.gitignore 命中的文件不出现）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mimcode.agent.tools import ToolRegistry, builtin_tools
from mimcode.agent.tools.bash import _resolve_shell
from mimcode.agent.tools.edit import Edit, apply_edits
from mimcode.agent.tools.truncate import truncate_head


def make_registry(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(cwd=str(tmp_path))


async def run_tool(registry: ToolRegistry, name: str, args: dict):
    tool = registry.get(name)
    assert tool is not None
    return await tool.execute("test-call", args)


# ---------------------------------------------------------------------------
# bash / shell
# ---------------------------------------------------------------------------


def test_shell_resolution_windows_powershell() -> None:
    """shell 解析：Windows → powershell（非 bash/cmd），其余 → bash。"""
    if sys.platform == "win32":
        executable, prefix = _resolve_shell()
        assert executable == "powershell"
        assert prefix == ["-NoProfile", "-Command"]
    else:
        executable, prefix = _resolve_shell()
        assert executable == "bash"
        assert prefix == ["-c"]


async def test_bash_echo(tmp_path: Path) -> None:
    """echo 回显 + 退出码 0。"""
    registry = make_registry(tmp_path)
    result = await run_tool(registry, "bash", {"command": "echo hello"})
    assert "hello" in result.content[0].text
    assert result.details["exit_code"] == 0


async def test_bash_nonzero_exit_code(tmp_path: Path) -> None:
    """非零退出码：输出含标记。"""
    if sys.platform == "win32":
        command = "exit 3"
    else:
        command = "exit 3"
    registry = make_registry(tmp_path)
    result = await run_tool(registry, "bash", {"command": command})
    assert "[Exit code: 3]" in result.content[0].text


async def test_bash_output_truncation(tmp_path: Path) -> None:
    """输出超行数上限：含截断标记且总行数受限（checklist）。"""
    if sys.platform == "win32":
        command = '0..4999 | ForEach-Object { "line$_" }'
    else:
        command = "for i in $(seq 0 4999); do echo line$i; done"
    registry = make_registry(tmp_path)
    result = await run_tool(registry, "bash", {"command": command})
    text = result.content[0].text
    assert "[Output truncated:" in text
    assert "5000" in text  # 总行数信息
    # 截断后不超过上限 + 标记
    content_lines = text.splitlines()
    assert len(content_lines) <= 2000 + 3  # 2000 行 + 退出码/截断标记行


async def test_bash_timeout(tmp_path: Path) -> None:
    """超时：进程终止并输出超时标记。"""
    if sys.platform == "win32":
        command = "Start-Sleep -Seconds 30"
    else:
        command = "sleep 30"
    registry = make_registry(tmp_path)
    result = await run_tool(registry, "bash", {"command": command, "timeout": 1})
    assert "[Timed out after 1s]" in result.content[0].text


# ---------------------------------------------------------------------------
# read / write
# ---------------------------------------------------------------------------


async def test_read_utf8_chinese(tmp_path: Path) -> None:
    """中文 UTF-8 读回字节相等（checklist：无乱码）。"""
    target = tmp_path / "中文文件.txt"
    target.write_text("你好，世界\n第二行", encoding="utf-8")
    registry = make_registry(tmp_path)
    result = await run_tool(registry, "read", {"path": str(target)})
    assert "你好，世界" in result.content[0].text
    assert "第二行" in result.content[0].text


async def test_read_relative_path(tmp_path: Path) -> None:
    """相对路径以 cwd 解析。"""
    (tmp_path / "rel.txt").write_text("relative", encoding="utf-8")
    registry = make_registry(tmp_path)
    result = await run_tool(registry, "read", {"path": "rel.txt"})
    assert "relative" in result.content[0].text


async def test_read_offset_limit(tmp_path: Path) -> None:
    """offset/limit 窗口读取 + 继续提示。"""
    target = tmp_path / "lines.txt"
    target.write_text("\n".join(f"line{i}" for i in range(1, 101)), encoding="utf-8")
    registry = make_registry(tmp_path)
    result = await run_tool(registry, "read", {"path": str(target), "offset": 10, "limit": 5})
    text = result.content[0].text
    assert "line10" in text
    assert "line14" in text
    assert "line15" not in text
    assert "Use offset=15 to continue" in text


async def test_read_offset_out_of_bounds(tmp_path: Path) -> None:
    """offset 超出文件长度：报错。"""
    target = tmp_path / "short.txt"
    target.write_text("one", encoding="utf-8")
    registry = make_registry(tmp_path)
    with pytest.raises(Exception, match="beyond end of file"):
        await run_tool(registry, "read", {"path": str(target), "offset": 99})


async def test_read_binary_rejected(tmp_path: Path) -> None:
    """二进制文件（含 NUL）被拒绝。"""
    target = tmp_path / "bin.dat"
    target.write_bytes(b"abc\x00def")
    registry = make_registry(tmp_path)
    with pytest.raises(Exception, match="binary"):
        await run_tool(registry, "read", {"path": str(target)})


async def test_write_creates_and_overwrites(tmp_path: Path) -> None:
    """写入：创建新文件（含父目录）与覆写。"""
    registry = make_registry(tmp_path)
    nested = tmp_path / "a" / "b" / "new.txt"
    result = await run_tool(registry, "write", {"path": str(nested), "content": "v1"})
    assert "created" in result.content[0].text
    assert nested.read_text(encoding="utf-8") == "v1"

    result = await run_tool(registry, "write", {"path": str(nested), "content": "v2"})
    assert "overwritten" in result.content[0].text
    assert nested.read_text(encoding="utf-8") == "v2"


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


async def test_edit_unique_replacement_with_diff(tmp_path: Path) -> None:
    """唯一匹配替换成功；diff 含删除行与新增行（checklist）。"""
    target = tmp_path / "code.py"
    target.write_text("def foo():\n    return 1\n", encoding="utf-8")
    registry = make_registry(tmp_path)
    result = await run_tool(
        registry,
        "edit",
        {
            "path": str(target),
            "edits": [{"oldText": "return 1", "newText": "return 2"}],
        },
    )
    assert "Successfully replaced 1 block(s)" in result.content[0].text
    assert target.read_text(encoding="utf-8") == "def foo():\n    return 2\n"
    diff = result.details["diff"]
    assert "-    return 1" in diff
    assert "+    return 2" in diff


async def test_edit_duplicate_occurrences_error(tmp_path: Path) -> None:
    """目标出现 2 次：报错含出现次数与唯一性提示（checklist）。"""
    target = tmp_path / "dup.txt"
    target.write_text("same\nsame\n", encoding="utf-8")
    registry = make_registry(tmp_path)
    with pytest.raises(Exception) as exc_info:
        await run_tool(
            registry,
            "edit",
            {"path": str(target), "edits": [{"oldText": "same", "newText": "other"}]},
        )
    message = str(exc_info.value)
    assert "2 times" in message
    assert "unique" in message
    # 文件未被修改
    assert target.read_text(encoding="utf-8") == "same\nsame\n"


async def test_edit_multiple_disjoint_edits(tmp_path: Path) -> None:
    """多编辑一次应用：各自对原文匹配。"""
    target = tmp_path / "multi.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    registry = make_registry(tmp_path)
    result = await run_tool(
        registry,
        "edit",
        {
            "path": str(target),
            "edits": [
                {"oldText": "alpha", "newText": "ALPHA"},
                {"oldText": "gamma", "newText": "GAMMA"},
            ],
        },
    )
    assert "Successfully replaced 2 block(s)" in result.content[0].text
    assert target.read_text(encoding="utf-8") == "ALPHA\nbeta\nGAMMA\n"


async def test_edit_overlap_rejected() -> None:
    """重叠编辑被拒绝。"""
    with pytest.raises(Exception, match="overlap"):
        apply_edits(
            "abc def",
            [Edit(old_text="abc", new_text="x"), Edit(old_text="c def", new_text="y")],
            "f",
        )


async def test_edit_not_found() -> None:
    """未命中报错。"""
    with pytest.raises(Exception, match="not found"):
        apply_edits("content", [Edit(old_text="missing", new_text="x")], "f")


async def test_edit_no_change_rejected() -> None:
    """无变化报错。"""
    with pytest.raises(Exception, match="not changed|no change"):
        apply_edits("content", [Edit(old_text="content", new_text="content")], "f")


async def test_edit_empty_old_text_rejected() -> None:
    """空 oldText 报错。"""
    with pytest.raises(Exception, match="empty"):
        apply_edits("content", [Edit(old_text="", new_text="x")], "f")


async def test_edit_crlf_preserved(tmp_path: Path) -> None:
    """CRLF 文件：匹配在 LF 空间，写回保留 CRLF。"""
    target = tmp_path / "crlf.txt"
    target.write_bytes(b"line one\r\nline two\r\n")
    registry = make_registry(tmp_path)
    await run_tool(
        registry,
        "edit",
        {"path": str(target), "edits": [{"oldText": "line one\n", "newText": "first\n"}]},
    )
    assert target.read_bytes() == b"first\r\nline two\r\n"


async def test_edit_preserves_bom(tmp_path: Path) -> None:
    """BOM 文件：匹配前剥离，写回保留。"""
    target = tmp_path / "bom.txt"
    target.write_bytes("\ufeffcontent here\n".encode("utf-8"))
    registry = make_registry(tmp_path)
    await run_tool(
        registry,
        "edit",
        {"path": str(target), "edits": [{"oldText": "content", "newText": "changed"}]},
    )
    assert target.read_bytes() == "\ufeffchanged here\n".encode("utf-8")


async def test_edit_json_string_edits_form(tmp_path: Path) -> None:
    """edits 以 JSON 字符串形态传入（模型容错，对齐 pi）。"""
    target = tmp_path / "json.txt"
    target.write_text("hello\n", encoding="utf-8")
    registry = make_registry(tmp_path)
    import json

    result = await run_tool(
        registry,
        "edit",
        {
            "path": str(target),
            "edits": json.dumps([{"oldText": "hello", "newText": "world"}]),
        },
    )
    assert "Successfully replaced" in result.content[0].text
    assert target.read_text(encoding="utf-8") == "world\n"


# ---------------------------------------------------------------------------
# ls / find / grep（含忽略规则）
# ---------------------------------------------------------------------------


@pytest.fixture
def ignored_tree(tmp_path: Path) -> Path:
    """构造含 .gitignore 的目录树。"""
    (tmp_path / ".gitignore").write_text("secret*\nbuild/\n", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("hidden", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.txt").write_text("artifact", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "keep.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


async def test_ls_respects_gitignore(ignored_tree: Path) -> None:
    """ls：忽略命中的条目不出现。"""
    registry = make_registry(ignored_tree)
    result = await run_tool(registry, "ls", {"path": str(ignored_tree)})
    text = result.content[0].text
    assert "keep.txt" in text
    assert "sub/" in text
    assert "secret.txt" not in text
    assert "build/" not in text


async def test_find_respects_gitignore(ignored_tree: Path) -> None:
    """find：忽略命中的文件不出现在结果（checklist）。"""
    registry = make_registry(ignored_tree)
    result = await run_tool(registry, "find", {"pattern": "*.txt"})
    text = result.content[0].text
    assert "keep.txt" in text
    assert "secret.txt" not in text
    assert "build" not in text


async def test_find_recursive_glob(ignored_tree: Path) -> None:
    """find 递归命中子目录文件。"""
    registry = make_registry(ignored_tree)
    result = await run_tool(registry, "find", {"pattern": "*.py"})
    assert "sub/keep.py" in result.content[0].text


async def test_grep_respects_gitignore(ignored_tree: Path) -> None:
    """grep：忽略命中的文件不被搜索。"""
    (ignored_tree / "secret.txt").write_text("needle\n", encoding="utf-8")
    (ignored_tree / "keep.txt").write_text("needle here\n", encoding="utf-8")
    registry = make_registry(ignored_tree)
    result = await run_tool(registry, "grep", {"pattern": "needle"})
    text = result.content[0].text
    assert "keep.txt:1" in text
    assert "secret.txt" not in text


async def test_grep_regex_and_case(tmp_path: Path) -> None:
    """grep：正则与大小写开关。"""
    (tmp_path / "a.txt").write_text("Foo bar\nfoo baz\n", encoding="utf-8")
    registry = make_registry(tmp_path)
    result = await run_tool(registry, "grep", {"pattern": "^Foo"})
    assert "a.txt:1" in result.content[0].text
    assert "a.txt:2" not in result.content[0].text

    result = await run_tool(registry, "grep", {"pattern": "^foo", "ignore_case": True})
    assert "a.txt:1" in result.content[0].text


async def test_ggrep_long_line_truncated(tmp_path: Path) -> None:
    """超长匹配行截断到上限字符（checklist：500 字符）。"""
    long_line = "x" * 800
    (tmp_path / "long.txt").write_text(long_line + "\n", encoding="utf-8")
    registry = make_registry(tmp_path)
    result = await run_tool(registry, "grep", {"pattern": "x"})
    match_line = result.content[0].text.splitlines()[0]
    # 输出行 = 文件名:行号:内容（内容截到 500 + 省略号）
    content_part = match_line.split(":", 2)[2]
    assert len(content_part) == 501  # 500 字符 + 省略号


# ---------------------------------------------------------------------------
# 注册表与协议
# ---------------------------------------------------------------------------


def test_builtin_tools_seven(tmp_path: Path) -> None:
    """七件套齐全且名字正确。"""
    tools = builtin_tools(str(tmp_path))
    names = [tool.name for tool in tools]
    assert names == ["bash", "read", "write", "edit", "ls", "find", "grep"]


def test_sequential_tools_declared(tmp_path: Path) -> None:
    """文件写类工具声明 sequential（bash/write/edit），其余 parallel。"""
    registry = make_registry(tmp_path)
    assert registry.get("bash").execution_mode == "sequential"
    assert registry.get("write").execution_mode == "sequential"
    assert registry.get("edit").execution_mode == "sequential"
    assert registry.get("read").execution_mode == "parallel"
    assert registry.get("grep").execution_mode == "parallel"


def test_registry_duplicate_registration(tmp_path: Path) -> None:
    """重名注册被拒绝。"""
    registry = make_registry(tmp_path)
    with pytest.raises(ValueError, match="工具已存在"):
        registry.register(registry.get("bash"))


def test_truncate_head_line_limit() -> None:
    """截断器：行数上限生效且不产生半行。"""
    content = "\n".join(f"line{i}" for i in range(1, 3001))
    result = truncate_head(content)
    assert result.truncated
    assert result.truncated_by == "lines"
    assert result.output_lines == 2000
    assert result.content.count("\n") == 2000


def test_truncate_head_byte_limit() -> None:
    """截断器：字节上限生效。"""
    line = "x" * 600  # 每行 601 字节
    content = "\n".join([line] * 100)  # 60KB
    result = truncate_head(content)
    assert result.truncated
    assert result.truncated_by == "bytes"
    assert result.output_bytes <= 50 * 1024


def test_truncate_head_first_line_exceeds() -> None:
    """截断器：首行独超字节上限 → 空内容 + 标记。"""
    result = truncate_head("y" * 60000)
    assert result.first_line_exceeds_limit
    assert result.content == ""
