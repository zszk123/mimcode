"""E2E-1/2/5：print 模式进程级全链路（checklist 端到端验收）。

- E2E-1 print 全链路：流式正文 → stdout、退出码 0、会话落盘
- E2E-2 会话续接：-c 第二进程的 faux 请求包含上一会话消息
- E2E-5 Windows shell：真实工具调用（PowerShell echo）→ 工具执行块 + 命令输出
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.e2e.conftest import (
    fixture,
    read_entries,
    read_requests,
    run_cli,
    session_files,
    text_chunks,
    tool_call_chunks,
    write_faux_endpoint,
)

pytestmark = pytest.mark.e2e


def setup_workspace(tmp_path: Path, fixtures: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    """构造 e2e 工作区（cwd/home/faux 目录）。"""
    cwd = tmp_path / "work"
    home = tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    faux_dir = write_faux_endpoint(cwd, fixtures)
    return cwd, home, faux_dir


async def test_e2e1_print_full_pipeline(tmp_path: Path) -> None:
    """E2E-1：print 全链路——stdout 流式正文、退出码 0、会话落盘。"""
    cwd, home, faux_dir = setup_workspace(
        tmp_path, [fixture("faux-model", text_chunks("你好，这是最终回复"))]
    )

    proc = run_cli(["-p", "测试提示", "--model", "faux-model"], cwd=cwd, home=home)

    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    stdout = proc.stdout.decode("utf-8")
    assert "你好，这是最终回复" in stdout

    # faux 收到一次流式请求，上下文只有本轮用户消息
    requests = read_requests(faux_dir)
    assert len(requests) == 1
    roles = [message["role"] for message in requests[0]["messages"]]
    assert roles == ["system", "user"]
    assert "测试提示" in requests[0]["messages"][1]["content"]

    # 会话落盘：header + user + assistant
    files = session_files(home)
    assert len(files) == 1
    entries = read_entries(files[0])
    assert entries[0]["type"] == "session"
    message_roles = [entry["message"]["role"] for entry in entries if entry["type"] == "message"]
    assert message_roles == ["user", "assistant"]


async def test_e2e2_continue_session(tmp_path: Path) -> None:
    """E2E-2：-c 第二进程的 faux 请求 messages 前缀包含上一会话全部消息。"""
    fixtures = [
        fixture("faux-model", text_chunks("第一答")),
        fixture("faux-model", text_chunks("第二答")),
    ]
    cwd, home, faux_dir = setup_workspace(tmp_path, fixtures)

    first = run_cli(["-p", "第一问", "--model", "faux-model"], cwd=cwd, home=home)
    assert first.returncode == 0, first.stderr.decode("utf-8", "replace")
    assert len(read_requests(faux_dir)) == 1

    second = run_cli(["-p", "继续", "--model", "faux-model", "-c"], cwd=cwd, home=home)
    assert second.returncode == 0, second.stderr.decode("utf-8", "replace")
    assert "第二答" in second.stdout.decode("utf-8")

    # 轮转续接：第二进程回放第二个脚本（不是重放第一个）
    requests = read_requests(faux_dir)
    assert len(requests) == 2

    # faux 收到的消息序列包含上一会话的消息（checklist 断言）
    second_messages = requests[1]["messages"]
    roles = [message["role"] for message in second_messages]
    assert roles == ["system", "user", "assistant", "user"]
    contents = [
        second_messages[1]["content"],
        second_messages[2]["content"],
        second_messages[3]["content"],
    ]
    assert contents[0] == "第一问"
    assert contents[1] == "第一答"
    assert contents[2] == "继续"

    # 会话仍是一个文件（-c 恢复不新建），历史完整
    files = session_files(home)
    assert len(files) == 1
    entries = read_entries(files[0])
    message_roles = [entry["message"]["role"] for entry in entries if entry["type"] == "message"]
    assert message_roles == ["user", "assistant", "user", "assistant"]


async def test_e2e5_windows_shell_tool_call(tmp_path: Path) -> None:
    """E2E-5：Windows shell——真实 bash 工具调用（PowerShell echo）。

    stderr 出现工具执行块；会话落盘 toolResult；第二轮回复引用命令输出。
    """
    fixtures = [
        fixture(
            "faux-model", tool_call_chunks("c1", "bash", json.dumps({"command": "echo hello"}))
        ),
        fixture("faux-model", text_chunks("命令输出为 hello")),
    ]
    cwd, home, faux_dir = setup_workspace(tmp_path, fixtures)

    proc = run_cli(["-p", "执行 echo", "--model", "faux-model"], cwd=cwd, home=home)

    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    stderr = proc.stderr.decode("utf-8")
    stdout = proc.stdout.decode("utf-8")
    # 工具执行块（print 模式的工具状态行）
    assert "[工具] bash" in stderr
    # 命令输出出现在最终回复与工具结果里
    assert "hello" in stdout

    # 工具真实执行：toolResult 内容含命令输出
    files = session_files(home)
    assert len(files) == 1
    entries = read_entries(files[0])
    tool_results = [
        entry["message"]
        for entry in entries
        if entry["type"] == "message" and entry["message"]["role"] == "toolResult"
    ]
    assert len(tool_results) == 1
    tool_text = "".join(
        block["text"] for block in tool_results[0]["content"] if block.get("type") == "text"
    )
    assert "hello" in tool_text

    # 请求序列：第一轮 user → 第二轮带 assistant + tool 消息
    requests = read_requests(faux_dir)
    assert len(requests) == 2
    roles = [message["role"] for message in requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    payload_tool: dict[str, Any] = requests[1]["messages"][3]
    assert "hello" in payload_tool["content"]
