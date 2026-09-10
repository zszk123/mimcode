"""E2E-3：interactive 模式 headless 冒烟（checklist 端到端验收）。

PTY 模拟的 Windows 等价形态（差异标注）：真实子进程内完成
AgentSession 全量装配（config → faux 端点 → 工具/技能/会话）与
渲染管线落地，仅绕开 prompt_toolkit 终端输入层——输入序列以
``run_agent_turn`` / ``handle_input`` 直调驱动（T12 单测已覆盖
键绑定层）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.e2e.conftest import (
    fixture,
    read_requests,
    run_driver,
    text_chunks,
    write_faux_endpoint,
)

pytestmark = pytest.mark.e2e
DRIVER = """\
import asyncio
import sys

from mimcode.app.agent_session import AgentSession
from mimcode.tui.app import InteractiveApp


async def main() -> int:
    cwd = sys.argv[1]
    session = AgentSession(cwd=cwd)
    app = InteractiveApp(
        cwd=cwd,
        agent_context=session.build_agent_context(),
        config_factory=session.make_loop_config,
        command_context_factory=session.command_context,
        persist_events=session.persist_events,
    )
    # 1) 普通输入：一轮 agent + 渲染落地
    await app.run_agent_turn("你好")
    # 2) slash 命令：切换模型
    result = await app.handle_input("/model faux-model")
    if result is None or result.error:
        return 4
    # 3) 退出
    await app.handle_input("/exit")
    return 0 if app.exit_requested else 5


raise SystemExit(asyncio.run(main()))
"""


async def test_e2e3_interactive_headless_smoke(tmp_path: Path) -> None:
    """E2E-3：输入提示 → 助手回复渲染完成 → /model → /exit 正常退出。"""
    cwd = tmp_path / "work"
    home = tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    faux_dir = write_faux_endpoint(cwd, [fixture("faux-model", text_chunks("交互回复正文"))])

    driver = cwd / "_driver_interactive.py"
    driver.write_text(DRIVER, encoding="utf-8")

    proc = run_driver(driver, str(cwd), cwd=cwd, home=home)

    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    stdout = proc.stdout.decode("utf-8")
    # 助手回复渲染完成（渲染管线落地）
    assert "交互回复正文" in stdout
    # slash 命令输出：模型切换
    assert "已切换模型: faux-test/faux-model" in stdout

    # faux 收到一轮流式请求（上下文只有该轮用户消息，无重复注入）
    requests = read_requests(faux_dir)
    assert len(requests) == 1
    roles = [message["role"] for message in requests[0]["messages"]]
    assert roles == ["system", "user"]

    # 交互轮次落盘（run_agent_turn 的 persist_events 接线）
    session_files = sorted((home / ".mimcode" / "sessions").rglob("*.jsonl"))
    assert len(session_files) == 1
    entries = [
        json.loads(line)
        for line in session_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    message_roles = [entry["message"]["role"] for entry in entries if entry["type"] == "message"]
    assert message_roles == ["user", "assistant"]
