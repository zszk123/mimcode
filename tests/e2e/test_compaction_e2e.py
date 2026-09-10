"""E2E-4：compaction 触发路径（checklist 端到端验收）。

构造超长会话 fixture（早期大消息压满上下文窗口），headless interactive
发送一条消息，验证：
- 发送前触发压缩（渲染产出含压缩摘要标记行）
- 摘要请求走独立 faux 回放（requests 第一条为摘要请求形态）
- 会话文件落 compaction 条目；恢复后上下文以摘要消息开头
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.e2e.conftest import fixture, read_requests, run_driver, text_chunks, write_faux_endpoint

from mimcode.app.session import SessionManager
from mimcode.types import AssistantMessage, TextBlock, UserMessage

pytestmark = pytest.mark.e2e

CONTEXT_WINDOW = 20_000
"""faux 模型上下文窗口（小值以触发阈值：tokens > window - 16384）。"""

BIG_TEXT = "x" * 8_000
"""早期大消息文本（8000 chars → 2000 tokens 估算）。"""

SUMMARY_TEXT = "## Goal\n- 验证压缩路径"
"""faux 回放的摘要文本。"""

DRIVER = """\
import asyncio
import sys

from mimcode.app.agent_session import AgentSession
from mimcode.tui.app import InteractiveApp


async def main() -> int:
    cwd = sys.argv[1]
    session = AgentSession(cwd=cwd, continue_session=True)
    compacted = False

    async def pre_turn():
        nonlocal compacted
        outcome = await session.maybe_compact()
        if outcome is None:
            return None
        compacted = True
        return f"已压缩上下文（~{outcome.tokens_before} tokens）"

    app = InteractiveApp(
        cwd=cwd,
        agent_context=session.build_agent_context(),
        config_factory=session.make_loop_config,
        command_context_factory=session.command_context,
        pre_turn_hook=pre_turn,
    )
    await app.run_agent_turn("继续")
    return 0 if compacted else 3


raise SystemExit(asyncio.run(main()))
"""


async def test_e2e4_compaction_triggers(tmp_path: Path) -> None:
    """E2E-4：超长会话 + 一条消息 → 压缩摘要标记 + compaction 条目。"""
    cwd = tmp_path / "work"
    home = tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    # 脚本序：轮 1 = 摘要请求回放；轮 2 = 正文回复
    faux_dir = write_faux_endpoint(
        cwd,
        [
            fixture("faux-model", text_chunks(SUMMARY_TEXT)),
            fixture("faux-model", text_chunks("压缩后的回复")),
        ],
        context_window=CONTEXT_WINDOW,
    )

    # 构造超长会话：早期 3 组大消息（~12000 tokens > 阈值 3616）
    session = SessionManager.create(str(cwd), home=home)
    for i in range(3):
        session.append_message(UserMessage(content=f"大问题{i}"))
        session.append_message(
            AssistantMessage(
                content=[TextBlock(text=BIG_TEXT)],
                api="openai",
                provider="faux-test",
                model="faux-model",
                stop_reason="stop",
            )
        )
    session.append_message(UserMessage(content="近期小问题"))

    driver = cwd / "_driver_compaction.py"
    driver.write_text(DRIVER, encoding="utf-8")
    proc = run_driver(driver, str(cwd), cwd=cwd, home=home)

    # 退出码 3 = 压缩未发生（驱动脚本的区分约定）
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    stdout = proc.stdout.decode("utf-8")
    # 渲染产出含压缩摘要标记（pre-turn 提示行）
    assert "已压缩上下文" in stdout
    # 压缩后本轮正文照常渲染
    assert "压缩后的回复" in stdout

    # 请求序列：第一条是摘要请求（system = 摘要提示 + 尾部摘要指令）
    requests = read_requests(faux_dir)
    assert len(requests) == 2
    first = requests[0]["messages"]
    assert first[0]["role"] == "system"
    assert "summarization assistant" in first[0]["content"]
    assert first[-1]["role"] == "user"
    assert "structured context checkpoint summary" in first[-1]["content"]

    # 会话落盘 compaction 条目
    session_files = sorted((home / ".mimcode" / "sessions").rglob("*.jsonl"))
    assert len(session_files) == 1
    import json

    entries = [
        json.loads(line)
        for line in session_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    compactions = [entry for entry in entries if entry["type"] == "compaction"]
    assert len(compactions) == 1
    assert compactions[0]["summary"] == SUMMARY_TEXT
    assert compactions[0]["tokens_before"] > 0
    assert compactions[0]["first_kept_entry_id"]

    # 恢复后上下文以摘要消息开头（被摘要的前缀省略）
    restored = SessionManager.open(session_files[0])
    context = restored.build_context()
    first_message = context.messages[0]
    assert isinstance(first_message, UserMessage)
    assert "[Compacted conversation summary]" in first_message.content
