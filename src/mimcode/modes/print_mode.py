"""print 模式：一次性执行并输出（对齐 pi modes/print-mode.ts）。

行为：
- 事件流式输出（text_delta → stdout 无换行追加，打字机效果）
- thinking 分离：仅在 stderr 输出摘要（不污染 stdout 管道）
- 工具执行：stderr 单行状态（stdout 只保留模型正文）
- 退出码：0 正常 / 2 错误（流 error 终态）
- 落盘：AgentSession.persist_events 全量持久化
"""

from __future__ import annotations

import sys
from typing import TextIO

from mimcode.agent.loop import AgentEvent
from mimcode.app.agent_session import AgentSession, run_prompt
from mimcode.types import (
    AssistantMessage,
    MessageEnd,
    MessageUpdate,
    StreamTextDelta,
    StreamThinkingDelta,
    ToolExecutionStart,
)


async def run_print_mode(
    session: AgentSession,
    prompt: str,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """执行 print 模式。

    Args:
        session: 装配好的会话。
        prompt: 用户提示词。
        stdout: 正文输出流。
        stderr: 状态/思考输出流。

    Returns:
        进程退出码（0 正常 / 2 错误）。
    """
    if not prompt.strip():
        stderr.write('print 模式需要提示词（mimcode -p "提示词"）\n')
        return 2

    emitted_text = False

    def on_event(event: AgentEvent) -> None:
        nonlocal emitted_text
        if isinstance(event, MessageUpdate):
            assistant_event = event.assistant_event
            if isinstance(assistant_event, StreamTextDelta):
                stdout.write(assistant_event.delta)
                stdout.flush()
                emitted_text = True
            elif isinstance(assistant_event, StreamThinkingDelta):
                stderr.write(".")  # 思考进度点（不泄漏内容）
                stderr.flush()
        elif isinstance(event, ToolExecutionStart):
            stderr.write(f"[工具] {event.tool_name}…\n")
        elif isinstance(event, MessageEnd) and event.message.role == "assistant":
            # 正文定稿换行（仅当本轮有正文输出）
            if emitted_text:
                stdout.write("\n")
                emitted_text = False

    new_messages = await run_prompt(session, prompt, on_event=on_event)

    # 错误终态 → 退出码 2
    for message in new_messages:
        if isinstance(message, AssistantMessage) and message.stop_reason == "error":
            error_text = message.error_message or "未知错误"
            stderr.write(f"错误: {error_text}\n")
            return 2
    return 0
