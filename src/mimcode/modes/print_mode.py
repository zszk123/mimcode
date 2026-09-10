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


def _is_text_delta(event: AgentEvent) -> bool:
    """assistant 流式正文事件判定。"""
    if event.type != "message_update":
        return False
    assistant_event = event.assistant_event
    return assistant_event.type == "text_delta"


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

    def on_event(event: AgentEvent) -> None:
        if _is_text_delta(event):
            stdout.write(event.assistant_event.delta)  # type: ignore[attr-defined]
            stdout.flush()
        elif event.type == "message_update":
            assistant_event = event.assistant_event
            if assistant_event.type == "thinking_delta":
                stderr.write(".")  # 思考进度点（不泄漏内容）
                stderr.flush()
        elif event.type == "tool_execution_start":
            stderr.write(f"[工具] {event.tool_name}…\n")
        elif event.type == "message_end" and event.message.role == "assistant":
            # 正文定稿换行（仅当有正文输出）
            stdout.write("\n")

    new_messages = await run_prompt(session, prompt, on_event=on_event)

    # 错误终态 → 退出码 2
    for message in new_messages:
        if (
            message.role == "assistant" and message.stop_reason == "error"  # type: ignore[union-attr]
        ):
            error_text = message.error_message or "未知错误"  # type: ignore[union-attr]
            stderr.write(f"错误: {error_text}\n")
            return 2
    return 0
