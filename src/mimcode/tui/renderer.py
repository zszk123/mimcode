"""渲染管线：AgentEvent → 渲染动作（纯逻辑层，可 headless 测试）。

设计：
- ``RenderAction``：终端动作的数据描述（写行/流式追加/状态翻转/错误），
  呈现层（app）决定落地方式（Rich/纯文本/测试缓冲）
- ``RendererPipeline``：消费 AgentEvent 序列，维护流式缓冲与
  思考折叠语义，产出动作列表

语义（对齐 pi interactive 的渲染行为）：
- 流式正文：text_delta 追加到当前行缓冲，turn 结束定稿为一行
- 思考块：折叠展示（一行摘要 + 可展开计数），不与正文混排
- 工具执行：start 展示「名称(参数摘要)」，end 追加结果预览与耗时标记
- message_end(assistant)：状态用量累计（TuiState）
- error/aborted 终态：产出错误动作并清除 busy（不残留加载动画）
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from mimcode.tui.state import TuiState
from mimcode.types import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    AssistantMessage,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    StreamTextDelta,
    StreamThinkingDelta,
    TextBlock,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolResult,
    TurnEnd,
    TurnStart,
    UserMessage,
)

TOOL_RESULT_PREVIEW_CHARS = 120
"""工具结果预览的截断长度。"""

THINKING_PREVIEW_CHARS = 60
"""折叠思考摘要的截断长度。"""


@dataclasses.dataclass(frozen=True)
class RenderAction:
    """一次渲染动作。"""

    kind: Literal["write_line", "stream_chunk", "set_busy", "clear_busy", "footer", "error"]
    text: str = ""
    """动作文本（行内容/流式片段/错误信息）。"""

    style: str = "system"
    """语义角色（主题取色）：user/assistant/tool/error/thinking/system。"""


class RendererPipeline:
    """AgentEvent → RenderAction 的纯逻辑转换器。"""

    def __init__(self, state: TuiState | None = None) -> None:
        self.state = state or TuiState()
        self._stream_buffer: list[str] = []
        self._thinking_buffer: list[str] = []
        self._thinking_block_open = False
        self._text_block_open = False
        self._turn_assistant: AssistantMessage | None = None
        self.update_calls = 0
        """流式增量回调计数（验收：text_delta 到达次数）。"""

    # ------------------------------------------------------------------
    # 事件入口
    # ------------------------------------------------------------------

    def handle(self, event: AgentEvent) -> list[RenderAction]:
        """处理一个事件，返回本次产出的动作（isinstance 窄化分发）。"""
        if isinstance(event, AgentStart):
            self.state.busy = True
            return [RenderAction(kind="set_busy", text="运行中")]

        if isinstance(event, TurnStart):
            self._stream_buffer = []
            self._thinking_buffer = []
            self._thinking_block_open = False
            self._text_block_open = False
            return []

        if isinstance(event, MessageStart):
            message = event.message
            if isinstance(message, UserMessage):
                preview = self._user_preview(message)
                return [RenderAction(kind="write_line", text=f"你: {preview}", style="user")]
            return []

        if isinstance(event, MessageUpdate):
            return self._handle_stream_delta(event.assistant_event)

        if isinstance(event, MessageEnd):
            return self._handle_message_end(event.message)

        if isinstance(event, ToolExecutionStart):
            preview = self._tool_args_preview(event.tool_name, event.args)
            return [
                RenderAction(
                    kind="write_line",
                    text=f"  ⚙ {event.tool_name}({preview})…",
                    style="tool",
                )
            ]

        if isinstance(event, ToolExecutionEnd):
            return self._handle_tool_end(event.tool_name, event.result, event.is_error)

        if isinstance(event, TurnEnd):
            self.state.turns += 1
            footer = RenderAction(kind="footer", text=self.state.footer_line())
            return [footer]

        if isinstance(event, AgentEnd):
            self.state.busy = False
            for message in event.messages:
                if isinstance(message, AssistantMessage) and message.stop_reason not in (
                    "aborted",
                    "error",
                ):
                    self.state.add_usage(message.usage)
            return [
                RenderAction(kind="footer", text=self.state.footer_line()),
                RenderAction(kind="clear_busy"),
            ]

        return []

    # ------------------------------------------------------------------
    # 流式增量
    # ------------------------------------------------------------------

    def _handle_stream_delta(self, assistant_event: object) -> list[RenderAction]:
        """流式增量：text 追加正文缓冲；thinking 追加折叠缓冲。"""
        if isinstance(assistant_event, StreamTextDelta):
            self._text_block_open = True
            self._stream_buffer.append(assistant_event.delta)
            self.update_calls += 1
            return [RenderAction(kind="stream_chunk", text=assistant_event.delta)]

        if isinstance(assistant_event, StreamThinkingDelta):
            delta = assistant_event.delta
            if not self._thinking_block_open:
                self._thinking_block_open = True
                return [
                    RenderAction(
                        kind="write_line",
                        text=self._thinking_header(),
                        style="thinking",
                    ),
                    RenderAction(kind="stream_chunk", text=delta),
                ]
            self._thinking_buffer.append(delta)
            return [RenderAction(kind="stream_chunk", text=delta)]
        return []

    def _thinking_header(self) -> str:
        """折叠思考区的头行。"""
        return "  ✻ 思考中（折叠，结束可展开）"

    # ------------------------------------------------------------------
    # 定稿
    # ------------------------------------------------------------------

    def _handle_message_end(self, message: object) -> list[RenderAction]:
        """消息定稿：assistant 产出正文行；工具结果并入工具块。"""
        if isinstance(message, AssistantMessage):
            self._turn_assistant = message
            actions: list[RenderAction] = []

            thinking_text = "".join(self._thinking_buffer)
            if self._thinking_block_open and thinking_text.strip():
                preview = thinking_text.strip()[:THINKING_PREVIEW_CHARS]
                suffix = "…" if len(thinking_text.strip()) > THINKING_PREVIEW_CHARS else ""
                actions.append(
                    RenderAction(
                        kind="write_line",
                        text=f"  ✻ 思考（{len(thinking_text)} 字符）: {preview}{suffix}",
                    )
                )

            text = self._assistant_text(message)
            if self._stream_buffer or text:
                body = text if text else "".join(self._stream_buffer)
                actions.append(
                    RenderAction(kind="write_line", text=f"mim: {body}", style="assistant")
                )
            if not actions:
                actions.append(
                    RenderAction(kind="write_line", text="mim: (空回复)", style="assistant")
                )
            return actions
        return []

    def _handle_tool_end(
        self, tool_name: str, result: ToolResult, is_error: bool
    ) -> list[RenderAction]:
        """工具结束：结果预览（错误标记）。"""
        preview = self._tool_result_preview(result)
        prefix = "  ✗" if is_error else "  ✓"
        return [
            RenderAction(
                kind="write_line",
                text=f"{prefix} {tool_name}: {preview}",
                style="error" if is_error else "tool",
            )
        ]

    # ------------------------------------------------------------------
    # 纯文本抽取助手
    # ------------------------------------------------------------------

    @staticmethod
    def _assistant_text(message: AssistantMessage) -> str:
        """助手消息的文本块拼接。"""
        return "".join(block.text for block in message.content if isinstance(block, TextBlock))

    @staticmethod
    def _user_preview(message: UserMessage) -> str:
        """用户消息预览（单行化）。"""
        content = message.content
        text = (
            content
            if isinstance(content, str)
            else "".join(block.text for block in content if isinstance(block, TextBlock))
        )
        return text.replace("\n", " ⏎ ")

    @staticmethod
    def _tool_args_preview(name: str, args: dict) -> str:
        """工具参数摘要。"""
        if not args:
            return ""
        items = []
        for key, value in list(args.items())[:2]:
            value_text = str(value)
            if len(value_text) > 40:
                value_text = value_text[:40] + "…"
            items.append(f"{key}={value_text}")
        joined = ", ".join(items)
        more = " …" if len(args) > 2 else ""
        return f"{joined}{more}"

    @staticmethod
    def _tool_result_preview(result: object) -> str:
        """工具结果预览（首文本块，截断）。"""
        content = getattr(result, "content", [])
        text = ""
        for block in content or []:
            if isinstance(block, TextBlock):
                text = block.text
                break
        text = text.replace("\n", " ⏎ ")
        if len(text) > TOOL_RESULT_PREVIEW_CHARS:
            return text[:TOOL_RESULT_PREVIEW_CHARS] + "…"
        return text

    @staticmethod
    def _is_new_user_message(message: UserMessage) -> bool:
        """区分 prompt 注入与 steering（v1：全部展示）。"""
        return True
