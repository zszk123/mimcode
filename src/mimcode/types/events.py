"""AgentEvent 与助手流事件协议。

语义对齐 pi：
- AgentEvent 联合（packages/agent/src/types.ts L428-443）
- AssistantMessageEvent 流事件（packages/ai/src/types.ts L534-550）

流契约（对齐 pi 的 StreamFn 约定）：流函数不抛异常；
失败以 error 事件 + stopReason 为 error/aborted 的终态消息编码进流。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from mimcode.types.messages import (
    AgentMessage,
    AssistantMessage,
    ToolCallBlock,
    ToolResult,
    ToolResultMessage,
)

# ---------------------------------------------------------------------------
# 助手流事件（pi: AssistantMessageEvent）
# ---------------------------------------------------------------------------


class StreamStart(BaseModel):
    """流开始，携带首个部分消息。"""

    type: Literal["start"] = "start"
    partial: AssistantMessage


class StreamTextStart(BaseModel):
    """文本块开始。"""

    type: Literal["text_start"] = "text_start"
    content_index: int
    partial: AssistantMessage


class StreamTextDelta(BaseModel):
    """文本增量。"""

    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class StreamTextEnd(BaseModel):
    """文本块结束，content 为该块完整文本。"""

    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    partial: AssistantMessage


class StreamThinkingStart(BaseModel):
    """思考块开始。"""

    type: Literal["thinking_start"] = "thinking_start"
    content_index: int
    partial: AssistantMessage


class StreamThinkingDelta(BaseModel):
    """思考增量。"""

    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class StreamThinkingEnd(BaseModel):
    """思考块结束，content 为该块完整思考文本。"""

    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: str
    partial: AssistantMessage


class StreamToolCallStart(BaseModel):
    """工具调用块开始。"""

    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int
    partial: AssistantMessage


class StreamToolCallDelta(BaseModel):
    """工具调用参数的 JSON 增量片段。"""

    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class StreamToolCallEnd(BaseModel):
    """工具调用块结束，携带解析完成的调用块。"""

    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    tool_call: ToolCallBlock
    partial: AssistantMessage


class StreamDone(BaseModel):
    """流正常结束，携带终态消息。"""

    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage


class StreamError(BaseModel):
    """流失败/中止（对齐 pi：失败不抛异常，编码为该事件）。"""

    type: Literal["error"] = "error"
    reason: Literal["aborted", "error"]
    error: AssistantMessage


AssistantStreamEvent = Annotated[
    StreamStart
    | StreamTextStart
    | StreamTextDelta
    | StreamTextEnd
    | StreamThinkingStart
    | StreamThinkingDelta
    | StreamThinkingEnd
    | StreamToolCallStart
    | StreamToolCallDelta
    | StreamToolCallEnd
    | StreamDone
    | StreamError,
    Field(discriminator="type"),
]
"""助手流事件联合（按 type 判别）。"""


# ---------------------------------------------------------------------------
# Agent 事件（pi: AgentEvent）
# ---------------------------------------------------------------------------


class AgentStart(BaseModel):
    """一次 agent 运行开始。"""

    type: Literal["agent_start"] = "agent_start"


class AgentEnd(BaseModel):
    """agent 运行结束，携带本次运行新增的全部消息。"""

    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage]


class TurnStart(BaseModel):
    """一轮开始（一轮 = 一次助手响应 + 其工具执行）。"""

    type: Literal["turn_start"] = "turn_start"


class TurnEnd(BaseModel):
    """一轮结束，携带本轮助手消息与工具结果消息。"""

    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage
    tool_results: list[ToolResultMessage]


class MessageStart(BaseModel):
    """用户/助手/工具结果消息进入上下文。"""

    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdate(BaseModel):
    """助手消息流式更新（仅流式中的助手消息）。"""

    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    assistant_event: AssistantStreamEvent


class MessageEnd(BaseModel):
    """消息定稿。"""

    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStart(BaseModel):
    """工具开始执行。"""

    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


class ToolExecutionUpdate(BaseModel):
    """工具执行中的部分结果（进度回调）。"""

    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    partial_result: Any


class ToolExecutionEnd(BaseModel):
    """工具执行结束。"""

    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: ToolResult
    is_error: bool


AgentEvent = Annotated[
    AgentStart
    | AgentEnd
    | TurnStart
    | TurnEnd
    | MessageStart
    | MessageUpdate
    | MessageEnd
    | ToolExecutionStart
    | ToolExecutionUpdate
    | ToolExecutionEnd,
    Field(discriminator="type"),
]
"""agent loop 对外事件联合（按 type 判别）。"""
