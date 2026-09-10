"""消息与内容块的数据模型。

语义对齐 pi（packages/ai/src/types.ts）：
- 内容块 type 标签、消息 role、stopReason 取值与 pi 完全一致
- 字段名按 Python 惯例转 snake_case（如 toolCallId -> tool_call_id）
- v1 裁剪：不迁移 pi 的 deferred/responseId/diagnostics/cost 等字段
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 内容块（pi: TextContent / ThinkingContent / ImageContent / ToolCall）
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    """文本内容块。"""

    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    """思考内容块（Claude extended thinking / 兼容端点 reasoning）。"""

    type: Literal["thinking"] = "thinking"
    thinking: str
    # Provider 侧不透明的回放签名（Claude 多轮 thinking 连续性所需）
    thinking_signature: str | None = None
    # 安全过滤改写过的思考：原文以不透明形式存于 thinking_signature
    redacted: bool = False


class ImageBlock(BaseModel):
    """图片内容块。v1 仅占位（多模态输入为 Out of Scope）。"""

    type: Literal["image"] = "image"
    data: str  # base64 编码
    mime_type: str


class ToolCallBlock(BaseModel):
    """工具调用块（助手消息内）。arguments 为已解析的 JSON 对象。"""

    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ImageBlock | ToolCallBlock,
    Field(discriminator="type"),
]
"""助手消息内容块联合（按 type 判别）。"""

UserContentBlock = Annotated[TextBlock | ImageBlock, Field(discriminator="type")]
"""用户/工具结果消息内容块联合（仅文本与图片）。"""

# ---------------------------------------------------------------------------
# 用量统计（pi: Usage，v1 裁剪掉 cacheWrite1h / cost / totalTokens 存储）
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    """单次请求的 token 用量。

    reasoning 是 output 的子集（provider 报告思考 token 时填写）。
    """

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0

    @property
    def total_tokens(self) -> int:
        """上下文口径的总 token（对齐 pi 的 calculateContextTokens）。"""
        return self.input + self.output + self.cache_read + self.cache_write


# ---------------------------------------------------------------------------
# 停止原因与协议标识
# ---------------------------------------------------------------------------

StopReason = Literal["pending", "stop", "length", "toolUse", "error", "aborted"]
"""对齐 pi 的 StopReason（v1 不含 deferred）。"""

Api = Literal["openai", "anthropic"]
"""协议标识：OpenAI 协议 / Claude 协议（协议即 provider）。"""


def _now_ms() -> int:
    """当前毫秒时间戳（对齐 pi 的 Date.now()）。"""
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# 消息（pi: UserMessage / AssistantMessage / ToolResultMessage）
# ---------------------------------------------------------------------------


class UserMessage(BaseModel):
    """用户消息。content 允许纯字符串或文本/图片块列表。"""

    role: Literal["user"] = "user"
    content: str | list[UserContentBlock]
    timestamp: int = Field(default_factory=_now_ms)


class AssistantMessage(BaseModel):
    """助手消息。

    流式过程中以 stop_reason=pending 的部分消息出现，
    结束时替换为终态（stop/length/toolUse/error/aborted）。
    """

    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock] = Field(default_factory=list)
    api: Api
    provider: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = "pending"
    error_message: str | None = None
    timestamp: int = Field(default_factory=_now_ms)


class ToolResultMessage(BaseModel):
    """工具结果消息（回填给模型的 toolResult 角色）。"""

    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[UserContentBlock] = Field(default_factory=list)
    # 工具自定义的结构化细节（不进入模型上下文）
    details: Any = None
    # 工具执行自身的用量（不计入主 LLM 上下文核算）
    usage: Usage | None = None
    is_error: bool = False
    timestamp: int = Field(default_factory=_now_ms)


AgentMessage = Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),
]
"""会话消息联合（按 role 判别）。"""


# ---------------------------------------------------------------------------
# 工具执行结果（pi: AgentToolResult，tool_execution_end 事件载荷）
# ---------------------------------------------------------------------------


class ToolResult(BaseModel):
    """工具执行结果（组装为 ToolResultMessage 前的形态）。"""

    content: list[UserContentBlock] = Field(default_factory=list)
    details: Any = None
    usage: Usage | None = None
    # 本批全部结果都置 True 时，agent loop 在本批工具后停止
    terminate: bool = False
