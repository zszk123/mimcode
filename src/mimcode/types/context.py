"""LLM 调用边界的数据模型。

语义对齐 pi：
- Tool（packages/ai/src/types.ts L514）→ ToolSpec（wire 级工具声明）
- Context（L521）→ LlmContext（system prompt + 消息 + 工具）
- Model 的 v1 子集 → ModelInfo（完整目录在 T4 扩展）
- SimpleStreamOptions 的 v1 子集 → StreamOptions

与 Agent 层（T6）的边界：AgentMessage 包装层在调用 LLM 前转换为
本模块的 LlmContext（对齐 pi agent-loop 的 convertToLlm 边界）。
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, Field

from mimcode.types.messages import AgentMessage, Api

ThinkingLevel = Literal["minimal", "low", "medium", "high"]
"""思考级别（对齐 pi 的 ThinkingLevel v1 子集，xhigh/max 不迁移）。

"off" 是 CLI/应用层语义：off 表示不传级别（不启用思考），
不会出现在本类型中（对齐 pi：agent 层把 off 归一为 None）。
"""


class ToolSpec(BaseModel):
    """工具的 wire 级声明（发给模型的 schema）。

    执行逻辑（execute/进度回调/渲染）在 T5 的 AgentTool 中，
    本模型只承载协议载荷。
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    """JSON Schema（object 类型）。空 dict 表示无参数工具。"""


class ModelInfo(BaseModel):
    """模型元数据（v1 子集）。

    T4 的目录与配置合并会产出本模型的实例列表。
    """

    id: str
    provider: str
    api: Api
    name: str | None = None
    context_window: int = 128000
    max_output_tokens: int = 8192
    supports_thinking: bool = False


class LlmContext(BaseModel):
    """LLM 调用上下文（对齐 pi 的 Context）。"""

    system_prompt: str | None = None
    messages: list[AgentMessage] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)


class StreamOptions(BaseModel):
    """流式调用选项（对齐 pi 的 SimpleStreamOptions v1 子集）。

    与 pi 的差异：TS 的 AbortSignal 在 Python 侧以 asyncio.Event 表达，
    置位即请求中止（provider 在 chunk 间隙检查并产出 aborted 终态）。
    """

    api_key: str | None = None
    signal: asyncio.Event | None = None
    thinking_level: ThinkingLevel | None = None
    max_tokens: int | None = None
