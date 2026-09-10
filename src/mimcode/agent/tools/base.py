"""AgentTool 协议：内置工具与插件工具的统一抽象。

对齐 pi 的 AgentTool（packages/agent/src/types.ts L150+ 的工具接口）：
- name/description/parameters（JSON Schema）→ wire 声明
- execute：参数已验证，产出 ToolResult；进度经 on_update 回调
- execution_mode：sequential 工具参与串行批（文件写/bash），
  其余并行（对齐 pi 的 executionMode 语义）

与 T3 的 ToolSpec 桥接：``to_spec()`` 产出 provider payload 的声明。
异常语义：execute 抛出的异常由 agent loop 捕获并归一为错误工具结果
（对齐 pi：错误不 crash loop，编码进 toolResult.isError）。
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from mimcode.types import TextBlock, ToolResult, ToolSpec

ToolUpdateCallback = Callable[[ToolResult], Awaitable[None] | None]
"""进度回调：部分结果（agent loop 转为 tool_execution_update 事件）。"""

ExecutionMode = Literal["sequential", "parallel"]


class AgentTool(ABC):
    """内置/插件工具的基类。

    子类需实现 ``parameters_schema`` 与 ``execute``；
    ``execution_mode`` 按默认 parallel，文件写类工具覆写为 sequential。
    """

    name: str = ""
    description: str = ""
    execution_mode: ExecutionMode = "parallel"

    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """工具参数的 JSON Schema（object 类型）。"""

    def to_spec(self) -> ToolSpec:
        """→ wire 级工具声明（provider payload）。"""
        return ToolSpec(
            name=self.name, description=self.description, parameters=self.parameters_schema()
        )

    @abstractmethod
    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        """执行工具。

        Args:
            tool_call_id: 调用 id（结果消息回填用）。
            args: 已验证的参数。
            signal: 中止信号；工具应在合适的等待点检查。
            on_update: 进度回调（部分结果）。

        Returns:
            工具结果（isError 语义由 agent loop 依据异常判定，
            工具内部错误应抛异常而非自行编码）。
        """

    @classmethod
    def text_result(cls, text: str, details: Any = None) -> ToolResult:
        """便捷构造：单文本块结果。"""
        return ToolResult(content=[TextBlock(text=text)], details=details)

    def check_aborted(self, signal: asyncio.Event | None) -> None:
        """检查中止信号；置位时抛异常（agent loop 归一为 aborted）。"""
        if signal is not None and signal.is_set():
            raise ToolAbortedError(f"工具 {self.name} 被中止")


class ToolAbortedError(Exception):
    """工具执行被用户中止。"""


class ToolArgumentError(Exception):
    """工具参数校验失败（如路径为空、offset 越界）。"""
