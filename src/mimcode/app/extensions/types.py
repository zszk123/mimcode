"""插件类型（Python 版插件 API 面）。

插件形态：模块级 ``register(ctx: ExtensionContext)`` 函数（对齐 TS 插件
的默认导出 register 约定，按 Python 习惯改为显式函数名）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from mimcode.agent.tools.base import AgentTool
from mimcode.types import AgentEvent

EventHandler = Callable[[AgentEvent], Awaitable[None] | None]
"""事件监听器（异步优先，同步兼容）。"""


class ExtensionContext:
    """插件可触达的注册面（对齐 pi ExtensionContext 的 v1 子集）。

    插件经 ``register(ctx)`` 获得 ctx，可：
    - ``register_tool``：注册自定义工具
    - ``register_command``：注册 slash 命令
    - ``on_event``：订阅 agent 事件
    - ``log``：写入诊断（/extensions 与启动提示可查）
    """

    def __init__(self, extension_name: str) -> None:
        self._extension_name = extension_name
        self.tools: list[AgentTool] = []
        self.commands: list[Any] = []
        self.event_handlers: list[tuple[str, EventHandler]] = []
        self.logs: list[str] = []
        self.registered_names: list[str] = []

    @property
    def name(self) -> str:
        """插件名（文件 stem）。"""
        return self._extension_name

    def register_tool(self, tool: AgentTool) -> None:
        """注册自定义工具（进入 agent 工具注册表）。"""
        self.tools.append(tool)
        self.registered_names.append(f"tool:{tool.name}")

    def register_command(
        self,
        name: str,
        handler: Callable[[Any, str], Awaitable[Any]],
        description: str,
        argument_hint: str | None = None,
    ) -> None:
        """注册 slash 命令。"""
        from mimcode.app.commands import SlashCommand

        self.commands.append(
            SlashCommand(
                name=name,
                description=description,
                argument_hint=argument_hint,
                handler=handler,
                source="extension",
            )
        )
        self.registered_names.append(f"command:{name}")

    def on_event(self, event_type: str, handler: EventHandler) -> None:
        """订阅 agent 事件（按事件 type 过滤；"*" 订阅全部）。"""
        self.event_handlers.append((event_type, handler))
        self.registered_names.append(f"event:{event_type}")

    def log(self, message: str) -> None:
        """诊断日志（不进入 LLM 上下文）。"""
        self.logs.append(f"[{self._extension_name}] {message}")
