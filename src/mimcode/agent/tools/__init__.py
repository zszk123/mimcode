"""内置工具注册表：七件套实例化与查询。"""

from __future__ import annotations

from mimcode.agent.tools.base import AgentTool
from mimcode.agent.tools.bash import BashTool
from mimcode.agent.tools.edit import EditTool
from mimcode.agent.tools.file_io import ReadTool, WriteTool
from mimcode.agent.tools.search import FindTool, GrepTool, LsTool


def builtin_tools(cwd: str) -> list[AgentTool]:
    """构造指定工作目录下的内置七件套。

    顺序即声明顺序（对齐 pi tools/index.ts 的角色）：
    bash/read/write/edit/ls/find/grep。
    """
    return [
        BashTool(cwd),
        ReadTool(cwd),
        WriteTool(cwd),
        EditTool(cwd),
        LsTool(cwd),
        FindTool(cwd),
        GrepTool(cwd),
    ]


class ToolRegistry:
    """工具注册表：名字 → 实例查询（agent loop T6 使用）。"""

    def __init__(self, tools: list[AgentTool] | None = None, cwd: str = ".") -> None:
        self._tools: dict[str, AgentTool] = {}
        for tool in tools if tools is not None else builtin_tools(cwd):
            if tool.name in self._tools:
                raise ValueError(f"重复注册工具: {tool.name}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool | None:
        """按名取工具。"""
        return self._tools.get(name)

    def all(self) -> list[AgentTool]:
        """全部已注册工具。"""
        return list(self._tools.values())

    def register(self, tool: AgentTool, *, replace: bool = False) -> None:
        """注册工具（插件入口，T11 使用）。

        Raises:
            ValueError: 重名且未允许替换。
        """
        if tool.name in self._tools and not replace:
            raise ValueError(f"工具已存在: {tool.name}（如需覆盖请置 replace=True）")
        self._tools[tool.name] = tool
