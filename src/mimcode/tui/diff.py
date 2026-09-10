"""Diff 渲染（对齐 pi diff.ts / edit-diff.ts 的角色）。

edit 工具结果 details.diff（unified diff 文本）→ Rich Text：
- ``-`` 行（删除）：红色
- ``+`` 行（新增）：绿色
- ``@@`` 行（块头）：青色
- 上下文行：默认色

语义映射（checklist）：删除行红/减号、新增行绿/加号。
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.text import Text

DIFF_MINUS_STYLE = "bold red"
DIFF_PLUS_STYLE = "bold green"
DIFF_HEADER_STYLE = "bold cyan"


def diff_renderable(diff_text: str) -> RenderableType:
    """unified diff 文本 → 带语义色的 Rich Text。"""
    text = Text()
    lines = diff_text.splitlines()
    for line in lines:
        if line.startswith("-") and not line.startswith("---"):
            text.append(line + "\n", style=DIFF_MINUS_STYLE)
        elif line.startswith("+") and not line.startswith("+++"):
            text.append(line + "\n", style=DIFF_PLUS_STYLE)
        elif line.startswith("@@"):
            text.append(line + "\n", style=DIFF_HEADER_STYLE)
        else:
            text.append(line + "\n")
    if not lines:
        text.append(diff_text)
    return text
