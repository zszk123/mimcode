"""Markdown 渲染（Rich Markdown，对齐 pi assistant-message 的 markdown 语义）。

流式期间纯文本追加（打字机光标），定稿后 markdown 渲染——
与 pi 一致：流式期间 plain text，message_end 后渲染完整块。

流式 markdown 容错（对齐 pi markdown-transform 的角色）：
Rich 的 Markdown 解析器对不完整语法宽容（未闭合代码块按普通段落渲染）。
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.markdown import Markdown


def markdown_renderable(text: str) -> Markdown:
    """正文文本 → Rich Markdown 渲染对象。"""
    return Markdown(text)


def render_markdown(text: str, *, width: int = 80, theme_name: str | None = None) -> str:
    """把 markdown 文本渲染为字符串（测试与 presenter 复用）。"""
    buffer = StringIO()
    console = Console(file=buffer, width=width, highlight=False)
    console.print(markdown_renderable(text))
    return buffer.getvalue()
