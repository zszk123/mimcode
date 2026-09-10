"""Rich 呈现层：RenderAction → 终端输出（T13 高保真渲染）。

职责（替换 T12 骨架的 apply_action 默认实现，管线动作不变）：
- write_line(assistant)：Rich Markdown 渲染正文（对齐 pi：定稿后 markdown）
- write_line(其他角色)：主题语义色行
- stream_chunk：纯文本追加（打字机光标，对齐 pi 流式 plain text）
- 工具结果含 diff（edit 工具 details）：diff 语义色渲染
"""

from __future__ import annotations

import sys
from typing import TextIO

from rich.console import Console
from rich.text import Text

from mimcode.tui import theme as theme_module
from mimcode.tui.diff import diff_renderable
from mimcode.tui.markdown import markdown_renderable
from mimcode.tui.renderer import RenderAction
from mimcode.tui.theme import Theme

ASSISTANT_PREFIX = "mim:"


class RichPresenter:
    """RenderAction 的 Rich 终端呈现。"""

    def __init__(
        self,
        *,
        theme: Theme | None = None,
        stream: TextIO = sys.stdout,
        width: int = 80,
    ) -> None:
        self.theme = theme or theme_module.DEFAULT_THEME
        # Console 输出到 stream（测试注入 StringIO）
        self.console = Console(file=stream, width=width, highlight=False)

    def set_theme(self, theme: Theme) -> None:
        """切换主题（checklist：切换后取色随主题）。"""
        self.theme = theme

    def style_for(self, role: str) -> str:
        """语义角色 → 当前主题样式（checklist 取色断言入口）。"""
        return self.theme.styles.get(role, "")

    # ------------------------------------------------------------------

    def apply_action(self, action: RenderAction) -> None:
        """落地一个渲染动作。"""
        if action.kind == "write_line":
            self._write_line(action)
        elif action.kind == "stream_chunk":
            # 流式打字机：纯文本追加 + flush（终端光标即动画）
            self.console.file.write(action.text)
            self.console.file.flush()
        elif action.kind == "set_busy":
            pass  # busy 状态由 TuiState 承载
        elif action.kind == "clear_busy":
            self.console.file.write("\n")
        elif action.kind == "footer":
            self.console.print(Text(action.text, style=self.style_for(theme_module.ROLE_SYSTEM)))
        elif action.kind == "error":
            self.console.print(
                Text(f"错误: {action.text}", style=self.style_for(theme_module.ROLE_ERROR))
            )

    def _write_line(self, action: RenderAction) -> None:
        """行落地：assistant 正文走 markdown；其余主题色行。"""
        text = action.text
        if action.style == "assistant" and text.startswith(ASSISTANT_PREFIX):
            body = text[len(ASSISTANT_PREFIX) :].strip()
            if body:
                self.console.print(
                    self._role_label(action.style), style=self.style_for(action.style), end=""
                )
                self.console.file.write("\n")
                self.console.print(markdown_renderable(body))
                return
        self.console.print(Text(text, style=self.style_for(action.style)))

    def _role_label(self, role: str) -> str:
        """角色前缀标签。"""
        return f"{ASSISTANT_PREFIX} "

    # ------------------------------------------------------------------
    # diff 渲染（edit 工具结果，checklist）
    # ------------------------------------------------------------------

    def render_diff(self, diff_text: str) -> str:
        """diff 文本 → 渲染字符串（测试捕获）。"""
        from io import StringIO

        buffer = StringIO()
        capture = Console(file=buffer, width=self.console.width, highlight=False)
        capture.print(diff_renderable(diff_text))
        return buffer.getvalue()
