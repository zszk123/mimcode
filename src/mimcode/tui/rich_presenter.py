"""Rich 呈现层：RenderAction → 终端输出（T13 高保真渲染）。

职责（替换 T12 骨架的 apply_action 默认实现，管线动作不变）：
- write_line(assistant)：Rich Markdown 渲染正文（对齐 pi：定稿后 markdown）
- write_line(其他角色)：主题语义色行
- 工具结果含 diff（edit 工具 details）：diff 语义色渲染
- 流式打字机（仅 TTY）：text_delta 纯文本追加；定稿 clear_stream
  擦除流式区后整体重绘——对齐 pi「流式 plain text → 定稿 markdown」，
  输出无重复
- 非 TTY（管道/测试捕获）：流式区与瞬时指示行不打印，正文由定稿
  markdown 承载
"""

from __future__ import annotations

import math
import sys
from typing import TextIO

from rich.cells import cell_len
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
        self._stream_parts: list[str] = []
        """流式打字文本（TTY 折行计数用）。"""
        self._erasable_lines = 0
        """瞬时指示行计数（思考中等），随流式区一并擦除。"""

    @property
    def _isatty(self) -> bool:
        """输出流是否真终端（决定打字机与 ANSI 擦除）。"""
        return bool(self.console.file.isatty())

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
            if action.erasable and not self._isatty:
                return  # 瞬时指示行不进入非交互输出
            self._write_line(action)
            if action.erasable:
                self._erasable_lines += 1
            if action.diff:
                self.console.print(diff_renderable(action.diff))
        elif action.kind == "stream_chunk":
            # 流式打字机（仅 TTY）：纯文本追加 + flush（终端光标即动画）
            if self._isatty:
                self.console.file.write(action.text)
                self.console.file.flush()
                self._stream_parts.append(action.text)
        elif action.kind == "clear_stream":
            self._clear_stream()
        elif action.kind == "set_busy":
            pass  # busy 状态由 TuiState 承载
        elif action.kind == "clear_busy":
            self._reset_stream_region()
            self.console.file.write("\n")
        elif action.kind == "footer":
            self.console.print(Text(action.text, style=self.style_for(theme_module.ROLE_SYSTEM)))
        elif action.kind == "error":
            self.console.print(
                Text(f"错误: {action.text}", style=self.style_for(theme_module.ROLE_ERROR))
            )

    def _clear_stream(self) -> None:
        """擦除流式打字区（TTY）：定稿内容在其上整体重绘。"""
        if self._isatty:
            text = "".join(self._stream_parts)
            total = self._wrapped_lines(text) + self._erasable_lines
            if total > 0:
                # 光标上移 total-1 行回到区域首行，回行首后清至屏尾
                self.console.file.write(f"\x1b[{total - 1}A\r\x1b[J")
                self.console.file.flush()
        self._reset_stream_region()

    def _reset_stream_region(self) -> None:
        """重置流式区跟踪。"""
        self._stream_parts = []
        self._erasable_lines = 0

    def _wrapped_lines(self, text: str) -> int:
        """流式文本占用的终端行数（CJK 宽度感知的折行计数）。"""
        if not text:
            return 0
        width = max(1, self.console.width)
        total = 0
        for segment in text.split("\n"):
            if segment:
                total += max(1, math.ceil(cell_len(segment) / width))
            else:
                total += 1
        return total

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
