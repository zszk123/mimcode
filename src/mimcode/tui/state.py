"""TUI 会话状态（footer 数据源与渲染缓冲）。

设计（对齐 pi 的 REPL 式交互，而非全屏 Application）：
- 状态由事件流驱动（RendererPipeline 更新）
- footer 摘要由状态导出（模型/会话/用量）
"""

from __future__ import annotations

import dataclasses

from mimcode.types import Usage


@dataclasses.dataclass
class TuiState:
    """交互会话的界面状态。"""

    current_model_id: str | None = None
    session_id: str | None = None
    busy: bool = False
    """agent 运行中（流式响应/工具执行）。"""

    usage_total: Usage = dataclasses.field(default_factory=Usage)
    """本会话累计用量（assistant 消息用量累加）。"""

    turns: int = 0
    """已完成的轮数。"""

    def add_usage(self, usage: Usage) -> None:
        """累计一条助手消息的用量。"""
        self.usage_total = Usage(
            input=self.usage_total.input + usage.input,
            output=self.usage_total.output + usage.output,
            cache_read=self.usage_total.cache_read + usage.cache_read,
            cache_write=self.usage_total.cache_write + usage.cache_write,
            reasoning=self.usage_total.reasoning + usage.reasoning,
        )

    def footer_line(self) -> str:
        """footer 摘要行（模型 / 会话 / 轮数 / token）。"""
        parts: list[str] = []
        if self.current_model_id:
            parts.append(self.current_model_id)
        if self.session_id:
            parts.append(f"会话 {self.session_id[:8]}")
        parts.append(f"{self.turns} 轮")
        tokens = self.usage_total.input + self.usage_total.output
        if tokens:
            parts.append(f"{tokens} tokens")
        return " | ".join(parts)
