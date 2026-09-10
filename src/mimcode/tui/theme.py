"""主题系统（对齐 pi theme.ts 的角色语义，v1 双主题）。

语义角色：user / assistant / tool / error / thinking / system / accent。
每角色映射到 Rich 样式字符串；两套内置主题（dark / light）。
"""

from __future__ import annotations

import dataclasses

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"
ROLE_ERROR = "error"
ROLE_THINKING = "thinking"
ROLE_SYSTEM = "system"
ROLE_ACCENT = "accent"

ALL_ROLES = (
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_ERROR,
    ROLE_THINKING,
    ROLE_SYSTEM,
    ROLE_ACCENT,
)


@dataclasses.dataclass(frozen=True)
class Theme:
    """主题：语义角色 → Rich 样式（颜色/加粗等）。"""

    name: str
    styles: dict[str, str]


DARK = Theme(
    name="dark",
    styles={
        ROLE_USER: "bold cyan",
        ROLE_ASSISTANT: "bright_white",
        ROLE_TOOL: "yellow",
        ROLE_ERROR: "bold red",
        ROLE_THINKING: "italic magenta",
        ROLE_SYSTEM: "grey50",
        ROLE_ACCENT: "bold green",
    },
)

LIGHT = Theme(
    name="light",
    styles={
        ROLE_USER: "bold blue",
        ROLE_ASSISTANT: "black",
        ROLE_TOOL: "dark_orange",
        ROLE_ERROR: "bold red3",
        ROLE_THINKING: "italic medium_purple4",
        ROLE_SYSTEM: "grey42",
        ROLE_ACCENT: "bold green4",
    },
)

THEMES: dict[str, Theme] = {DARK.name: DARK, LIGHT.name: LIGHT}
"""内置主题表（按名取）。"""

DEFAULT_THEME = DARK


def get_theme(name: str) -> Theme:
    """按名取主题（未知名回落默认）。"""
    return THEMES.get(name, DEFAULT_THEME)
