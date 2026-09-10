"""工具输出截断（对齐 pi core/tools/truncate.ts）。

双限截断（先命中者生效）：
- 行数上限（默认 2000 行）
- 字节数上限（默认 50KB）

截断永远不产生半行（尾截断的 bash 边界情形除外，v1 不含 bash 尾截断）。
"""

from __future__ import annotations

import dataclasses
from typing import Literal

DEFAULT_MAX_LINES = 2000
"""默认行数上限（对齐 pi DEFAULT_MAX_LINES）。"""

DEFAULT_MAX_BYTES = 50 * 1024
"""默认字节上限（对齐 pi DEFAULT_MAX_BYTES = 50KB）。"""

GREP_MAX_LINE_LENGTH = 500
"""grep 匹配行的最大字符数（对齐 pi GREP_MAX_LINE_LENGTH）。"""


@dataclasses.dataclass(frozen=True)
class TruncationResult:
    """截断结果元数据。"""

    content: str
    truncated: bool
    truncated_by: Literal["lines", "bytes"] | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int


def _split_lines_for_counting(content: str) -> list[str]:
    """按换行切行计数（结尾换行不计尾空行，对齐 pi）。"""
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def format_size(num_bytes: int) -> str:
    """字节数的人类可读格式（对齐 pi formatSize）。"""
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """头部截断：保留前 N 行/字节（适合读文件场景）。

    不产生半行；首行本身超过字节上限时返回空内容并置
    first_line_exceeds_limit（对齐 pi truncateHead 语义）。
    """
    total_lines = len(_split_lines_for_counting(content))
    total_bytes = len(content.encode("utf-8"))

    lines = _split_lines_for_counting(content)
    kept: list[str] = []
    size = 0
    truncated = False
    truncated_by: Literal["lines", "bytes"] | None = None
    first_line_exceeds = False

    if lines and len(lines[0].encode("utf-8")) > max_bytes:
        # 首行独超上限：返回空并标记（对齐 pi：指向 bash 兜底）
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    for line in lines:
        line_bytes = len(line.encode("utf-8")) + 1  # +1 换行
        if len(kept) >= max_lines:
            truncated, truncated_by = True, "lines"
            break
        if size + line_bytes > max_bytes:
            truncated, truncated_by = True, "bytes"
            break
        kept.append(line)
        size += line_bytes

    output = "\n".join(kept) + ("\n" if kept else "")
    return TruncationResult(
        content=output,
        truncated=truncated,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(kept),
        output_bytes=len(output.encode("utf-8")),
        first_line_exceeds_limit=first_line_exceeds,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )
