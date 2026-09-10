"""读文件工具（对齐 pi core/tools/read.ts 的 v1 子集）。

- 文本文件：UTF-8 读取，行数/字节双限截断，附继续读取提示
- offset（1 起始行号）与 limit（行数）支持
- 二进制检测：含 NUL 字节时报错（不误读二进制为文本）
- 图片：v1 Out of Scope（多模态输入裁剪）
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mimcode.agent.tools.base import AgentTool, ToolArgumentError
from mimcode.agent.tools.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size
from mimcode.types import ToolResult


def _resolve_path(raw: Any, cwd: str) -> Path:
    """参数路径 → 绝对 Path（相对路径以 cwd 解析）。"""
    if not isinstance(raw, str) or not raw.strip():
        raise ToolArgumentError("read: 'path' must be a non-empty string")
    path = Path(raw)
    return path if path.is_absolute() else (Path(cwd) / path)


class ReadTool(AgentTool):
    """读取文件内容。"""

    name = "read"
    description = (
        "Read the contents of a text file. Output is truncated to "
        f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
        "Use offset/limit for large files."
    )

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative or absolute)",
                },
                "offset": {
                    "type": "number",
                    "description": "Line number to start reading from (1-indexed)",
                },
                "limit": {"type": "number", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update=None,
    ) -> ToolResult:
        del tool_call_id, on_update
        path = _resolve_path(args.get("path"), self.cwd)
        offset = args.get("offset")
        limit = args.get("limit")

        self.check_aborted(signal)
        if not path.is_file():
            raise ToolArgumentError(f"File not found: {path}")
        if offset is not None and (not isinstance(offset, int) or offset < 1):
            raise ToolArgumentError("read: 'offset' must be a positive integer (1-indexed)")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ToolArgumentError("read: 'limit' must be a positive integer")

        self.check_aborted(signal)
        raw = await asyncio.to_thread(path.read_bytes)
        if b"\x00" in raw[:8192]:
            raise ToolArgumentError(
                f"read: {path} appears to be a binary file (binary reads are not supported)"
            )
        self.check_aborted(signal)
        text = raw.decode("utf-8", errors="replace")

        all_lines = text.splitlines()
        total_lines = len(all_lines)

        start_index = (offset - 1) if offset else 0
        if start_index >= total_lines:
            raise ToolArgumentError(
                f"Offset {offset} is beyond end of file ({total_lines} lines total)"
            )
        if limit is not None:
            selected = all_lines[start_index : start_index + limit]
        else:
            selected = all_lines[start_index:]

        # 用户 limit 优先；未指定时截断器兜底
        joined = "\n".join(selected) + ("\n" if selected else "")
        output_text = joined
        truncation_note = ""

        from mimcode.agent.tools.truncate import truncate_head

        if limit is None:
            truncation = truncate_head(joined)
            if truncation.truncated:
                output_text = truncation.content
                shown = truncation.output_lines + start_index
                truncation_note = (
                    f"\n\n[Truncated: showing {shown} of {total_lines} lines. "
                    f"Use offset={shown + 1} to continue.]"
                )
        elif start_index + limit < total_lines:
            next_offset = start_index + limit + 1
            remaining = total_lines - (start_index + limit)
            truncation_note = (
                f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            )

        details = {
            "path": str(path),
            "total_lines": total_lines,
            "start_line": start_index + 1,
            "bytes": len(raw),
        }
        return self.text_result(output_text + truncation_note, details=details)


class WriteTool(AgentTool):
    """写文件工具（对齐 pi core/tools/write.ts 的 v1 子集）：整文件覆写。"""

    name = "write"
    description = (
        "Write content to a file (overwrites existing content). Creates parent directories."
    )
    execution_mode = "sequential"

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative or absolute)",
                },
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update=None,
    ) -> ToolResult:
        del tool_call_id, on_update
        path = _resolve_path(args.get("path"), self.cwd)
        content = args.get("content")
        if not isinstance(content, str):
            raise ToolArgumentError("write: 'content' must be a string")

        self.check_aborted(signal)
        existed = path.is_file()
        original_bytes = path.stat().st_size if existed else 0

        await asyncio.to_thread(_write_file, path, content)
        self.check_aborted(signal)

        new_bytes = len(content.encode("utf-8"))
        action = "overwritten" if existed else "created"
        details = {
            "path": str(path),
            "original_bytes": original_bytes,
            "new_bytes": new_bytes,
        }
        return self.text_result(
            f"Successfully wrote to {path} ({action}, {format_size(new_bytes)}).",
            details=details,
        )


def _write_file(path: Path, content: str) -> None:
    """线程池写文件（父目录自动创建）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
