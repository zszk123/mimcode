"""列目录 / 查找文件 / 搜索内容（对齐 pi ls.ts / find.ts / grep.ts 的 v1 子集）。

与 pi 的差异：pi 用 ripgrep 子进程；mimcode 用纯 Python 实现
（pathspec 处理 .gitignore 规则），行为语义保持一致。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from mimcode.agent.tools.base import AgentTool, ToolArgumentError
from mimcode.agent.tools.ignore import is_ignored, load_ignore_specs
from mimcode.agent.tools.truncate import GREP_MAX_LINE_LENGTH, truncate_head
from mimcode.types import ToolResult

DEFAULT_GREP_LIMIT = 100
"""grep 默认最大命中数（对齐 pi DEFAULT_LIMIT）。"""

LS_DIRECTORY_MARKER = "/"
"""ls 输出中目录的后缀标记。"""


def _resolve_dir(raw: Any, cwd: str) -> Path:
    if raw is None:
        return Path(cwd)
    if not isinstance(raw, str) or not raw.strip():
        raise ToolArgumentError("'path' must be a non-empty string")
    path = Path(raw)
    return path if path.is_absolute() else (Path(cwd) / path)


class LsTool(AgentTool):
    """列目录内容（文件/目录混排，目录带 / 后缀）。"""

    name = "ls"
    description = (
        "List directory contents. Directories are marked with a trailing '/'. "
        "Entries respect .gitignore. Hidden files are included."
    )

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list (default: current directory)",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update=None,
    ) -> ToolResult:
        del tool_call_id, on_update
        path = _resolve_dir(args.get("path"), self.cwd)

        self.check_aborted(signal)
        if not path.is_dir():
            raise ToolArgumentError(f"Not a directory: {path}")

        specs = load_ignore_specs(path)
        entries = sorted(path.iterdir(), key=lambda entry: entry.name)
        lines: list[str] = []
        for entry in entries:
            if is_ignored(entry, path, specs):
                continue
            if entry.is_dir():
                lines.append(entry.name + LS_DIRECTORY_MARKER)
            else:
                lines.append(entry.name)
        output = "\n".join(lines)
        details = {"path": str(path), "entries": len(lines)}
        return self.text_result(output, details=details)


class FindTool(AgentTool):
    """按名称模式查找文件（glob 语法）。"""

    name = "find"
    description = (
        "Find files by glob pattern (e.g. '*.py' or '**/*.ts'). Respects .gitignore. "
        "Searches recursively from the given path (default: current directory)."
    )

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match file names (e.g. '*.py')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search from (default: current directory)",
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update=None,
    ) -> ToolResult:
        del tool_call_id, on_update
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ToolArgumentError("find: 'pattern' must be a non-empty glob")
        root = _resolve_dir(args.get("path"), self.cwd)

        self.check_aborted(signal)
        if not root.is_dir():
            raise ToolArgumentError(f"Not a directory: {root}")

        specs = load_ignore_specs(root)
        found: list[Path] = []
        for candidate in root.rglob(pattern):
            self.check_aborted(signal)
            if candidate.is_symlink() or ".git" in candidate.parts[len(root.parts) :]:
                continue
            if is_ignored(candidate, root, specs):
                continue
            found.append(candidate)

        found.sort()
        lines = [path.relative_to(root).as_posix() for path in found]
        truncation = truncate_head("\n".join(lines))
        note = (
            f"\n\n[Truncated: showing {truncation.output_lines} "
            f"of {truncation.total_lines} results.]"
            if truncation.truncated
            else ""
        )
        details = {"root": str(root), "pattern": pattern, "matches": len(lines)}
        return self.text_result(truncation.content + note, details=details)


class GrepTool(AgentTool):
    """按正则/字面量搜索文件内容。"""

    name = "grep"
    description = (
        "Search file contents for a pattern. Returns matching lines with file paths "
        f"and line numbers. Respects .gitignore. "
        f"Output is truncated to {DEFAULT_GREP_LIMIT} matches. "
        f"Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars."
    )

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex or literal string)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default: current directory)",
                },
                "glob": {
                    "type": "string",
                    "description": "Filter files by glob pattern, e.g. '*.py'",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)",
                },
                "is_regex": {
                    "type": "boolean",
                    "description": "Treat pattern as regex (default: true)",
                },
                "limit": {
                    "type": "number",
                    "description": (
                        f"Maximum number of matches to return (default: {DEFAULT_GREP_LIMIT})"
                    ),
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update=None,
    ) -> ToolResult:
        del tool_call_id, on_update
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ToolArgumentError("grep: 'pattern' must be a non-empty string")
        root = _resolve_dir(args.get("path"), self.cwd)
        glob = args.get("glob")
        if glob is not None and not isinstance(glob, str):
            raise ToolArgumentError("grep: 'glob' must be a string")
        ignore_case = bool(args.get("ignore_case", False))
        is_regex = bool(args.get("is_regex", True))
        limit = args.get("limit", DEFAULT_GREP_LIMIT)
        if not isinstance(limit, int) or limit < 1:
            raise ToolArgumentError("grep: 'limit' must be a positive integer")

        flags = re.IGNORECASE if ignore_case else 0
        try:
            matcher = re.compile(pattern if is_regex else re.escape(pattern), flags)
        except re.error as exc:
            raise ToolArgumentError(f"grep: invalid pattern: {exc}") from exc

        self.check_aborted(signal)
        if root.is_file():
            files = [root]
            base = root.parent
        else:
            if not root.is_dir():
                raise ToolArgumentError(f"Not a directory or file: {root}")
            specs = load_ignore_specs(root)
            files = [
                candidate
                for candidate in root.rglob(glob or "*")
                if candidate.is_file()
                and not candidate.is_symlink()
                and ".git" not in candidate.parts[len(root.parts) :]
                and not is_ignored(candidate, root, specs)
            ]
            base = root

        lines_out: list[str] = []
        match_count = 0
        truncated = False
        for file in sorted(files):
            self.check_aborted(signal)
            if match_count >= limit:
                truncated = True
                break
            try:
                content = await asyncio.to_thread(file.read_text, "utf-8")
            except (OSError, UnicodeDecodeError):
                continue  # 二进制/不可读文件跳过（对齐 rg 行为）
            for line_number, line in enumerate(content.splitlines(), start=1):
                if matcher.search(line):
                    if match_count >= limit:
                        truncated = True
                        break
                    display = line.strip()
                    if len(display) > GREP_MAX_LINE_LENGTH:
                        display = display[:GREP_MAX_LINE_LENGTH] + "…"
                    rel = file.relative_to(base)
                    lines_out.append(f"{rel.as_posix()}:{line_number}:{display}")
                    match_count += 1

        note = f"\n\n[Truncated: {limit} match limit reached.]" if truncated else ""
        details = {"matches": match_count, "limit": limit, "truncated": truncated}
        return self.text_result("\n".join(lines_out) + note, details=details)
