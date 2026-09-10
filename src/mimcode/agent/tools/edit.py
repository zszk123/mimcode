"""编辑文件工具（对齐 pi core/tools/edit.ts 的 v1 子集）。

核心语义（对齐 pi applyEditsToNormalizedContent）：
- 多编辑一次应用：每个 oldText 对原文（应用前）匹配，非互相叠加
- oldText 必须在文件中唯一（occurrences > 1 → 报错含次数）
- 编辑区间不得重叠（嵌套）→ 报错
- 全部应用后无变化 → 报错
- CRLF 归一：匹配在 LF 空间进行，写回保留原换行风格
- BOM：匹配前剥离，写回保留

v1 裁剪：不做模糊匹配（pi 的 fuzzyFindText），纯精确匹配。
diff：结果 details 携带 unified diff（difflib 生成）。
"""

from __future__ import annotations

import asyncio
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimcode.agent.tools.base import AgentTool, ToolArgumentError
from mimcode.types import ToolResult


@dataclass(frozen=True)
class Edit:
    """单条替换。"""

    old_text: str
    new_text: str


def _resolve_path(raw: Any, cwd: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ToolArgumentError("edit: 'path' must be a non-empty string")
    path = Path(raw)
    return path if path.is_absolute() else (Path(cwd) / path)


def _normalize_edits(raw: Any) -> list[Edit]:
    """归一 edits 参数（容忍 JSON 字符串形态与单对象形态，对齐 pi）。"""
    if isinstance(raw, str):
        import json

        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None

    if isinstance(raw, dict) and "oldText" in raw and "newText" in raw:
        raw = [raw]

    if not isinstance(raw, list) or not raw:
        raise ToolArgumentError(
            "edit: 'edits' must contain at least one replacement "
            "(each entry has oldText and newText)"
        )
    edits: list[Edit] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ToolArgumentError("edit: each entry in 'edits' must be an object")
        old_text = entry.get("oldText")
        new_text = entry.get("newText")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ToolArgumentError(
                "edit: each entry in 'edits' must have string 'oldText' and 'newText'"
            )
        edits.append(Edit(old_text=old_text, new_text=new_text))
    return edits


def _split_bom(raw: str) -> tuple[str, str]:
    """剥离 BOM（对齐 pi splitBom：模型不会在 oldText 里带不可见 BOM）。"""
    if raw.startswith("\ufeff"):
        return "\ufeff", raw[1:]
    return "", raw


def _detect_line_ending(content: str) -> str:
    """检测主导换行风格（对齐 pi detectLineEnding）。"""
    if content.count("\r\n") > content.count("\n") - content.count("\r\n"):
        return "\r\n"
    return "\n"


def apply_edits(content: str, edits: list[Edit], path: str) -> tuple[str, str]:
    """在 LF 归一空间应用编辑序列，返回 (原文, 新文)。

    Raises:
        ToolArgumentError: 空 oldText / 未命中 / 多处命中 / 重叠 / 无变化。
    """
    normalized = content.replace("\r\n", "\n")
    normalized_edits = [
        Edit(
            old_text=edit.old_text.replace("\r\n", "\n"),
            new_text=edit.new_text.replace("\r\n", "\n"),
        )
        for edit in edits
    ]

    for index, edit in enumerate(normalized_edits):
        if not edit.old_text:
            raise ToolArgumentError(
                f"edits[{index}].oldText is empty. Provide the exact text to replace "
                f"(with a few surrounding lines for uniqueness if needed) in {path}."
            )

    matches: list[tuple[int, int, int, str]] = []  # (edit_index, start, length, new_text)
    for index, edit in enumerate(normalized_edits):
        start = normalized.find(edit.old_text)
        if start == -1:
            raise ToolArgumentError(
                f"edits[{index}].oldText was not found in {path}. "
                "Make sure it matches the file exactly (including whitespace and indentation)."
            )
        occurrences = normalized.count(edit.old_text)
        if occurrences > 1:
            raise ToolArgumentError(
                f"edits[{index}].oldText was found {occurrences} times in {path}. "
                "Include more surrounding lines to make it unique."
            )
        matches.append((index, start, len(edit.old_text), edit.new_text))

    matches.sort(key=lambda item: item[1])
    for previous, current in zip(matches, matches[1:], strict=False):
        if previous[1] + previous[2] > current[1]:
            raise ToolArgumentError(
                f"edits[{previous[0]}] and edits[{current[0]}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    new_parts: list[str] = []
    cursor = 0
    for _, start, length, replacement in matches:
        new_parts.append(normalized[cursor:start])
        new_parts.append(replacement)
        cursor = start + length
    new_parts.append(normalized[cursor:])
    new_content = "".join(new_parts)

    if normalized == new_content:
        raise ToolArgumentError(
            f"None of the edits changed {path} (oldText equals newText). "
            "Omit the edit call if no change is intended."
        )
    return normalized, new_content


def _unified_diff(path: str, old: str, new: str) -> str:
    """生成 unified diff（对齐 pi generateUnifiedPatch 的角色，context=4）。"""
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=path,
        tofile=path,
        n=4,
    )
    return "".join(diff)


class EditTool(AgentTool):
    """精确文本替换编辑工具。"""

    name = "edit"
    description = (
        "Edit a file using exact text replacement. Every edits[].oldText must match "
        "a unique, non-overlapping region of the original file. Each oldText is matched "
        "against the original file, not after earlier edits are applied."
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
                "edits": {
                    "type": "array",
                    "description": "List of replacements to apply in one call",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": "Exact text to replace (must be unique in the file)",
                            },
                            "newText": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["oldText", "newText"],
                    },
                },
            },
            "required": ["path", "edits"],
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
        edits = _normalize_edits(args.get("edits"))

        self.check_aborted(signal)
        if not path.is_file():
            raise ToolArgumentError(f"Could not edit file: {path} (file not found).")

        raw_bytes = await asyncio.to_thread(path.read_bytes)
        raw = raw_bytes.decode("utf-8")
        self.check_aborted(signal)

        bom, text = _split_bom(raw)
        original_ending = _detect_line_ending(text)
        base_content, new_content = apply_edits(text, edits, str(path))
        self.check_aborted(signal)

        final_content = bom + new_content.replace("\n", original_ending)
        await asyncio.to_thread(_write_file, path, final_content)
        self.check_aborted(signal)

        diff = _unified_diff(str(path), base_content, new_content)
        details = {
            "diff": diff,
            "edits_applied": len(edits),
            "path": str(path),
        }
        return self.text_result(
            f"Successfully replaced {len(edits)} block(s) in {path}.",
            details=details,
        )


def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="")
