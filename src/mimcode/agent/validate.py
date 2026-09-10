"""工具参数验证（对齐 pi packages/ai/src/utils/validation.ts 的 validateToolArguments）。

语义子集：必填缺失 / 类型不匹配 → 抛错，错误信息含字段路径
与收到的参数（对齐 pi 的错误格式）；不做 TypeBox 的类型强转
（保守拒绝，模型重发）。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import jsonschema

if TYPE_CHECKING:
    from mimcode.agent.tools.base import AgentTool
    from mimcode.types import ToolCallBlock


def validate_tool_arguments(tool: AgentTool, tool_call: ToolCallBlock) -> dict[str, Any]:
    """按工具 schema 验证调用参数。

    Args:
        tool: 目标工具。
        tool_call: 模型发出的调用块（arguments 已尽力解析）。

    Returns:
        验证通过的参数（原 dict 引用）。

    Raises:
        ValueError: 验证失败（错误含字段路径与收到的参数，
        对齐 pi 的错误文本语义，供模型自纠错）。
    """
    schema = tool.parameters_schema()
    validator = jsonschema.Draft202012Validator(schema)
    args = tool_call.arguments

    errors = sorted(validator.iter_errors(args), key=lambda error: list(error.absolute_path))
    if errors:
        lines = [f"  - {format_validation_path(error)}: {error.message}" for error in errors]
        error_message = (
            f'Validation failed for tool "{tool_call.name}":\n'
            f"{chr(10).join(lines)}\n\n"
            f"Received arguments:\n{json.dumps(args, indent=2, ensure_ascii=False)}"
        )
        raise ValueError(error_message)
    return args


def format_validation_path(error: jsonschema.ValidationError) -> str:
    """字段路径的展示格式（对齐 pi formatValidationPath）。"""
    path = list(error.absolute_path)
    if not path:
        return "(root)"
    return ".".join(str(segment) for segment in path)
