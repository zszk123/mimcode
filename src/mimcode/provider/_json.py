"""流式工具调用参数的 JSON 解析（对齐 pi utils/json-parse.ts 的语义子集）。

工具调用的 arguments 以 JSON 文本增量到达。pi 的策略：
1. 直接 json.loads；
2. 失败则做「尽力修复」：补齐未闭合的字符串与容器后重试；
3. 仍失败返回空对象（截断保护在 agent 层做：stopReason=length 时
   整批工具调用会被判错，不会执行，见 T6 的迁移语义）。
"""

from __future__ import annotations

import json
from typing import Any


def partial_json_loads(partial: str | None) -> dict[str, Any]:
    """尽力解析一段可能不完整的 JSON 文本。

    Args:
        partial: 累积中的 JSON 片段（工具调用 arguments）。

    Returns:
        解析出的对象；不可解析或顶层非对象时返回空 dict（不抛异常）。
    """
    if not partial or not partial.strip():
        return {}
    try:
        result = json.loads(partial)
    except json.JSONDecodeError:
        repaired = _close_partial_json(partial)
        try:
            result = json.loads(repaired)
        except json.JSONDecodeError:
            return {}
    return result if isinstance(result, dict) else {}


def _close_partial_json(text: str) -> str:
    """补齐未闭合的字符串字面量与数组/对象容器。

    逐字符扫描维护状态机（字符串/转义/容器栈/最后有效字符），
    结束时：闭合未终结的字符串、剥离悬挂逗号、冒号后补占位
    null、逆序闭合打开的容器。
    """
    in_string = False
    escaped = False
    stack: list[str] = []
    last_nonspace = ""

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                last_nonspace = '"'
            continue
        if char == '"':
            in_string = True
            last_nonspace = '"'
        elif char in "{[":
            stack.append(char)
            last_nonspace = char
        elif char in "}]":
            if stack:
                stack.pop()
            last_nonspace = char
        elif not char.isspace():
            last_nonspace = char

    repaired = text
    if in_string:
        # 未闭合的字符串：截掉不完整的转义序列后补引号
        if escaped:
            repaired = repaired[:-1]
        repaired += '"'

    stripped = repaired.rstrip()
    if stack and stripped.endswith(","):
        # 悬挂逗号（截断常见形态）：直接剥离
        repaired = stripped[:-1]
    elif stack and last_nonspace == ":":
        # 冒号后缺值：补占位 null
        repaired += "null"

    # 逆序闭合所有打开的容器
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired
