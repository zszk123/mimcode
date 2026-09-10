"""shell 执行工具（对齐 pi core/tools/bash.ts，Windows 自动降级 PowerShell）。

Windows 平台无 bash 时使用 PowerShell（``powershell -Command``）；
Unix 使用 ``bash -c``。超时与中止经进程终止实现。
输出截断：行数/字节双限（truncate.py），退出码非 0 视为错误文本附加。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from mimcode.agent.tools.base import AgentTool, ToolArgumentError
from mimcode.agent.tools.truncate import TruncationResult, format_size, truncate_head
from mimcode.types import ToolResult

_TIMEOUT_MAX_SECONDS = 2_147_483_647 / 1000
"""对齐 pi 的超时上限（ms 上限换算为秒）。"""


def _resolve_shell() -> tuple[str, list[str]]:
    """解析平台 shell：Windows → PowerShell，其余 → bash。

    Returns:
        (可执行文件, 固定参数前缀)；命令经 ``-c`` / ``-Command`` 传入。
    """
    if sys.platform == "win32":
        return "powershell", ["-NoProfile", "-Command"]
    return "bash", ["-c"]


def _resolve_timeout(timeout: float | None) -> float | None:
    """校验并归一超时秒数（对齐 pi resolveTimeoutMs 的校验语义）。"""
    if timeout is None:
        return None
    if timeout <= 0 or timeout != timeout:  # NaN 检查
        raise ToolArgumentError("Invalid timeout: must be a finite number of seconds")
    if timeout > _TIMEOUT_MAX_SECONDS:
        raise ToolArgumentError(f"Invalid timeout: maximum is {int(_TIMEOUT_MAX_SECONDS)} seconds")
    return timeout


class BashTool(AgentTool):
    """执行 shell 命令（Windows 上为 PowerShell）。"""

    name = "bash"
    description = (
        "Execute a shell command (PowerShell on Windows, bash on Unix). "
        f"Output is truncated to 2000 lines / {format_size(50 * 1024)}. "
        "Optional timeout in seconds."
    )
    execution_mode = "sequential"

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (optional, no default timeout)",
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update=None,
    ) -> ToolResult:
        """执行命令并收集合并输出（stdout+stderr）。"""
        del on_update, tool_call_id  # v1 不做流式进度；保留签名
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolArgumentError("bash: 'command' must be a non-empty string")
        timeout = _resolve_timeout(args.get("timeout"))

        self.check_aborted(signal)
        shell, prefix = _resolve_shell()
        proc = await asyncio.create_subprocess_exec(
            *shell.split() if sys.platform != "win32" else [shell],
            *prefix,
            command,
            cwd=self.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            proc.kill()
            stdout, stderr = await proc.communicate()

        self.check_aborted(signal)

        text = (stdout + stderr).decode("utf-8", errors="replace")
        truncation: TruncationResult | None = truncate_head(text) if text else None
        output = truncation.content if truncation else ""

        parts: list[str] = []
        exit_code = proc.returncode
        if exit_code is None:
            parts.append("[Process was killed]")
        elif exit_code != 0:
            parts.append(f"[Exit code: {exit_code}]")
        if timed_out:
            parts.append(f"[Timed out after {timeout}s]")
        if truncation is not None and truncation.truncated:
            limit_desc = (
                f"{truncation.output_lines} of {truncation.total_lines} lines"
                if truncation.truncated_by == "lines"
                else format_size(truncation.max_bytes)
            )
            parts.append(f"[Output truncated: {limit_desc}]")

        details: dict[str, Any] = {
            "exit_code": exit_code,
            "shell": shell,
        }
        if truncation is not None:
            details["truncation"] = {
                "truncated": truncation.truncated,
                "total_bytes": truncation.total_bytes,
            }
        return self.text_result("\n".join([output, *parts]).strip(), details=details)
