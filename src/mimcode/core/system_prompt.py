"""系统提示组装（对齐 pi core/system-prompt.ts 的角色，v1 子集）。

结构：基础身份 + 技能清单注入（T9）+ 工具使用守则。
"""

from __future__ import annotations

from mimcode.app.skills import LoadSkillsResult, format_skills_for_prompt

BASE_PROMPT = """You are mimcode, a terminal-based AI coding assistant.

You help with software engineering tasks: reading and editing code, running
commands, and answering questions about the codebase. Prefer concrete,
verifiable actions over speculation. When a task needs files or commands,
use the available tools instead of asking the user to run them.

Keep answers concise and technical."""

TOOL_GUIDELINES = """
## Tool usage

- Use bash/read/write/edit/ls/find/grep tools to inspect and modify files.
- For file edits, prefer the edit tool with exact unique text replacement.
- Shell commands run in the project directory (PowerShell on Windows).
- Tool outputs are truncated; continue reading with offset when needed."""


def build_system_prompt(skills: LoadSkillsResult | None = None) -> str:
    """组装系统提示：基础身份 + 技能注入 + 工具守则。

    Args:
        skills: 技能加载结果（None 跳过注入）。
    """
    parts = [BASE_PROMPT]
    if skills is not None:
        skills_block = format_skills_for_prompt(skills.skills)
        if skills_block:
            parts.append(skills_block)
    parts.append(TOOL_GUIDELINES)
    return "\n".join(parts)
