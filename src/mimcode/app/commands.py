"""slash 命令层（对齐 pi slash-commands.ts 的角色 + 分发语义）。

v1 命令集（checklist）：/model /thinking /resume /fork /compact /skills
/extensions /exit /help。

设计：
- ``CommandContext``：命令可触达的应用状态（会话/注册表/技能/插件/
  模型解析）——T12 TUI 与 T14 print 共用同一执行层
- ``CommandHandler``：纯异步函数（返回提示文本；不直接渲染）
- 执行结果统一 ``CommandResult``（output/error），交互层决定呈现方式

与 pi 的差异：pi 的命令在 interactive 模式内联实现（打开选择器 UI）；
mimcode 把「动作 + 状态变更」抽为可单测的纯逻辑层，UI 选择器在
T12 作为交互包装接入（参数直传时与 pi 行为一致）。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from typing import Any

from mimcode.app.session import SessionManager
from mimcode.provider.registry import Registry


@dataclasses.dataclass
class CommandContext:
    """命令执行上下文（应用状态句柄）。"""

    cwd: str
    session: SessionManager | None = None
    registry: Registry | None = None
    home: Any = None  # Path（测试注入），避免顶层导入 pathlib 循环
    # T11 接线：extensions 加载结果
    extension_names: list[str] = dataclasses.field(default_factory=list)
    # 技能加载结果（T9）
    skill_names: list[str] = dataclasses.field(default_factory=list)
    # 当前模型（/model 无参时展示用）
    current_model_id: str | None = None
    current_thinking_level: str | None = None


@dataclasses.dataclass(frozen=True)
class CommandResult:
    """命令执行结果。"""

    output: str = ""
    error: bool = False
    # 会话切换请求（resume/fork 产生，交互层执行切换）
    switch_session: SessionManager | None = None
    # 退出请求（/exit）
    request_exit: bool = False
    # 模型切换请求（/model，主循环接线）
    new_model_id: str | None = None
    new_thinking_level: str | None = None


CommandHandler = Callable[[CommandContext, str], Awaitable[CommandResult]]
"""命令处理器：``async def handler(ctx, args_text) -> CommandResult``。"""


@dataclasses.dataclass(frozen=True)
class SlashCommand:
    """命令注册条目。"""

    name: str
    description: str
    argument_hint: str | None = None
    handler: CommandHandler | None = None
    source: str = "builtin"  # builtin | extension


class CommandRegistry:
    """命令注册表：名字 → 命令；分发与补全数据源。"""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        """注册命令。

        Raises:
            ValueError: 重名。
        """
        if command.name in self._commands:
            raise ValueError(f"命令已存在: /{command.name}")
        self._commands[command.name] = command

    def get(self, name: str) -> SlashCommand | None:
        """按名取命令。"""
        return self._commands.get(name)

    def all(self) -> list[SlashCommand]:
        """全部命令（按名排序，补全/帮助用）。"""
        return [self._commands[name] for name in sorted(self._commands)]

    async def execute(self, name: str, args_text: str, context: CommandContext) -> CommandResult:
        """分发执行（未知命令 → 错误提示，不崩溃）。"""
        command = self._commands.get(name)
        if command is None:
            available = " ".join(f"/{entry.name}" for entry in self.all())
            return CommandResult(
                error=True,
                output=f"未知命令: /{name}。可用命令: {available}",
            )
        if command.handler is None:
            return CommandResult(error=True, output=f"/{name} 尚未接线")
        return await command.handler(context, args_text)


def parse_command_input(raw: str) -> tuple[str, str] | None:
    """解析用户输入为 (命令名, 参数文本)；非命令输入返回 None。

    语义：以 ``/`` 开头才视为命令（对齐 pi）；首空白分词，其余整体为参数。
    """
    stripped = raw.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped[1:].split(maxsplit=1)
    if not parts:
        return None
    name = parts[0]
    args = parts[1] if len(parts) > 1 else ""
    return name, args


# ---------------------------------------------------------------------------
# 内置命令处理器（九件，checklist）
# ---------------------------------------------------------------------------


async def _cmd_help(context: CommandContext, args: str) -> CommandResult:
    """/help：命令清单。"""
    del context, args
    registry = builtin_command_registry()
    lines = []
    for command in registry.all():
        hint = f" {command.argument_hint}" if command.argument_hint else ""
        lines.append(f"/{command.name}{hint} - {command.description}")
    return CommandResult(output="\n".join(lines))


async def _cmd_model(context: CommandContext, args: str) -> CommandResult:
    """/model：查询或切换模型。"""
    registry = context.registry
    if registry is None:
        return CommandResult(error=True, output="模型注册表不可用")
    arg = args.strip()
    if not arg:
        if context.current_model_id:
            models = "\n".join(f"  {model.provider}/{model.id}" for model in registry.all_models())
            return CommandResult(
                output=f"当前模型: {context.current_model_id}\n可用模型:\n{models}"
            )
        return CommandResult(error=True, output="用法: /model <provider/model 或 模型 id>")

    resolution = registry.resolve_model(arg)
    if resolution is None:
        return CommandResult(
            error=True,
            output=f"未找到模型: {arg}（--list-models 查看可用模型）",
        )
    return CommandResult(
        output=f"已切换模型: {resolution.endpoint_name}/{resolution.model.id}",
        new_model_id=resolution.model.id,
    )


async def _cmd_thinking(context: CommandContext, args: str) -> CommandResult:
    """/thinking：查询或设置思考级别。"""
    del context
    levels = "off / minimal / low / medium / high"
    arg = args.strip()
    if not arg:
        return CommandResult(output=f"用法: /thinking <level>（可选: {levels}）")
    if arg not in ("off", "minimal", "low", "medium", "high"):
        return CommandResult(error=True, output=f"无效级别: {arg}（可选: {levels}）")
    return CommandResult(output=f"已设置思考级别: {arg}", new_thinking_level=arg)


async def _cmd_resume(context: CommandContext, args: str) -> CommandResult:
    """/resume：恢复其他会话（无参 → 最近）。"""
    from mimcode.app.session_store import SessionFormatError, list_sessions, sessions_root

    del SessionFormatError
    home = context.home
    root = sessions_root(home) if home is not None else None
    if root is None:
        root = sessions_root()
    directory = root / _encode_cwd(context.cwd)

    arg = args.strip()
    if arg:
        # 按会话 id 精确恢复
        for info in list_sessions(directory):
            if info.id == arg:
                try:
                    session = SessionManager.open(info.path)
                except Exception as exc:  # noqa: BLE001 - 会话文件损坏归一为错误结果
                    return CommandResult(error=True, output=f"会话无法打开: {exc}")
                return CommandResult(
                    output=f"已恢复会话: {session.session_id}",
                    switch_session=session,
                )
        return CommandResult(error=True, output=f"未找到会话: {arg}")

    sessions = list_sessions(directory)
    if not sessions:
        return CommandResult(error=True, output="当前目录没有可恢复的会话")
    info = sessions[0]
    try:
        session = SessionManager.open(info.path)
    except Exception as exc:  # noqa: BLE001 - 会话文件损坏归一为错误结果
        return CommandResult(error=True, output=f"会话无法打开: {exc}")
    preview = info.first_message or "(空会话)"
    return CommandResult(
        output=f"已恢复会话: {session.session_id}（{preview}）",
        switch_session=session,
    )


async def _cmd_fork(context: CommandContext, args: str) -> CommandResult:
    """/fork：从当前（或指定 id）会话分叉新会话。"""
    del args
    session = context.session
    if session is None:
        return CommandResult(error=True, output="当前没有活动会话可分叉")
    forked = SessionManager.fork_from(session.session_file, context.cwd, home=context.home)
    return CommandResult(
        output=f"已分叉新会话: {forked.session_id}",
        switch_session=forked,
    )


async def _cmd_compact(context: CommandContext, args: str) -> CommandResult:
    """/compact：手动触发上下文压缩（需要模型/流函数时由 T12 注入执行器）。"""
    del args
    if context.session is None:
        return CommandResult(error=True, output="当前没有活动会话可压缩")

    # 无模型执行器时的降级提示（T12 接线后传入）
    executor = getattr(context, "compact_executor", None)
    if executor is None:
        return CommandResult(
            error=True,
            output="压缩需要模型连接（交互模式内执行 /compact）",
        )
    entry_id = await executor()
    if entry_id is None:
        return CommandResult(output="上下文未超阈值或摘要不可用，未压缩")
    return CommandResult(output=f"已压缩（条目 {entry_id}）")


async def _cmd_skills(context: CommandContext, args: str) -> CommandResult:
    """/skills：列出已加载技能。"""
    del args
    names = context.skill_names
    if not names:
        return CommandResult(output="没有已加载的技能")
    return CommandResult(output="已加载技能:\n" + "\n".join(f"  {name}" for name in names))


async def _cmd_extensions(context: CommandContext, args: str) -> CommandResult:
    """/extensions：列出已加载插件。"""
    del args
    names = context.extension_names
    if not names:
        return CommandResult(output="没有已加载插件")
    return CommandResult(output="已加载插件:\n" + "\n".join(f"  {name}" for name in names))


async def _cmd_exit(context: CommandContext, args: str) -> CommandResult:
    """/exit：退出请求。"""
    del context, args
    return CommandResult(output="再见", request_exit=True)


def _encode_cwd(cwd: str) -> str:
    """cwd → 会话目录编码名（复用 session_store 逻辑）。"""
    from mimcode.app.session_store import encode_cwd_to_dir_name

    return encode_cwd_to_dir_name(cwd)


def builtin_commands() -> list[SlashCommand]:
    """内置九命令（checklist）。"""
    return [
        SlashCommand(
            name="model",
            description="查询或切换模型",
            argument_hint="<provider/model 或模型 id>",
            handler=_cmd_model,
        ),
        SlashCommand(
            name="thinking",
            description="查询或设置思考级别",
            argument_hint="<level>",
            handler=_cmd_thinking,
        ),
        SlashCommand(
            name="resume",
            description="恢复其他会话（无参取最近）",
            argument_hint="[会话 id]",
            handler=_cmd_resume,
        ),
        SlashCommand(
            name="fork",
            description="从当前会话分叉新会话",
            handler=_cmd_fork,
        ),
        SlashCommand(
            name="compact",
            description="手动压缩上下文",
            handler=_cmd_compact,
        ),
        SlashCommand(
            name="skills",
            description="列出已加载技能",
            handler=_cmd_skills,
        ),
        SlashCommand(
            name="extensions",
            description="列出已加载插件",
            handler=_cmd_extensions,
        ),
        SlashCommand(
            name="exit",
            description="退出 mimcode",
            handler=_cmd_exit,
        ),
        SlashCommand(
            name="help",
            description="显示命令帮助",
            handler=_cmd_help,
        ),
    ]


_BUILTIN_REGISTRY: CommandRegistry | None = None


def builtin_command_registry() -> CommandRegistry:
    """内置命令注册表（惰性单例；扩展命令 T11 追加注册）。"""
    global _BUILTIN_REGISTRY
    if _BUILTIN_REGISTRY is None:
        registry = CommandRegistry()
        for command in builtin_commands():
            registry.register(command)
        _BUILTIN_REGISTRY = registry
    return _BUILTIN_REGISTRY
