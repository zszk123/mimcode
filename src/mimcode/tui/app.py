"""Interactive 模式骨架（prompt_toolkit REPL + 事件消费 + 渲染落地）。

架构（对齐 pi interactive-mode 的职责边界，REPL 形态）：
- ``InteractiveApp``：PromptSession 驱动输入；agent 运行时事件经
  RendererPipeline 转为动作后落地终端
- 中断分级（对齐 pi）：Esc 运行中 → 中止请求；Ctrl-D → 退出
  （Ctrl-C 空闲 → 退出，运行中 → 中断）
- 命令输入（/ 开头）经 CommandRegistry 分发

渲染落地：v1 骨架用行式打印；T13 高保真渲染替换呈现层
（管线动作不变）。

测试策略：完整 REPL 交互留 T15 e2e（PTY）；本模块的
``run_agent_turn``（事件消费 + 渲染执行）与输入/命令分发可 headless 单测。
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from typing import TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from mimcode.agent.loop import AgentContext, AgentLoopConfig, agent_loop
from mimcode.app.commands import (
    CommandContext,
    CommandResult,
    builtin_command_registry,
    parse_command_input,
)
from mimcode.tui.renderer import RenderAction, RendererPipeline
from mimcode.tui.state import TuiState
from mimcode.types import AgentEvent, UserMessage

INPUT_MAX_CHARS = 100_000
"""输入长度软上限（超限提示）。"""

ConfigFactory = Callable[[], AgentLoopConfig]
"""每轮 agent 的配置工厂（模型切换后重建配置）。"""


class InteractiveApp:
    """交互式 REPL 应用骨架。"""

    def __init__(
        self,
        *,
        cwd: str,
        agent_context: AgentContext,
        config_factory: ConfigFactory,
        command_context_factory: Callable[[], CommandContext] | None = None,
        stream: TextIO = sys.stdout,
        exit_event: asyncio.Event | None = None,
    ) -> None:
        self.cwd = cwd
        self.agent_context = agent_context
        self.config_factory = config_factory
        self.command_context_factory = command_context_factory
        self.state = TuiState()
        self.pipeline = RendererPipeline(self.state)
        self.stream = stream
        self._interrupt_event = asyncio.Event()
        self._exit_event = exit_event or asyncio.Event()
        self._history = InMemoryHistory()
        self.command_registry = builtin_command_registry()

    # ------------------------------------------------------------------
    # 渲染动作落地
    # ------------------------------------------------------------------

    def apply_action(self, action: RenderAction) -> None:
        """把渲染动作写到输出流（T13 替换为 Rich 呈现）。"""
        if action.kind == "write_line":
            print(action.text, file=self.stream)
        elif action.kind == "stream_chunk":
            print(action.text, end="", file=self.stream)
            self.stream.flush()
        elif action.kind == "set_busy":
            self.state.busy = True
        elif action.kind == "clear_busy":
            self.state.busy = False
            print("", file=self.stream)
        elif action.kind == "footer":
            pass  # 骨架：footer 状态由 footer_line 单独查询
        elif action.kind == "error":
            print(f"错误: {action.text}", file=self.stream)

    async def _consume(self, events: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
        """事件消费 + 渲染执行（headless 可测）。"""
        consumed: list[AgentEvent] = []
        async for event in events:
            for action in self.pipeline.handle(event):
                self.apply_action(action)
            consumed.append(event)
        return consumed

    # ------------------------------------------------------------------
    # agent 轮次
    # ------------------------------------------------------------------

    async def run_agent_turn(self, prompt: str) -> list[AgentEvent]:
        """执行一轮 agent：注入用户消息、消费事件、渲染。"""
        self._interrupt_event.clear()
        user_message = UserMessage(content=prompt)
        self.agent_context.messages.append(user_message)

        config = self.config_factory()
        events = agent_loop(
            [user_message],
            self.agent_context,
            config,
            signal=self._interrupt_event,
        )
        return await self._consume(events)

    async def interrupt(self) -> None:
        """中断当前 agent 请求。"""
        self._interrupt_event.set()

    def request_exit(self) -> None:
        """请求退出。"""
        self._exit_event.set()

    @property
    def exit_requested(self) -> bool:
        return self._exit_event.is_set()

    # ------------------------------------------------------------------
    # 输入解析与命令分发
    # ------------------------------------------------------------------

    async def handle_input(self, raw: str) -> CommandResult | None:
        """处理一次输入；命令返回结果，普通输入返回 None。"""
        if len(raw) > INPUT_MAX_CHARS:
            print(
                f"输入过长（{len(raw)} 字符 > {INPUT_MAX_CHARS}），已忽略",
                file=self.stream,
            )
            return CommandResult(output="", error=True)

        parsed = parse_command_input(raw)
        if parsed is None:
            return None
        name, args = parsed
        context = (
            self.command_context_factory()
            if self.command_context_factory is not None
            else CommandContext(cwd=self.cwd, current_model_id=self.state.current_model_id)
        )
        result = await self.command_registry.execute(name, args, context)
        if result.output:
            print(result.output, file=self.stream)
        if result.request_exit:
            self.request_exit()
        if result.new_model_id:
            self.state.current_model_id = result.new_model_id
        return result

    # ------------------------------------------------------------------
    # REPL 主循环（prompt_toolkit 集成，T15 e2e 驱动）
    # ------------------------------------------------------------------

    def _build_key_bindings(self) -> KeyBindings:
        """Esc 中断运行；Ctrl-D 退出。"""
        bindings = KeyBindings()

        @bindings.add("escape", eager=True)
        async def _interrupt_key(event) -> None:
            if self.state.busy:
                await self.interrupt()
                print("(已请求中断…)", file=self.stream)

        @bindings.add("c-d")
        async def _exit_key(event) -> None:
            self.request_exit()
            event.app.exit(exception=EOFError)

        return bindings

    async def run_async(self) -> int:
        """REPL 主循环。"""
        session: PromptSession = PromptSession(
            history=self._history,
            key_bindings=self._build_key_bindings(),
            multiline=False,
        )
        print(
            "mimcode 交互模式（/help 查看命令，Esc 中断，Ctrl-D 退出）",
            file=self.stream,
        )
        while not self._exit_event.is_set():
            try:
                raw = await session.prompt_async(FormattedText([("", "> ")]))
            except (EOFError, KeyboardInterrupt):
                if self.state.busy:
                    await self.interrupt()
                    continue
                break

            stripped = raw.strip()
            if not stripped:
                continue

            command_result = await self.handle_input(stripped)
            if command_result is not None:
                continue

            await self.run_agent_turn(stripped)
        return 0
