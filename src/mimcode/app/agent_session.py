"""AgentSession：print/interactive 共用的会话装配层（对齐 pi agent-session.ts 角色）。

装配内容：
- 端点注册表（T4）与模型解析（--model 或默认端点）
- 工具注册表（T5 七件套 + T11 插件工具）
- 系统提示（基础 + T9 技能注入）
- 会话（T7）：新建 / --continue 恢复 / --fork 分叉
- 命令注册表（T10 + T11 插件命令）
- agent loop 配置工厂（模型切换后重建）

事件订阅：会话层把 agent_end 的新增消息持久化到 JSONL（对齐 pi
AgentSession 的事件→存储接线）。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from mimcode.agent.loop import AgentContext, AgentEvent, AgentLoopConfig
from mimcode.agent.tools import ToolRegistry
from mimcode.app.commands import CommandContext
from mimcode.app.extensions import (
    LoadExtensionsResult,
    apply_extensions,
    extension_names,
    load_extensions,
)
from mimcode.app.session import SessionManager
from mimcode.app.skills import LoadSkillsResult
from mimcode.app.skills import load_skills as load_skills_impl
from mimcode.config import EndpointEntry
from mimcode.core.system_prompt import build_system_prompt
from mimcode.provider.auth import resolve_api_key
from mimcode.provider.catalog import default_endpoint_name
from mimcode.provider.registry import Registry, build_registry
from mimcode.types import AgentMessage, ModelInfo, ThinkingLevel


class AgentSessionError(Exception):
    """会话装配错误（模型未解析 / 配置错误等）。"""


class AgentSession:
    """装配完成的运行会话（print 与 interactive 共用）。"""

    def __init__(
        self,
        *,
        cwd: str,
        home: Path | None = None,
        model_spec: str | None = None,
        thinking_level: str | None = None,
        continue_session: bool = False,
        fork: bool = False,
        session_id: str | None = None,
        user_config: object | None = None,
    ) -> None:
        self.cwd = cwd
        self.home = home

        # --- 配置与注册表 ---
        from mimcode.config import Config, load_config

        if isinstance(user_config, Config):
            config = user_config
        else:
            config = load_config(cwd=Path(cwd), home=home)
        self.registry: Registry = build_registry(config)

        # --- 模型解析 ---
        resolved_model = self._resolve_model(model_spec)
        self.model: ModelInfo = resolved_model
        self.model_spec = model_spec
        self.thinking_level: ThinkingLevel | None = self._normalize_thinking(thinking_level)

        # --- 技能与插件 ---
        self.skills: LoadSkillsResult = load_skills_impl(cwd, home=home)
        self.extensions: LoadExtensionsResult = load_extensions(cwd, home=home)
        self.extension_names: list[str] = extension_names(self.extensions)

        # --- 工具注册表（七件套 + 插件） ---
        self.tools = ToolRegistry(cwd=cwd)
        self.command_registry = _fresh_command_registry()
        self.applied_extensions = apply_extensions(
            self.extensions,
            tool_registry=self.tools,
            command_registry=self.command_registry,
        )

        # --- 系统提示 ---
        self.system_prompt = build_system_prompt(self.skills)

        # --- 会话 ---
        self.session = self._open_session(continue_session, fork, session_id)

    # ------------------------------------------------------------------
    # 装配细节
    # ------------------------------------------------------------------

    def _resolve_model(self, model_spec: str | None) -> ModelInfo:
        """解析目标模型（--model 或默认端点首模型）。"""
        if model_spec:
            resolution = self.registry.resolve_model(model_spec)
            if resolution is None:
                raise AgentSessionError(f"未找到模型: {model_spec}（--list-models 查看可用模型）")
            return resolution.model

        default_name = default_endpoint_name(self.registry.config)
        if default_name is None:
            raise AgentSessionError("无可用端点（检查 ~/.mimcode/config.toml）")
        provider = self.registry.get_provider(default_name)
        if provider is None or not provider.get_models():
            raise AgentSessionError(f"端点 '{default_name}' 无可用模型")
        return provider.get_models()[0]

    @staticmethod
    def _normalize_thinking(level: str | None) -> ThinkingLevel | None:
        """CLI 级别 → 流级别（off 表示 None，对齐 pi）。"""
        if level is None or level == "off":
            return None
        return level  # type: ignore[return-value]

    def _open_session(
        self, continue_session: bool, fork: bool, session_id: str | None
    ) -> SessionManager:
        """打开/恢复/分叉会话。"""
        if fork:
            source = (
                SessionManager.open(self._find_session_file(session_id))
                if session_id
                else SessionManager.latest(self.cwd, home=self.home)
            )
            if source is None:
                raise AgentSessionError("没有可分叉的会话（先创建一个）")
            return SessionManager.fork_from(source.session_file, self.cwd, home=self.home)
        if continue_session:
            session = SessionManager.latest(self.cwd, home=self.home)
            if session is None:
                return SessionManager.create(self.cwd, home=self.home)
            return session
        return SessionManager.create(self.cwd, home=self.home)

    def _find_session_file(self, session_id: str) -> Path:
        """按会话 id 找文件（fork --session 语义）。"""
        from mimcode.app.session_store import list_sessions, sessions_root

        root = sessions_root(self.home) if self.home is not None else sessions_root()
        directory = root / _encode(self.cwd)
        for info in list_sessions(directory):
            if info.id == session_id:
                return info.path
        raise AgentSessionError(f"未找到会话: {session_id}")

    # ------------------------------------------------------------------
    # 运行入口
    # ------------------------------------------------------------------

    def build_agent_context(self) -> AgentContext:
        """构建 agent 上下文（系统提示 + 会话历史 + 工具）。"""
        session_context = self.session.build_context()
        return AgentContext(
            system_prompt=self.system_prompt,
            messages=list(session_context.messages),
            tools=self.tools.all(),
        )

    def make_loop_config(self, stream_fn: Callable | None = None) -> AgentLoopConfig:
        """构建 loop 配置（API key 经解析链）。"""

        endpoint = self._endpoint_entry()
        api_key = (
            resolve_api_key(endpoint, environ=_snapshot_environ()) if endpoint is not None else None
        )
        return AgentLoopConfig(
            model=self.model,
            stream_fn=stream_fn or self._default_stream_fn(),
            api_key=api_key,
            thinking_level=self.thinking_level,
        )

    def _endpoint_entry(self) -> EndpointEntry | None:
        """当前模型对应的端点条目（key 解析用）。"""
        return self.registry.config.endpoints.get(self.model.provider)

    def _default_stream_fn(self):
        """模型所属端点 provider 的流函数。"""
        provider = self.registry.get_provider(self.model.provider)
        if provider is None:
            raise AgentSessionError(f"端点 '{self.model.provider}' 不存在")
        return provider.stream

    # ------------------------------------------------------------------
    # 事件持久化（对齐 pi：agent_end 新增消息落盘）
    # ------------------------------------------------------------------

    async def persist_events(self, events: list[AgentEvent]) -> int:
        """把 agent_end 携带的新增消息写入会话文件。

        Returns:
            持久化的消息数。
        """
        stored = 0
        for event in events:
            if event.type == "agent_end":
                for message in event.messages:
                    self.session.append_message(message)
                    stored += 1
        return stored

    # ------------------------------------------------------------------
    # 命令上下文（T10 命令层接线）
    # ------------------------------------------------------------------

    def command_context(self) -> CommandContext:
        """命令执行上下文。"""
        return CommandContext(
            cwd=self.cwd,
            session=self.session,
            registry=self.registry,
            home=self.home,
            skill_names=[skill.name for skill in self.skills.skills],
            extension_names=self.extension_names,
            current_model_id=self.model.id,
            current_thinking_level=self.thinking_level,
        )

    # ------------------------------------------------------------------
    # 模型切换（/model 后主循环接线）
    # ------------------------------------------------------------------

    def switch_model(self, model_spec: str) -> ModelInfo:
        """切换模型（命令结果接线）。"""
        resolution = self.registry.resolve_model(model_spec)
        if resolution is None:
            raise AgentSessionError(f"未找到模型: {model_spec}")
        self.model = resolution.model
        self.session.append_model_change(resolution.model.provider, resolution.model.id)
        return resolution.model


def _fresh_command_registry():
    """内置命令注册表副本（插件命令追加，不污染单例）。"""
    from mimcode.app.commands import CommandRegistry, builtin_commands

    registry = CommandRegistry()
    for command in builtin_commands():
        registry.register(command)
    return registry


def _encode(cwd: str) -> str:
    from mimcode.app.session_store import encode_cwd_to_dir_name

    return encode_cwd_to_dir_name(cwd)


def _snapshot_environ() -> dict[str, str]:
    """环境变量快照（key 解析链输入）。"""
    import os

    return dict(os.environ)


async def run_prompt(
    session: AgentSession,
    prompt: str,
    *,
    on_event: Callable[[AgentEvent], None] | None = None,
) -> list[AgentMessage]:
    """执行一次完整 agent 运行并持久化（print 模式核心）。

    Args:
        session: 装配好的会话。
        prompt: 用户提示词。
        on_event: 事件回调（渲染接线）。

    Returns:
        本次运行新增的消息（含 prompt 与回复）。
    """
    from mimcode.agent.loop import agent_loop
    from mimcode.types import UserMessage

    context = session.build_agent_context()
    config = session.make_loop_config()

    events: list[AgentEvent] = []
    async for event in agent_loop([UserMessage(content=prompt)], context, config):
        events.append(event)
        if on_event is not None:
            on_event(event)
    await session.persist_events(events)
    if events and events[-1].type == "agent_end":
        return events[-1].messages
    return []
