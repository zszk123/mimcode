"""T10 slash 命令测试。

checklist 对应项：
- /model /thinking /resume /fork /compact /skills /extensions /exit /help
  九个命令注册齐全（逐项断言）
- /model <id> 后下一次流式请求使用的模型 id 等于所设值（以
  CommandResult.new_model_id 传导——主循环接线点）
- 未知命令 /nope → 错误提示而非崩溃
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mimcode.app.commands import (
    CommandContext,
    CommandRegistry,
    builtin_command_registry,
    builtin_commands,
    parse_command_input,
)
from mimcode.app.session import SessionManager
from mimcode.config import Config
from mimcode.provider.registry import build_registry
from mimcode.types import TextBlock, UserMessage


def make_context(tmp_path: Path, **overrides: object) -> CommandContext:
    """构造命令上下文（带注册表与会话）。"""
    defaults: dict[str, object] = {
        "cwd": str(tmp_path),
        "registry": build_registry(Config()),
        "home": tmp_path,
        "skill_names": ["pdf-tools"],
        "extension_names": ["demo-ext"],
    }
    defaults.update(overrides)
    return CommandContext(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 注册（checklist：九命令逐项断言）
# ---------------------------------------------------------------------------


def test_builtin_commands_nine_registered() -> None:
    """checklist：九命令注册齐全。"""
    registry = builtin_command_registry()
    expected = {
        "model",
        "thinking",
        "resume",
        "fork",
        "compact",
        "skills",
        "extensions",
        "exit",
        "help",
    }
    names = {command.name for command in registry.all()}
    assert names == expected
    for command in registry.all():
        assert command.handler is not None
        assert command.description


def test_builtin_registry_singleton() -> None:
    """注册表单例（扩展命令 T11 追加注册共享同一实例）。"""
    first = builtin_command_registry()
    second = builtin_command_registry()
    assert first is second


def test_register_duplicate_rejected() -> None:
    """重名注册被拒。"""
    registry = CommandRegistry()
    command = builtin_commands()[0]
    registry.register(command)
    with pytest.raises(ValueError, match="命令已存在"):
        registry.register(command)


def test_parse_command_input() -> None:
    """输入解析：命令/参数分词；非命令返回 None。"""
    assert parse_command_input("/model openai/gpt-5") == ("model", "openai/gpt-5")
    assert parse_command_input("/help") == ("help", "")
    assert parse_command_input("  /thinking   high  ") == ("thinking", "high")
    assert parse_command_input("普通消息") is None
    assert parse_command_input("") is None
    assert parse_command_input("/") is None


# ---------------------------------------------------------------------------
# /model（checklist：切换后模型 id 断言）
# ---------------------------------------------------------------------------


async def test_model_switch_resolves_to_id(tmp_path: Path) -> None:
    """checklist：/model <id> → new_model_id 传导（裸 id 解析）。"""
    context = make_context(tmp_path)
    result = await builtin_command_registry().execute("model", "deepseek-chat", context)
    assert result.error is False
    assert result.new_model_id == "deepseek-chat"
    assert "已切换模型" in result.output
    assert "deepseek" in result.output


async def test_model_switch_scoped(tmp_path: Path) -> None:
    """/model provider/model 形态。"""
    context = make_context(tmp_path)
    result = await builtin_command_registry().execute("model", "zhipu/glm-4.6", context)
    assert result.error is False
    assert result.new_model_id == "glm-4.6"


async def test_model_switch_unknown(tmp_path: Path) -> None:
    """/model 未知模型：错误提示。"""
    context = make_context(tmp_path)
    result = await builtin_command_registry().execute("model", "no-such-model", context)
    assert result.error is True
    assert "no-such-model" in result.output


async def test_model_status_without_args(tmp_path: Path) -> None:
    """/model 无参：当前模型 + 模型清单。"""
    context = make_context(tmp_path, current_model_id="faux-gpt")
    result = await builtin_command_registry().execute("model", "", context)
    assert result.error is False
    assert "faux-gpt" in result.output
    assert "deepseek-chat" in result.output  # 清单含其他端点模型


# ---------------------------------------------------------------------------
# /thinking
# ---------------------------------------------------------------------------


async def test_thinking_set_and_invalid(tmp_path: Path) -> None:
    """/thinking：合法级别设置成功；非法值报错。"""
    context = make_context(tmp_path)
    for level in ("off", "minimal", "low", "medium", "high"):
        result = await builtin_command_registry().execute("thinking", level, context)
        assert result.error is False
        assert result.new_thinking_level == level

    bad = await builtin_command_registry().execute("thinking", "ultra", context)
    assert bad.error is True
    assert "ultra" in bad.output


# ---------------------------------------------------------------------------
# /resume / fork（会话切换请求）
# ---------------------------------------------------------------------------


async def test_resume_latest_session(tmp_path: Path) -> None:
    """/resume 无参：恢复最近会话（switch_session 请求）。"""
    from mimcode.types import AssistantMessage

    first = SessionManager.create(str(tmp_path), home=tmp_path)
    first.append_message(UserMessage(content="旧会话"))

    import os

    second = SessionManager.create(str(tmp_path), home=tmp_path)
    second.append_message(UserMessage(content="新会话问题"))
    second.append_message(
        AssistantMessage(
            content=[TextBlock(text="新会话回答")],
            api="openai",
            provider="p",
            model="m",
            stop_reason="stop",
        )
    )
    os.utime(second.session_file, None)  # 保证最新

    context = make_context(tmp_path)
    result = await builtin_command_registry().execute("resume", "", context)
    assert result.error is False
    assert result.switch_session is not None
    assert result.switch_session.session_id == second.session_id
    assert "新会话问题" in result.output


async def test_resume_by_id(tmp_path: Path) -> None:
    """/resume <id>：按 id 精确恢复。"""
    target = SessionManager.create(str(tmp_path), home=tmp_path)
    target.append_message(UserMessage(content="目标会话"))
    other = SessionManager.create(str(tmp_path), home=tmp_path)
    other.append_message(UserMessage(content="其他会话"))

    context = make_context(tmp_path)
    result = await builtin_command_registry().execute("resume", other.session_id, context)
    assert result.switch_session is not None
    assert result.switch_session.session_id == other.session_id


async def test_resume_unknown_id(tmp_path: Path) -> None:
    """/resume 未知 id：错误提示。"""
    SessionManager.create(str(tmp_path), home=tmp_path)
    context = make_context(tmp_path)
    result = await builtin_command_registry().execute("resume", "nope-id", context)
    assert result.error is True
    assert "nope-id" in result.output


async def test_resume_no_sessions(tmp_path: Path) -> None:
    """/resume 无会话：错误提示。"""
    context = make_context(tmp_path)
    result = await builtin_command_registry().execute("resume", "", context)
    assert result.error is True


async def test_fork_requests_switch(tmp_path: Path) -> None:
    """/fork：分叉新会话并请求切换；原会话不动。"""
    session = SessionManager.create(str(tmp_path), home=tmp_path)
    session.append_message(UserMessage(content="原始问题"))
    original_id = session.session_id

    context = make_context(tmp_path, session=session)
    result = await builtin_command_registry().execute("fork", "", context)
    assert result.error is False
    assert result.switch_session is not None
    assert result.switch_session.session_id != original_id
    # 分叉会话含原会话消息
    texts = [m.content for m in result.switch_session.build_context().messages]
    assert texts == ["原始问题"]


async def test_fork_without_session(tmp_path: Path) -> None:
    """/fork 无活动会话：错误提示。"""
    context = make_context(tmp_path, session=None)
    result = await builtin_command_registry().execute("fork", "", context)
    assert result.error is True


# ---------------------------------------------------------------------------
# /compact /skills /extensions
# ---------------------------------------------------------------------------


async def test_compact_requires_executor(tmp_path: Path) -> None:
    """/compact：无执行器时降级提示（T12 接线后可用）。"""
    session = SessionManager.create(str(tmp_path), home=tmp_path)
    context = make_context(tmp_path, session=session)
    result = await builtin_command_registry().execute("compact", "", context)
    assert result.error is True
    assert "压缩需要模型连接" in result.output


async def test_compact_without_session(tmp_path: Path) -> None:
    """/compact 无会话：错误提示。"""
    context = make_context(tmp_path, session=None)
    result = await builtin_command_registry().execute("compact", "", context)
    assert result.error is True


async def test_skills_listing(tmp_path: Path) -> None:
    """/skills：列出技能名。"""
    context = make_context(tmp_path)
    result = await builtin_command_registry().execute("skills", "", context)
    assert result.error is False
    assert "pdf-tools" in result.output

    empty = make_context(tmp_path, skill_names=[])
    empty_result = await builtin_command_registry().execute("skills", "", empty)
    assert "没有已加载" in empty_result.output


async def test_extensions_listing(tmp_path: Path) -> None:
    """/extensions：列出插件名。"""
    context = make_context(tmp_path)
    result = await builtin_command_registry().execute("extensions", "", context)
    assert result.error is False
    assert "demo-ext" in result.output


# ---------------------------------------------------------------------------
# /exit /help /未知命令（checklist）
# ---------------------------------------------------------------------------


async def test_exit_requests_exit(tmp_path: Path) -> None:
    """/exit：退出请求标记。"""
    context = make_context(tmp_path)
    result = await builtin_command_registry().execute("exit", "", context)
    assert result.error is False
    assert result.request_exit is True


async def test_help_lists_all_commands(tmp_path: Path) -> None:
    """/help：九命令全部出现。"""
    context = make_context(tmp_path)
    result = await builtin_command_registry().execute("help", "", context)
    assert result.error is False
    for name in (
        "model",
        "thinking",
        "resume",
        "fork",
        "compact",
        "skills",
        "extensions",
        "exit",
        "help",
    ):
        assert f"/{name}" in result.output


async def test_unknown_command_error_not_crash(tmp_path: Path) -> None:
    """checklist：/nope → 错误提示而非崩溃。"""
    context = make_context(tmp_path)
    result = await builtin_command_registry().execute("nope", "", context)
    assert result.error is True
    assert "未知命令" in result.output
    assert "/model" in result.output  # 可用命令清单提示
