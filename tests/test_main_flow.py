"""T14 主流程接线测试。

checklist 对应项（E2E-1/E2E-2 在 T15 走进程级；此处验证模块级全链路）：
- AgentSession 装配：模型解析/系统提示含技能/工具注册表/会话 flag
- print 模式：流式输出/工具状态/退出码/落盘
- -c 续接：第二次运行的上下文含首会话全部消息
- 首次启动引导与错误路径
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mimcode.app.agent_session import AgentSession, AgentSessionError, run_prompt
from mimcode.config import Config, EndpointEntry
from mimcode.modes.print_mode import run_print_mode
from mimcode.provider.faux import FauxFixture, FauxOpenAIProvider
from mimcode.types import UserMessage

# ---------------------------------------------------------------------------
# faux 端点注入：让 AgentSession 全链路跑在回放上
# ---------------------------------------------------------------------------


def make_faux_config(scripts: list[FauxFixture]) -> Config:
    """构造把 faux 端点声明为唯一端点的配置。"""
    return Config(
        endpoints={
            "faux": EndpointEntry(
                protocol="openai",
                api_key_env="FAUX_KEY",
                default=True,
                models={"faux-model": {}},
            )
        }
    )


def openai_text_chunks(text: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "chatcmpl-text",
            "model": "faux-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-text",
            "model": "faux-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {"id": "chatcmpl-text", "model": "faux-model", "choices": []},
    ]


def openai_tool_call_chunks(call_id: str, name: str, args: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"chatcmpl-{call_id}",
            "model": "faux-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": f"chatcmpl-{call_id}",
            "model": "faux-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {"id": f"chatcmpl-{call_id}", "model": "faux-model", "choices": []},
    ]


def make_scripted(scripts: list[FauxFixture]) -> FauxOpenAIProvider:
    """多轮脚本 faux（非线程安全：测试内顺序使用）。"""
    provider = FauxOpenAIProvider(scripts[0])
    provider.__dict__["_scripts"] = scripts
    provider.__dict__["_index"] = 0

    original = provider._raw_chunks

    async def scripted(payload, options):
        provider.requests.append(payload)
        scripts = provider.__dict__["_scripts"]
        index = min(provider.__dict__["_index"], len(scripts) - 1)
        provider.__dict__["_index"] = index + 1
        for chunk in scripts[index].chunks:
            yield chunk

    provider._raw_chunks = scripted  # type: ignore[method-assign]
    del original
    return provider


def patch_session_stream(session: AgentSession, provider: FauxOpenAIProvider) -> None:
    """把会话的默认流函数替换为脚本 faux。"""
    session._default_stream_fn = lambda: provider.stream  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------


async def test_session_assembles_and_resolves_model(tmp_path: Path) -> None:
    """装配：默认端点解析 + 工具注册表 + 系统提示。"""
    session = AgentSession(
        cwd=str(tmp_path),
        home=tmp_path,
        user_config=make_faux_config([]),
    )
    assert session.model.id == "faux-model"
    assert session.model.provider == "faux"

    # 工具七件套
    tool_names = {tool.name for tool in session.tools.all()}
    assert {"bash", "read", "write", "edit", "ls", "find", "grep"} <= tool_names

    # 系统提示组装
    assert "mimcode" in session.system_prompt
    assert "Tool usage" in session.system_prompt

    # 会话新建
    assert session.session.session_file.exists()
    assert session.session.session_file.suffix == ".jsonl"


async def test_session_model_spec_and_unknown(tmp_path: Path) -> None:
    """--model 解析与未知模型报错。"""
    config = make_faux_config([])
    session = AgentSession(
        cwd=str(tmp_path), home=tmp_path, model_spec="faux/faux-model", user_config=config
    )
    assert session.model.id == "faux-model"

    with pytest.raises(AgentSessionError, match="未找到模型"):
        AgentSession(cwd=str(tmp_path), home=tmp_path, model_spec="nope", user_config=config)


async def test_session_skills_injected_into_prompt(tmp_path: Path) -> None:
    """技能加载并注入系统提示（checklist：技能名出现在系统提示）。"""
    # 全局技能目录：<home>/.mimcode/skills（session 传 home=tmp_path）
    skills_dir = tmp_path / ".mimcode" / "skills"
    skill_dir = skills_dir / "pdf-tools"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: pdf-tools\ndescription: 处理 PDF\n---\n正文\n", encoding="utf-8"
    )

    session = AgentSession(cwd=str(tmp_path), home=tmp_path, user_config=make_faux_config([]))
    assert "pdf-tools" in session.system_prompt
    assert "<available_skills>" in session.system_prompt


# ---------------------------------------------------------------------------
# print 模式全链路（checklist E2E-1 模块级）
# ---------------------------------------------------------------------------


async def test_print_mode_full_pipeline(tmp_path: Path) -> None:
    """print：流式正文到 stdout、工具状态到 stderr、退出码 0、落盘。"""
    from mimcode.agent.tools.base import AgentTool

    class EchoTool(AgentTool):
        name = "echo"
        description = "echo"

        def parameters_schema(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }

        async def execute(self, tool_call_id, args, signal=None, on_update=None):
            return self.text_result(f"echo:{args.get('text', '')}")

    scripted = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-model",
                chunks=openai_tool_call_chunks("c1", "echo", json.dumps({"text": "hi"})),
            ),
            FauxFixture(
                protocol="openai",
                model="faux-model",
                chunks=openai_text_chunks("最终回复"),
            ),
        ]
    )

    import io

    stdout, stderr = io.StringIO(), io.StringIO()
    session = AgentSession(cwd=str(tmp_path), home=tmp_path, user_config=make_faux_config([]))
    session.tools.register(EchoTool())
    patch_session_stream(session, scripted)

    exit_code = await run_print_mode(session, "做点什么", stdout=stdout, stderr=stderr)

    assert exit_code == 0
    out = stdout.getvalue()
    assert "最终回复" in out
    assert "[工具] echo" in stderr.getvalue()

    # 落盘：完整对话（用户/助手/toolResult/助手）
    context = session.session.build_context()
    roles = [message.role for message in context.messages]
    assert roles == ["user", "assistant", "toolResult", "assistant"]


async def test_print_mode_stream_exit_code_error(tmp_path: Path) -> None:
    """流 error 终态 → 退出码 2 + stderr 错误。"""
    error_chunks = [
        {
            "id": "chatcmpl-err",
            "model": "faux-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}],
        },
    ]
    scripted = make_scripted(
        [FauxFixture(protocol="openai", model="faux-model", chunks=error_chunks)]
    )

    import io

    session = AgentSession(cwd=str(tmp_path), home=tmp_path, user_config=make_faux_config([]))
    patch_session_stream(session, scripted)

    exit_code = await run_print_mode(session, "hi", stdout=io.StringIO(), stderr=io.StringIO())
    # content_filter → error 终态 → 2
    assert exit_code == 2


async def test_print_mode_requires_prompt(tmp_path: Path) -> None:
    """print 模式空提示词：错误提示。"""
    import io

    session = AgentSession(cwd=str(tmp_path), home=tmp_path, user_config=make_faux_config([]))
    stderr = io.StringIO()
    exit_code = await run_print_mode(session, "", stdout=io.StringIO(), stderr=stderr)
    assert exit_code == 2
    assert "提示词" in stderr.getvalue()


# ---------------------------------------------------------------------------
# -c 续接（checklist：第二次运行上下文含首会话消息）
# ---------------------------------------------------------------------------


async def test_continue_session_context(tmp_path: Path) -> None:
    """-c：第一次落盘后第二次 AgentSession(continue) 携带历史。"""
    scripted1 = make_scripted(
        [FauxFixture(protocol="openai", model="faux-model", chunks=openai_text_chunks("第一答"))]
    )
    session1 = AgentSession(cwd=str(tmp_path), home=tmp_path, user_config=make_faux_config([]))
    patch_session_stream(session1, scripted1)
    await run_prompt(session1, "第一问")

    # 续接装配：continue=True
    scripted2 = make_scripted(
        [FauxFixture(protocol="openai", model="faux-model", chunks=openai_text_chunks("第二答"))]
    )
    session2 = AgentSession(
        cwd=str(tmp_path),
        home=tmp_path,
        continue_session=True,
        user_config=make_faux_config([]),
    )
    patch_session_stream(session2, scripted2)

    context = session2.build_agent_context()
    texts = [
        m.content if isinstance(m, UserMessage) else m.content[0].text for m in context.messages
    ]
    assert texts == ["第一问", "第一答"]

    # 第二次运行后全量历史
    await run_prompt(session2, "第二问")
    final = session2.build_agent_context()
    texts2 = [
        m.content if isinstance(m, UserMessage) else m.content[0].text for m in final.messages
    ]
    assert texts2 == ["第一问", "第一答", "第二问", "第二答"]


async def test_fork_session_from_latest(tmp_path: Path) -> None:
    """--fork：分叉会话含源历史、源不动。"""
    scripted = make_scripted(
        [FauxFixture(protocol="openai", model="faux-model", chunks=openai_text_chunks("答"))]
    )
    session1 = AgentSession(cwd=str(tmp_path), home=tmp_path, user_config=make_faux_config([]))
    patch_session_stream(session1, scripted)
    await run_prompt(session1, "原始问题")
    source_file = session1.session.session_file
    source_id = session1.session.session_id

    forked = AgentSession(
        cwd=str(tmp_path), home=tmp_path, fork=True, user_config=make_faux_config([])
    )
    assert forked.session.session_id != source_id
    texts = [
        m.content if isinstance(m, UserMessage) else m.content[0].text
        for m in forked.build_agent_context().messages
    ]
    assert texts == ["原始问题", "答"]

    # 分叉后继续运行，源会话不被修改
    forked_script = make_scripted(
        [FauxFixture(protocol="openai", model="faux-model", chunks=openai_text_chunks("新答"))]
    )
    patch_session_stream(forked, forked_script)
    await run_prompt(forked, "新问题")

    from mimcode.app.session import SessionManager

    source = SessionManager.open(source_file)
    assert len(source.build_context().messages) == 2


# ---------------------------------------------------------------------------
# 命令上下文与模型切换接线
# ---------------------------------------------------------------------------


async def test_command_context_and_model_switch(tmp_path: Path) -> None:
    """命令上下文装配 + /model 切换落会话条目。"""
    config = Config(
        endpoints={
            "faux": EndpointEntry(
                protocol="openai", api_key_env="FAUX_KEY", default=True, models={"faux-model": {}}
            ),
            "alt": EndpointEntry(
                protocol="openai", api_key_env="ALT_KEY", models={"alt-model": {}}
            ),
        }
    )
    session = AgentSession(cwd=str(tmp_path), home=tmp_path, user_config=config)

    ctx = session.command_context()
    assert ctx.current_model_id == "faux-model"
    assert ctx.session is not None
    assert ctx.registry is not None

    # 模型切换：状态 + model_change 条目
    session.switch_model("alt-model")
    assert session.model.id == "alt-model"
    assert session.model.provider == "alt"
    # 会话设置条目记录了变更（build_context 还原）
    assert session.session.build_context().settings.model_id == "alt-model"


# ---------------------------------------------------------------------------
# 事件持久化
# ---------------------------------------------------------------------------


async def test_run_prompt_persists_all_messages(tmp_path: Path) -> None:
    """run_prompt：agent_end 消息全量落盘。"""
    scripted = make_scripted(
        [FauxFixture(protocol="openai", model="faux-model", chunks=openai_text_chunks("ok"))]
    )
    session = AgentSession(cwd=str(tmp_path), home=tmp_path, user_config=make_faux_config([]))
    patch_session_stream(session, scripted)

    messages = await run_prompt(session, "问题")
    assert [m.role for m in messages] == ["user", "assistant"]

    stored = session.session.build_context().messages
    assert [m.role for m in stored] == ["user", "assistant"]
