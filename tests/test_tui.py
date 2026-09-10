"""T12 TUI 骨架测试（渲染管线与事件消费，headless）。

checklist 对应项：
- 渲染管线：含 thinking + text + toolCall 的 faux 流，渲染产出包含
  折叠态思考区、正文、工具执行块
- 流式增量：text_delta 分 3 片到达时渲染更新被调用 ≥3 次且最终拼接完整
- 中断：agent 运行中触发中断后流以 stopReason=aborted 结束且
  busy 状态清除（不残留加载动画）
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

from mimcode.agent.loop import AgentContext, AgentLoopConfig
from mimcode.agent.tools.base import AgentTool
from mimcode.provider.faux import FauxFixture, FauxOpenAIProvider
from mimcode.tui.app import InteractiveApp
from mimcode.tui.renderer import RendererPipeline
from mimcode.tui.state import TuiState
from mimcode.types import (
    AgentEnd,
    AgentStart,
    AssistantMessage,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    StreamTextDelta,
    StreamThinkingDelta,
    TextBlock,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolResult,
    ToolResultMessage,
    TurnEnd,
    TurnStart,
    Usage,
    UserMessage,
)

FIXTURES = Path(__file__).parent / "fixtures"


def make_assistant(**overrides: Any) -> AssistantMessage:
    fields: dict[str, Any] = {
        "content": [],
        "api": "openai",
        "provider": "faux-openai",
        "model": "faux-gpt",
        "stop_reason": "stop",
    }
    fields.update(overrides)
    return AssistantMessage(**fields)


# ---------------------------------------------------------------------------
# 渲染管线（checklist：thinking 折叠 + 正文 + 工具块）
# ---------------------------------------------------------------------------


def test_pipeline_full_stream_rendering() -> None:
    """checklist：thinking + text + toolCall 流的渲染产出。"""
    state = TuiState()
    pipeline = RendererPipeline(state)

    partial = make_assistant(stop_reason="pending")

    actions: list[Any] = []
    for event in [
        AgentStart(),
        TurnStart(),
        MessageStart(message=UserMessage(content="帮我查天气")),
        # thinking 流
        MessageUpdate(
            message=partial,
            assistant_event=StreamThinkingDelta(content_index=0, delta="先查工具", partial=partial),
        ),
        # text 流
        MessageUpdate(
            message=partial,
            assistant_event=StreamTextDelta(content_index=1, delta="调用工具", partial=partial),
        ),
        # 工具执行
        ToolExecutionStart(tool_call_id="c1", tool_name="weather", args={"city": "北京"}),
        ToolExecutionEnd(
            tool_call_id="c1",
            tool_name="weather",
            result=ToolResult(content=[TextBlock(text="北京 晴")]),
            is_error=False,
        ),
        # 定稿
        MessageEnd(message=make_assistant(content=[TextBlock(text="调用工具查询北京天气：晴")])),
        TurnEnd(
            message=make_assistant(content=[TextBlock(text="调用工具查询北京天气：晴")]),
            tool_results=[
                ToolResultMessage(
                    tool_call_id="c1",
                    tool_name="weather",
                    content=[TextBlock(text="北京 晴")],
                )
            ],
        ),
        AgentEnd(messages=[]),
    ]:
        actions.extend(pipeline.handle(event))

    kinds = [action.kind for action in actions]
    lines = [action.text for action in actions if action.kind == "write_line"]
    all_text = "\n".join(lines)

    # 用户消息行
    assert any(line.startswith("你:") and "帮我查天气" in line for line in lines)
    # 折叠思考区头行 + 定稿摘要
    assert any("思考" in line for line in lines)
    # 工具执行块（start 与 end）
    assert any("⚙ weather" in line and "city=北京" in line for line in lines)
    assert any("✓ weather" in line and "北京 晴" in line for line in lines)
    # 助手正文
    assert any(line.startswith("mim:") and "晴" in line for line in lines)
    # 流式块存在
    assert "stream_chunk" in kinds
    # busy 状态翻转
    assert "set_busy" in kinds and "clear_busy" in kinds
    assert state.busy is False
    del all_text


def test_pipeline_error_terminal_clears_busy() -> None:
    """checklist：aborted/error 终态清除 busy（不残留加载状态）。"""
    state = TuiState()
    pipeline = RendererPipeline(state)
    pipeline.handle(AgentStart())
    assert state.busy is True

    aborted = make_assistant(stop_reason="aborted", error_message="用户中断")
    pipeline.handle(AgentEnd(messages=[aborted]))
    assert state.busy is False


def test_pipeline_accumulates_usage_and_footer() -> None:
    """用量累计与 footer（模型/会话/轮数/token）。"""
    state = TuiState(current_model_id="faux-gpt", session_id="abcdef123456")
    pipeline = RendererPipeline(state)

    assistant = make_assistant(usage=Usage(input=100, output=50))
    pipeline.handle(AgentStart())
    pipeline.handle(AgentEnd(messages=[UserMessage(content="q"), assistant]))

    assert state.usage_total.input == 100
    assert state.usage_total.output == 50
    footer = state.footer_line()
    assert "faux-gpt" in footer
    assert "abcdef12" in footer
    assert "150 tokens" in footer


def test_pipeline_tool_error_marked() -> None:
    """工具错误结果带 ✗ 标记。"""
    state = TuiState()
    pipeline = RendererPipeline(state)
    actions = pipeline.handle(
        ToolExecutionEnd(
            tool_call_id="c",
            tool_name="bash",
            result=ToolResult(content=[TextBlock(text="命令失败")]),
            is_error=True,
        )
    )
    assert actions[0].text.startswith("  ✗")
    assert "命令失败" in actions[0].text


def test_pipeline_long_tool_result_truncated() -> None:
    """工具结果预览截断。"""
    pipeline = RendererPipeline(TuiState())
    long_text = "x" * 500
    actions = pipeline.handle(
        ToolExecutionEnd(
            tool_call_id="c",
            tool_name="read",
            result=ToolResult(content=[TextBlock(text=long_text)]),
            is_error=False,
        )
    )
    assert len(actions[0].text) < 200
    assert "…" in actions[0].text


def test_pipeline_empty_assistant_renders_placeholder() -> None:
    """空助手回复渲染占位行。"""
    pipeline = RendererPipeline(TuiState())
    actions = pipeline.handle(MessageEnd(message=make_assistant(content=[])))
    assert any(action.text == "mim: (空回复)" for action in actions)


def test_pipeline_footer_on_turn_end() -> None:
    """turn_end 产出 footer 动作。"""
    state = TuiState(current_model_id="m1")
    pipeline = RendererPipeline(state)
    actions = pipeline.handle(
        TurnEnd(
            message=make_assistant(content=[]),
            tool_results=[],
        )
    )
    assert actions and actions[0].kind == "footer"
    assert "m1" in actions[0].text
    assert state.turns == 1


# ---------------------------------------------------------------------------
# 流式增量（checklist：3 片 delta → 3 次更新 + 完整拼接）
# ---------------------------------------------------------------------------


def test_stream_delta_count_and_assembly() -> None:
    """checklist：3 片 text_delta → ≥3 次 stream_chunk + 拼接完整。"""
    pipeline = RendererPipeline(TuiState())
    partial = make_assistant(stop_reason="pending")

    for chunk in ("你好", "，", "世界"):
        pipeline.handle(
            MessageUpdate(
                message=partial,
                assistant_event=StreamTextDelta(content_index=0, delta=chunk, partial=partial),
            )
        )

    assert pipeline.update_calls >= 3
    chunks = list(pipeline._stream_buffer)
    assert "".join(pipeline._stream_buffer) == "你好，世界"
    del chunks


def test_stream_delta_via_message_end_assembly() -> None:
    """流式缓冲与定稿文本一致。"""
    pipeline = RendererPipeline(TuiState())
    partial = make_assistant(stop_reason="pending")
    for chunk in ("A", "B", "C"):
        pipeline.handle(
            MessageUpdate(
                message=partial,
                assistant_event=StreamTextDelta(content_index=0, delta=chunk, partial=partial),
            )
        )
    actions = pipeline.handle(MessageEnd(message=make_assistant(content=[TextBlock(text="ABC")])))
    final_lines = [action.text for action in actions if action.kind == "write_line"]
    assert any(line == "mim: ABC" for line in final_lines)


# ---------------------------------------------------------------------------
# InteractiveApp 事件消费（headless 全链路）
# ---------------------------------------------------------------------------


class EchoTool(AgentTool):
    name = "echo"
    description = "echo"

    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, tool_call_id, args, signal=None, on_update=None) -> ToolResult:
        return self.text_result(f"echo:{args.get('text', '')}")


def openai_tool_call_chunks(call_id: str, name: str, args: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"chatcmpl-{call_id}",
            "model": "faux-gpt",
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
            "model": "faux-gpt",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {"id": f"chatcmpl-{call_id}", "model": "faux-gpt", "choices": []},
    ]


def openai_text_chunks(text: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "chatcmpl-text",
            "model": "faux-gpt",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-text",
            "model": "faux-gpt",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {"id": "chatcmpl-text", "model": "faux-gpt", "choices": []},
    ]


class ScriptedProvider(FauxOpenAIProvider):
    """多轮脚本 faux。"""

    def __init__(self, scripts: list[FauxFixture]) -> None:
        super().__init__(scripts[0])
        self.scripts = scripts
        self._index = 0

    async def _raw_chunks(self, payload, options):
        self.requests.append(payload)
        index = min(self._index, len(self.scripts) - 1)
        self._index += 1
        for chunk in self.scripts[index].chunks:
            yield chunk


def make_app(provider: ScriptedProvider, tools: list[AgentTool] | None = None) -> InteractiveApp:
    """组装 headless 可测的 app。"""
    output = io.StringIO()
    context = AgentContext(tools=tools or [])
    config = AgentLoopConfig(
        model=provider.get_models()[0],
        stream_fn=provider.stream,
    )
    app = InteractiveApp(
        cwd=".",
        agent_context=context,
        config_factory=lambda: config,
        stream=output,
    )
    return app


async def test_app_renders_full_turn() -> None:
    """app 全链路：工具轮 + 文本轮渲染到输出流。"""
    provider = ScriptedProvider(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "echo", json.dumps({"text": "你好"})),
            ),
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_text_chunks("完成查询"),
            ),
        ]
    )
    app = make_app(provider, tools=[EchoTool()])

    events = await app.run_agent_turn("查询")
    output = app.stream.getvalue() if isinstance(app.stream, io.StringIO) else ""

    # 事件完整消费（agent_end 收尾）
    assert events[-1].type == "agent_end"
    # 渲染落盘：用户行 / 工具块 / 助手正文
    assert "你: 查询" in output
    assert "⚙ echo" in output
    assert "✓ echo" in output
    assert "mim: 完成查询" in output
    # busy 清除（无残留加载）
    assert app.state.busy is False


async def test_app_interruption_mid_run() -> None:
    """checklist：运行中中断 → aborted 终态 + busy 清除。"""

    class SlowProvider(ScriptedProvider):
        """首 chunk 后阻塞的流（模拟长响应）。"""

        def __init__(self) -> None:
            super().__init__([FauxFixture(protocol="openai", model="faux-gpt", chunks=[])])
            self._release = asyncio.Event()

        async def _raw_chunks(self, payload, options):
            self.requests.append(payload)
            yield {
                "id": "chatcmpl-slow",
                "model": "faux-gpt",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "开头"},
                        "finish_reason": None,
                    }
                ],
            }
            await self._release.wait()
            yield {
                "id": "chatcmpl-slow",
                "model": "faux-gpt",
                "choices": [{"index": 0, "delta": {"content": "后续"}, "finish_reason": "stop"}],
            }

    provider = SlowProvider()
    app = make_app(provider)
    task = asyncio.create_task(app.run_agent_turn("长任务"))

    await asyncio.sleep(0.05)
    assert app.state.busy is True
    # 模拟 Esc：请求中断并放行阻塞的流（下一 chunk 间隙触发中止）
    await app.interrupt()
    provider._release.set()
    events = await asyncio.wait_for(task, timeout=2)

    # 流以 aborted 终态结束
    agent_end = events[-1]
    assert agent_end.type == "agent_end"
    assistants = [m for m in agent_end.messages if m.role == "assistant"]
    assert assistants and assistants[-1].stop_reason == "aborted"
    # busy 清除（不残留）
    assert app.state.busy is False


async def test_app_command_dispatch_and_exit() -> None:
    """命令输入分发：/exit 请求退出。"""
    provider = ScriptedProvider(
        [FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("x"))]
    )
    app = make_app(provider)

    result = await app.handle_input("/exit")
    assert result is not None
    assert result.request_exit is True
    assert app.exit_requested is True

    unknown = await app.handle_input("/nope")
    assert unknown is not None and unknown.error is True

    plain = await app.handle_input("普通消息")
    assert plain is None


async def test_app_oversized_input_rejected() -> None:
    """超长输入被拒（不进入 agent）。"""
    provider = ScriptedProvider(
        [FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("x"))]
    )
    app = make_app(provider)
    result = await app.handle_input("x" * 200_000)
    assert result is not None and result.error is True
    assert provider.requests == []  # 未发起请求
