"""T6 agent loop 测试。

checklist 对应项：
- 双循环事件序列：agent_start → message(用户) → 助手流 → tool_execution_* →
  steering 消息 message_start/end → 第二轮助手流 → agent_end（按序断言）
- 截断保护：stopReason=length 且 2 个工具调用时均不执行、各产出 isError
  结果、错误文本含「未执行/截断」语义
- 并行批：2 个可并行工具的 tool_execution_end 先到先发，
  工具结果消息按助手消息原始顺序发出
- before 钩子 block：reason 原样出现在结果文本中
- prepare_next_turn 换模型：第二轮流式调用收到的模型 id 等于新值

测试驱动：ScriptedProvider（faux 的多轮脚本扩展，每轮返回不同 fixture，
复用真实翻译与守卫路径）。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from mimcode.agent.loop import (
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    BeforeToolCallResult,
    TurnUpdate,
    agent_loop,
    agent_loop_continue,
)
from mimcode.agent.tools.base import AgentTool, ToolResult
from mimcode.provider.faux import FauxFixture, FauxOpenAIProvider
from mimcode.types import AssistantMessage, ModelInfo, TextBlock, ToolResultMessage, UserMessage

FIXTURES = Path(__file__).parent / "fixtures"


def openai_tool_call_chunks(call_id: str, name: str, args: str) -> list[dict[str, Any]]:
    """构造单个工具调用的 chunk 序列（参数分两片）。"""
    half = len(args) // 2
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
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args[:half]}}]},
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
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args[half:]}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {
            "id": f"chatcmpl-{call_id}",
            "model": "faux-gpt",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        },
    ]


def openai_two_tool_calls_chunks() -> list[dict[str, Any]]:
    """一条助手消息携带两个工具调用（并行批测试用）。"""
    return [
        {
            "id": "chatcmpl-two",
            "model": "faux-gpt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "fast", "arguments": ""},
                            },
                            {
                                "index": 1,
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "slow", "arguments": ""},
                            },
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-two",
            "model": "faux-gpt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"text": "f"}'}},
                            {"index": 1, "function": {"arguments": '{"text": "s"}'}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {"id": "chatcmpl-two", "model": "faux-gpt", "choices": []},
    ]


def openai_text_chunks(text: str) -> list[dict[str, Any]]:
    """纯文本回复 chunk。"""
    return [
        {
            "id": "chatcmpl-text",
            "model": "faux-gpt",
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
            "model": "faux-gpt",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {"id": "chatcmpl-text", "model": "faux-gpt", "choices": []},
    ]


def openai_error_chunks() -> list[dict[str, Any]]:
    """流内错误（finish_reason=content_filter）chunk。"""
    return [
        {
            "id": "chatcmpl-err",
            "model": "faux-gpt",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}],
        },
    ]


class ScriptedProvider(FauxOpenAIProvider):
    """多轮脚本 faux：第 n 次调用返回 scripts[n]。"""

    def __init__(self, scripts: list[FauxFixture]) -> None:
        super().__init__(scripts[0])
        self.scripts = scripts
        self._call_index = 0
        self.models_seen: list[str] = []
        self.contexts_seen: list[list[Any]] = []

    async def _raw_chunks(
        self, payload: dict[str, Any], options: Any
    ) -> AsyncIterator[dict[str, Any]]:
        self.requests.append(payload)
        self.models_seen.append(payload["model"])
        self.contexts_seen.append(payload["messages"])
        index = min(self._call_index, len(self.scripts) - 1)
        self._call_index += 1
        for chunk in self.scripts[index].chunks:
            yield chunk


class EchoTool(AgentTool):
    """回显工具（并行/串行行为测试）。"""

    description = "echo args back"

    def __init__(self, name: str, delay: float = 0.0) -> None:
        self.name = name
        self.delay = delay
        self.executed: list[dict[str, Any]] = []

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(
        self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None
    ) -> ToolResult:
        await asyncio.sleep(self.delay)
        self.executed.append(dict(args))
        return self.text_result(f"echo:{args.get('text', '')}")


def make_scripted(scripts: list[FauxFixture]) -> ScriptedProvider:
    return ScriptedProvider(scripts)


def make_config(provider: ScriptedProvider, **overrides: Any) -> AgentLoopConfig:
    defaults: dict[str, Any] = {
        "model": provider.get_models()[0],
        "stream_fn": provider.stream,
    }
    defaults.update(overrides)
    return AgentLoopConfig(**defaults)


async def collect_events(
    config: AgentLoopConfig, context: AgentContext, prompt: str = "你好"
) -> list[Any]:
    return [event async for event in agent_loop([UserMessage(content=prompt)], context, config)]


def event_types(events: list[Any]) -> list[str]:
    return [event.type for event in events]


def assistant_tail() -> AssistantMessage:
    """工具结果前置的助手消息样本。"""
    return AssistantMessage(
        content=[TextBlock(text="prev")],
        api="openai",
        provider="faux-openai",
        model="faux-gpt",
        stop_reason="stop",
    )


def tool_result_tail() -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="prev-call",
        tool_name="echo",
        content=[TextBlock(text="prev result")],
    )


# ---------------------------------------------------------------------------
# 基本流：文本 + 工具调用双轮
# ---------------------------------------------------------------------------


async def test_two_round_text_tool_flow() -> None:
    """双轮：工具调用 → 工具结果 → 第二轮文本；事件序列完整。"""
    provider = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "echo", json.dumps({"text": "你好"})),
            ),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("完成")),
        ]
    )
    echo = EchoTool("echo")
    context = AgentContext(tools=[echo])
    config = make_config(provider)

    events = await collect_events(config, context)

    types = event_types(events)
    assert types[0] == "agent_start"
    assert types[1] == "turn_start"
    assert "message_start" in types and "message_end" in types
    assert "message_update" in types
    assert "tool_execution_start" in types
    assert "tool_execution_end" in types
    assert types.count("turn_start") == 2
    assert types[-1] == "agent_end"

    assert echo.executed == [{"text": "你好"}]

    # 第二轮流式调用收到了第一轮的工具结果消息
    second_round_messages = provider.contexts_seen[1]
    roles = [message["role"] for message in second_round_messages]
    assert roles == ["user", "assistant", "tool"]

    # agent_end 携带本次新增消息
    agent_end = events[-1]
    assert [message.role for message in agent_end.messages] == [
        "user",
        "assistant",
        "toolResult",
        "assistant",
    ]


async def test_context_accumulates_messages() -> None:
    """上下文累计：新增消息经 agent_end 返回，后续轮次携带完整历史。"""
    provider = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "echo", '{"text": "x"}'),
            ),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("ok")),
        ]
    )
    context = AgentContext(tools=[EchoTool("echo")])
    config = make_config(provider)
    events = await collect_events(config, context)

    # pi 语义：loop 在上下文副本上工作；新增消息经 agent_end 返回，
    # 会话层（T7）以 agent_end.messages 持久化
    assert [message.role for message in events[-1].messages] == [
        "user",
        "assistant",
        "toolResult",
        "assistant",
    ]
    # 第二轮流式请求携带完整历史
    roles = [message["role"] for message in provider.contexts_seen[1]]
    assert roles == ["user", "assistant", "tool"]


# ---------------------------------------------------------------------------
# steering / follow-up（checklist：事件序列按序断言）
# ---------------------------------------------------------------------------


async def test_steering_message_injection() -> None:
    """steering：第一轮工具后注入用户消息，再进入第二轮。"""
    provider = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "echo", '{"text": "a"}'),
            ),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("done")),
        ]
    )
    echo = EchoTool("echo")
    context = AgentContext(tools=[echo])
    steering: list[UserMessage] = [UserMessage(content="补充指示")]

    async def get_steering() -> list[UserMessage]:
        if steering:
            return [steering.pop(0)]
        return []

    config = make_config(provider, get_steering_messages=get_steering)

    events = await collect_events(config, context)
    types = event_types(events)

    # pi 语义：起点 drain 的 steering 消息在第一轮流式响应之前注入
    # （turn_start 与 prompt 消息之后、assistant 流之前）
    steering_start_indices = [
        i
        for i, event in enumerate(events)
        if event.type == "message_start"
        and event.message.role == "user"
        and i > 1
        and getattr(event.message, "content", None) == "补充指示"
    ]
    assert steering_start_indices, "steering 用户消息未注入"
    # 注入位置在第一个 assistant 消息事件之前
    first_assistant = next(
        i
        for i, event in enumerate(events)
        if event.type == "message_start" and event.message.role == "assistant"
    )
    assert steering_start_indices[0] < first_assistant

    assert types.count("turn_start") == 2

    # 第二轮流式请求的上下文含 steering 消息
    roles = [message["role"] for message in provider.contexts_seen[1]]
    assert roles == ["user", "user", "assistant", "tool"]


async def test_follow_up_continues_run() -> None:
    """follow-up：agent 本应停止时收到新消息则继续。"""
    provider = make_scripted(
        [
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("第一答")),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("第二答")),
        ]
    )
    context = AgentContext()
    follow_up_queue: list[UserMessage] = [UserMessage(content="追问")]

    async def get_follow_up() -> list[UserMessage]:
        if follow_up_queue:
            return [follow_up_queue.pop(0)]
        return []

    config = make_config(provider, get_follow_up_messages=get_follow_up)
    events = await collect_events(config, context)

    final_messages = events[-1].messages
    assistant_texts = [
        message.content[0].text for message in final_messages if message.role == "assistant"
    ]
    assert assistant_texts == ["第一答", "第二答"]


# ---------------------------------------------------------------------------
# 截断保护（checklist）
# ---------------------------------------------------------------------------


async def test_truncated_tool_calls_all_failed() -> None:
    """stopReason=length：2 个工具调用均不执行、isError、错误文本含截断语义。"""
    chunks = [
        {
            "id": "chatcmpl-trunc",
            "model": "faux-gpt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "t1",
                                "type": "function",
                                "function": {"name": "echo", "arguments": '{"text": "a"}'},
                            },
                            {
                                "index": 1,
                                "id": "t2",
                                "type": "function",
                                "function": {"name": "echo", "arguments": '{"text": "b"}'},
                            },
                        ],
                    },
                    "finish_reason": "length",
                }
            ],
        },
        {"id": "chatcmpl-trunc", "model": "faux-gpt", "choices": []},
    ]
    provider = make_scripted(
        [
            FauxFixture(protocol="openai", model="faux-gpt", chunks=chunks),
            # 判错批后模型重发/转文本（单脚本重放会无限循环）
            FauxFixture(
                protocol="openai", model="faux-gpt", chunks=openai_text_chunks("已重新表述")
            ),
        ]
    )
    echo = EchoTool("echo")
    context = AgentContext(tools=[echo])
    config = make_config(provider)

    events = await collect_events(config, context)

    assert echo.executed == []

    tool_results = [message for message in events[-1].messages if message.role == "toolResult"]
    assert len(tool_results) == 2
    for result in tool_results:
        assert result.is_error is True
        text = result.content[0].text
        assert "not executed" in text
        assert "truncated" in text
        assert "Re-issue" in text

    ends = [event for event in events if event.type == "tool_execution_end"]
    assert len(ends) == 2
    assert all(end.is_error for end in ends)


# ---------------------------------------------------------------------------
# 并行批（checklist：完成序 end、源序消息）
# ---------------------------------------------------------------------------


async def test_parallel_batch_source_order_backfill() -> None:
    """并行批：工具结果消息按助手消息源序回填（checklist）。"""
    provider = make_scripted(
        [
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_two_tool_calls_chunks()),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("fin")),
        ]
    )
    fast = EchoTool("fast", delay=0.0)
    slow = EchoTool("slow", delay=0.05)
    context = AgentContext(tools=[fast, slow])
    config = make_config(provider)

    events = await collect_events(config, context)

    assert fast.executed == [{"text": "f"}]
    assert slow.executed == [{"text": "s"}]

    tool_results = [message for message in events[-1].messages if message.role == "toolResult"]
    assert [message.tool_name for message in tool_results] == ["fast", "slow"]

    message_starts = [
        event
        for event in events
        if event.type == "message_start" and event.message.role == "toolResult"
    ]
    assert [event.message.tool_name for event in message_starts] == ["fast", "slow"]


async def test_parallel_completion_order_emitted() -> None:
    """并行批：慢工具先 start 但后 end——end 按完成序（checklist）。"""
    provider = make_scripted(
        [
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_two_tool_calls_chunks()),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("fin")),
        ]
    )
    fast = EchoTool("fast", delay=0.0)
    slow = EchoTool("slow", delay=0.05)
    context = AgentContext(tools=[fast, slow])
    config = make_config(provider)

    events = await collect_events(config, context)
    ends = [event for event in events if event.type == "tool_execution_end"]
    assert [end.tool_name for end in ends] == ["fast", "slow"]  # fast 先完成


async def test_sequential_mode_executes_in_order() -> None:
    """串行模式：配置 sequential 时逐个执行。"""
    provider = make_scripted(
        [
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_two_tool_calls_chunks()),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("fin")),
        ]
    )
    fast = EchoTool("fast")
    slow = EchoTool("slow")
    context = AgentContext(tools=[fast, slow])
    config = make_config(provider, tool_execution="sequential")

    events = await collect_events(config, context)
    tool_results = [message for message in events[-1].messages if message.role == "toolResult"]
    assert [message.tool_name for message in tool_results] == ["fast", "slow"]
    assert fast.executed and slow.executed


async def test_sequential_tool_forces_serial_batch() -> None:
    """批内含 sequential 工具（execution_mode 声明）时走串行批。"""
    provider = make_scripted(
        [
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_two_tool_calls_chunks()),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("fin")),
        ]
    )
    fast = EchoTool("fast")
    fast.execution_mode = "sequential"
    slow = EchoTool("slow")
    context = AgentContext(tools=[fast, slow])
    config = make_config(provider)  # 默认 parallel，但批内有 sequential 工具

    events = await collect_events(config, context)
    tool_results = [message for message in events[-1].messages if message.role == "toolResult"]
    assert [message.tool_name for message in tool_results] == ["fast", "slow"]


# ---------------------------------------------------------------------------
# 钩子（checklist：before block 的 reason 原样出现）
# ---------------------------------------------------------------------------


async def test_before_tool_call_blocks_with_reason() -> None:
    """before 钩子 block：工具不执行，reason 原样出现在结果文本。"""
    provider = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "echo", '{"text": "x"}'),
            ),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("ok")),
        ]
    )
    echo = EchoTool("echo")
    context = AgentContext(tools=[echo])

    async def before(ctx: Any) -> BeforeToolCallResult:
        return BeforeToolCallResult(block=True, reason="安全策略禁止执行该工具")

    config = make_config(provider, before_tool_call=before)
    events = await collect_events(config, context)

    assert echo.executed == []
    tool_results = [message for message in events[-1].messages if message.role == "toolResult"]
    assert tool_results[0].is_error is True
    assert tool_results[0].content[0].text == "安全策略禁止执行该工具"


async def test_after_tool_call_overrides_content() -> None:
    """after 钩子：content 覆盖。"""
    provider = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "echo", '{"text": "x"}'),
            ),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("ok")),
        ]
    )
    context = AgentContext(tools=[EchoTool("echo")])

    async def after(ctx: Any) -> AfterToolCallResult:
        return AfterToolCallResult(content=[TextBlock(text="改写后的结果")])

    config = make_config(provider, after_tool_call=after)
    events = await collect_events(config, context)

    tool_results = [message for message in events[-1].messages if message.role == "toolResult"]
    assert tool_results[0].content[0].text == "改写后的结果"


async def test_prepare_next_turn_switches_model() -> None:
    """prepare_next_turn：第二轮换模型（checklist：模型 id 断言）。"""
    provider = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "echo", '{"text": "x"}'),
            ),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("ok")),
        ]
    )
    new_model = ModelInfo(id="faux-gpt-other", provider="faux-openai", api="openai")
    context = AgentContext(tools=[EchoTool("echo")])

    async def prepare(ctx: Any) -> TurnUpdate:
        return TurnUpdate(model=new_model)

    config = make_config(provider, prepare_next_turn=prepare)
    await collect_events(config, context)

    assert provider.models_seen == ["faux-gpt", "faux-gpt-other"]


async def test_should_stop_after_turn_exits() -> None:
    """should_stop_after_turn：返回 True 时优雅停止（正常事件序列）。"""
    provider = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "echo", '{"text": "x"}'),
            ),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("never")),
        ]
    )
    context = AgentContext(tools=[EchoTool("echo")])

    async def should_stop(ctx: Any) -> bool:
        return True

    config = make_config(provider, should_stop_after_turn=should_stop)
    events = await collect_events(config, context)

    assert len(provider.models_seen) == 1
    assert events[-1].type == "agent_end"
    types = event_types(events)
    assert types[-2] == "turn_end"


# ---------------------------------------------------------------------------
# 错误流与参数验证
# ---------------------------------------------------------------------------


async def test_stream_error_ends_agent_normally() -> None:
    """流 error 终态：turn_end + agent_end 正常收束（不抛异常）。"""
    provider = make_scripted(
        [FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_error_chunks())]
    )
    context = AgentContext()
    config = make_config(provider)

    events = await collect_events(config, context)
    types = event_types(events)
    assert types[-1] == "agent_end"
    agent_end = events[-1]
    assistant = [m for m in agent_end.messages if m.role == "assistant"][-1]
    assert assistant.stop_reason == "error"
    assert assistant.error_message is not None


async def test_unknown_tool_call_fails_gracefully() -> None:
    """未知工具：错误结果（不崩溃）。"""
    provider = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "no_such_tool", "{}"),
            ),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("ok")),
        ]
    )
    context = AgentContext(tools=[EchoTool("echo")])
    config = make_config(provider)

    events = await collect_events(config, context)
    tool_results = [message for message in events[-1].messages if message.role == "toolResult"]
    assert tool_results[0].is_error is True
    assert "not found" in tool_results[0].content[0].text


async def test_invalid_tool_arguments_fail_gracefully() -> None:
    """参数验证失败：错误结果含验证细节（模型可自纠错）。"""
    provider = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "echo", "{}"),  # 缺必填 text
            ),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("ok")),
        ]
    )
    context = AgentContext(tools=[EchoTool("echo")])
    config = make_config(provider)

    events = await collect_events(config, context)
    tool_results = [message for message in events[-1].messages if message.role == "toolResult"]
    assert tool_results[0].is_error is True
    text = tool_results[0].content[0].text
    assert "Validation failed" in text
    assert "text" in text


async def test_tool_execution_exception_becomes_error_result() -> None:
    """工具执行抛异常：归一为错误结果（不中断 loop）。"""

    class BoomTool(AgentTool):
        name = "boom"
        description = "always fails"

        def parameters_schema(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(
            self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None
        ) -> ToolResult:
            raise RuntimeError("磁盘已满")

    provider = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "boom", "{}"),
            ),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("ok")),
        ]
    )
    context = AgentContext(tools=[BoomTool()])
    config = make_config(provider)

    events = await collect_events(config, context)
    tool_results = [message for message in events[-1].messages if message.role == "toolResult"]
    assert tool_results[0].is_error is True
    assert "磁盘已满" in tool_results[0].content[0].text


async def test_tool_update_events_emitted() -> None:
    """工具进度回调 → tool_execution_update 事件。"""

    class ProgressTool(AgentTool):
        name = "progress"
        description = "reports progress"

        def parameters_schema(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(
            self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None
        ) -> ToolResult:
            if on_update is not None:
                await on_update(ToolResult(content=[TextBlock(text="中间结果")]))
            return self.text_result("最终结果")

    provider = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "progress", "{}"),
            ),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("ok")),
        ]
    )
    context = AgentContext(tools=[ProgressTool()])
    config = make_config(provider)

    events = await collect_events(config, context)
    updates = [event for event in events if event.type == "tool_execution_update"]
    assert len(updates) == 1
    assert updates[0].partial_result.content[0].text == "中间结果"


# ---------------------------------------------------------------------------
# agent_loop_continue
# ---------------------------------------------------------------------------


async def test_agent_loop_continue_requires_valid_tail() -> None:
    """continue 前置校验：空上下文 / assistant 结尾均拒绝。"""
    provider = make_scripted(
        [FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("x"))]
    )
    config = make_config(provider)

    with pytest.raises(ValueError, match="no messages"):
        async for _ in agent_loop_continue(AgentContext(), config):
            pass

    assistant_tail_context = AgentContext(messages=[UserMessage(content="hi"), assistant_tail()])
    with pytest.raises(ValueError, match="assistant"):
        async for _ in agent_loop_continue(assistant_tail_context, config):
            pass


async def test_agent_loop_continue_from_tool_result() -> None:
    """continue：从 toolResult 结尾的上下文继续（本轮流式）。"""
    provider = make_scripted(
        [FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("续"))]
    )
    context = AgentContext(
        messages=[UserMessage(content="hi"), assistant_tail(), tool_result_tail()]
    )
    config = make_config(provider)

    events = [event async for event in agent_loop_continue(context, config)]
    assert events[-1].type == "agent_end"
    assert [m.role for m in events[-1].messages] == ["assistant"]


# ---------------------------------------------------------------------------
# terminate 早停
# ---------------------------------------------------------------------------


async def test_terminate_all_batch_stops() -> None:
    """批内全部 terminate=True：agent 停止（对齐 pi 早停规则）。"""

    class TerminatorTool(AgentTool):
        name = "stopper"
        description = "requests termination"

        def parameters_schema(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(
            self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None
        ) -> ToolResult:
            result = self.text_result("done, stop")
            result.terminate = True
            return result

    provider = make_scripted(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "stopper", "{}"),
            ),
            FauxFixture(protocol="openai", model="faux-gpt", chunks=openai_text_chunks("never")),
        ]
    )
    context = AgentContext(tools=[TerminatorTool()])
    config = make_config(provider)

    events = await collect_events(config, context)
    assert len(provider.models_seen) == 1
    assert events[-1].type == "agent_end"
