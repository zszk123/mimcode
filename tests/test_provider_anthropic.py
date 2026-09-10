"""Claude 协议 provider 测试。

checklist 对应项：
- content_block_delta 三种增量（thinking/text/tool_use 输入）事件流均正确产出
- thinking 预算映射与 payload 消息转换（连续 toolResult 分组等）
"""

from __future__ import annotations

from pathlib import Path

from mimcode.provider.anthropic_protocol import convert_anthropic_messages
from mimcode.provider.faux import FauxAnthropicProvider
from mimcode.types import (
    AssistantMessage,
    AssistantStreamEvent,
    LlmContext,
    ModelInfo,
    StreamOptions,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> FauxAnthropicProvider:
    return FauxAnthropicProvider.from_file(FIXTURES / f"{name}.json")


async def collect(
    provider: FauxAnthropicProvider,
    model: ModelInfo,
    context: LlmContext,
    options: StreamOptions | None = None,
) -> list[AssistantStreamEvent]:
    """收集一次流式调用的全部事件（顺带完成 payload 记录）。"""
    return [event async for event in provider.stream(model, context, options)]


async def collect_events(name: str) -> list[AssistantStreamEvent]:
    """加载 fixture 并收集流事件。"""
    provider = load(name)
    return await collect(provider, provider.get_models()[0], LlmContext())


def make_anthropic_model(**overrides: object) -> ModelInfo:
    """构造测试模型元数据。"""
    fields: dict[str, object] = {
        "id": "faux-claude",
        "provider": "faux-anthropic",
        "api": "anthropic",
        "max_output_tokens": 32768,
        "supports_thinking": True,
    }
    fields.update(overrides)
    return ModelInfo.model_validate(fields)


async def test_full_stream_thinking_text_tool() -> None:
    """thinking + text + tool_use：三种增量事件流均正确产出。"""
    events = await collect_events("anthropic_thinking_text_tool")
    assert [e.type for e in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    done = events[-1]
    assert done.type == "done"
    thinking, text, tool = done.message.content
    # signature_delta 累积进签名槽
    assert thinking == ThinkingBlock(thinking="思考", thinking_signature="sig-1")
    assert text == TextBlock(text="你好")
    assert tool == ToolCallBlock(id="toolu_1", name="bash", arguments={"command": "echo hi"})
    assert done.message.stop_reason == "toolUse"


async def test_usage_merge_and_reasoning_tokens() -> None:
    """message_start 初始用量 + message_delta 更新 + thinking_tokens → reasoning。"""
    events = await collect_events("anthropic_thinking_text_tool")
    done = events[-1]
    assert done.type == "done"
    usage = done.message.usage
    assert usage.input == 25
    assert usage.output == 40
    assert usage.cache_read == 4
    assert usage.cache_write == 6
    assert usage.reasoning == 12


async def test_text_stream_simple() -> None:
    """纯文本流：序列与终态。"""
    events = await collect_events("anthropic_text")
    assert [e.type for e in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    done = events[-1]
    assert done.type == "done"
    assert done.message.content == [TextBlock(text="你好")]
    assert done.message.stop_reason == "stop"
    assert done.message.usage.input == 25
    assert done.message.usage.output == 6


async def test_redacted_thinking_block() -> None:
    """redacted_thinking → 标记 redacted 的 thinking 块，不透明载荷在签名槽。"""
    events = await collect_events("anthropic_redacted")
    done = events[-1]
    assert done.type == "done"
    redacted = done.message.content[0]
    assert redacted == ThinkingBlock(
        thinking="[Reasoning redacted]",
        thinking_signature="opaque-redacted-payload",
        redacted=True,
    )


async def test_thinking_budget_mapping() -> None:
    """级别 → 预算：min(级别默认, max_tokens - 1024)，低于 1024 不启用。"""
    provider = load("anthropic_text")
    model = make_anthropic_model()
    context = LlmContext(system_prompt="系统提示", messages=[UserMessage(content="hi")])

    # high=16384，上限 32768-1024=31744 → 16384
    await collect(provider, model, context, StreamOptions(thinking_level="high"))
    payload = provider.requests[-1]
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert payload["system"] == "系统提示"
    assert payload["max_tokens"] == 32768

    # max_tokens=4096 → 预算压到 3072
    small = make_anthropic_model(max_output_tokens=4096)
    await collect(provider, small, context, StreamOptions(thinking_level="high"))
    assert provider.requests[-1]["thinking"]["budget_tokens"] == 3072

    # max_tokens=2048 → 预算 1024（恰好达 Anthropic 下限，仍启用）
    tiny = make_anthropic_model(max_output_tokens=2048)
    await collect(provider, tiny, context, StreamOptions(thinking_level="high"))
    assert provider.requests[-1]["thinking"]["budget_tokens"] == 1024

    # max_tokens=1024 → 无答案余量，不启用 thinking
    no_room = make_anthropic_model(max_output_tokens=1024)
    await collect(provider, no_room, context, StreamOptions(thinking_level="high"))
    assert "thinking" not in provider.requests[-1]

    # 未声明支持 thinking 的模型不启用
    plain = make_anthropic_model(supports_thinking=False)
    await collect(provider, plain, context, StreamOptions(thinking_level="high"))
    assert "thinking" not in provider.requests[-1]

    # 未传级别不启用
    await collect(provider, model, context, None)
    assert "thinking" not in provider.requests[-1]


async def test_payload_tool_result_grouping() -> None:
    """连续 toolResult 分组为单条 user 消息；thinking/tool_use 回传规则。"""
    provider = load("anthropic_text")
    model = make_anthropic_model(supports_thinking=False)
    context = LlmContext(
        messages=[
            AssistantMessage(
                content=[
                    ThinkingBlock(thinking="推理", thinking_signature="sig"),
                    TextBlock(text="调用工具"),
                    ToolCallBlock(id="t1", name="bash", arguments={"command": "ls"}),
                ],
                api="anthropic",
                provider="faux-anthropic",
                model="faux-claude",
                timestamp=1,
            ),
            ToolResultMessage(
                tool_call_id="t1",
                tool_name="bash",
                content=[TextBlock(text="a.txt")],
                timestamp=2,
            ),
            ToolResultMessage(
                tool_call_id="t2",
                tool_name="read",
                content=[TextBlock(text="内容")],
                is_error=True,
                timestamp=3,
            ),
        ],
    )
    await collect(provider, model, context)
    assert provider.requests[-1]["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "推理", "signature": "sig"},
                {"type": "text", "text": "调用工具"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "a.txt"}],
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": [{"type": "text", "text": "内容"}],
                    "is_error": True,
                },
            ],
        },
    ]


async def test_payload_tools_and_empty_user_text_skipped() -> None:
    """工具声明 → input_schema；空文本用户消息被剔除。"""
    provider = load("anthropic_text")
    model = make_anthropic_model(supports_thinking=False)
    context = LlmContext(
        messages=[
            UserMessage(content="   ", timestamp=1),
            UserMessage(content="你好", timestamp=2),
        ],
        tools=[
            ToolSpec(
                name="bash",
                description="执行命令",
                parameters={"type": "object", "properties": {"command": {"type": "string"}}},
            )
        ],
    )
    await collect(provider, model, context)
    payload = provider.requests[-1]
    assert payload["messages"] == [{"role": "user", "content": "你好"}]
    assert payload["tools"] == [
        {
            "name": "bash",
            "description": "执行命令",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        }
    ]


def test_message_conversion_rules() -> None:
    """转换纯函数：无签名 thinking 降级 text、redacted 透传、空块剔除。"""
    context = LlmContext(
        messages=[
            AssistantMessage(
                content=[
                    TextBlock(text="  "),
                    ThinkingBlock(thinking="无签名思考"),
                    ThinkingBlock(thinking="密文", thinking_signature="data-1", redacted=True),
                    ToolCallBlock(id="t1", name="ls", arguments={}),
                ],
                api="anthropic",
                provider="p",
                model="m",
                timestamp=1,
            ),
        ],
    )
    params = convert_anthropic_messages(context)
    assert params == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "无签名思考"},
                {"type": "redacted_thinking", "data": "data-1"},
                {"type": "tool_use", "id": "t1", "name": "ls", "input": {}},
            ],
        }
    ]
