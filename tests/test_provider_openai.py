"""OpenAI 协议 provider 测试。

checklist 对应项：
- tool_calls 参数跨 chunk 拆分时最终 arguments 可被 json.loads 解析
- delta 带 reasoning_content 时产生 thinking 事件且终态含 thinking 块
"""

from __future__ import annotations

import json
from pathlib import Path

from mimcode.provider.faux import FauxOpenAIProvider
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


def load(name: str) -> FauxOpenAIProvider:
    return FauxOpenAIProvider.from_file(FIXTURES / f"{name}.json")


async def collect(
    provider: FauxOpenAIProvider,
    model: ModelInfo,
    context: LlmContext,
    options: StreamOptions | None = None,
) -> list[AssistantStreamEvent]:
    """收集一次流式调用的全部事件。"""
    return [event async for event in provider.stream(model, context, options)]


def make_full_context() -> LlmContext:
    """构造覆盖 system/user/assistant(toolCall)/toolResult 的上下文。"""
    return LlmContext(
        system_prompt="你是助手",
        messages=[
            UserMessage(content="列出目录", timestamp=1),
            AssistantMessage(
                content=[
                    ToolCallBlock(
                        id="call-1", name="bash", arguments={"command": "ls"}, type="toolCall"
                    ),
                    TextBlock(text="我用工具"),
                ],
                api="openai",
                provider="faux-openai",
                model="faux-gpt",
                timestamp=2,
            ),
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="bash",
                content=[TextBlock(text="file.txt")],
                timestamp=3,
            ),
        ],
        tools=[
            ToolSpec(
                name="bash",
                description="执行 shell 命令",
                parameters={"type": "object", "properties": {"command": {"type": "string"}}},
            )
        ],
    )


async def test_text_stream_event_sequence() -> None:
    """纯文本流：事件序列 start→text_start→text_delta×2→text_end→done。"""
    provider = load("openai_text")
    model = provider.get_models()[0]
    events = await collect(provider, model, LlmContext(messages=[UserMessage(content="你好")]))
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


async def test_text_stream_usage_parsed() -> None:
    """usage chunk 归一：cached_tokens → cache_read，reasoning_tokens → reasoning。"""
    provider = load("openai_text")
    model = provider.get_models()[0]
    events = await collect(provider, model, LlmContext(messages=[UserMessage(content="你好")]))
    done = events[-1]
    assert done.type == "done"
    assert done.message.usage.input == 10
    assert done.message.usage.output == 5
    assert done.message.usage.cache_read == 3
    assert done.message.usage.cache_write == 0
    assert done.message.usage.reasoning == 2


async def test_tool_call_arguments_json_parseable() -> None:
    """工具调用参数跨 chunk 拆分：最终 arguments 可被 json.loads 解析。"""
    provider = load("openai_tool_calls")
    model = provider.get_models()[0]
    events = await collect(provider, model, LlmContext())
    assert [e.type for e in events] == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    done = events[-1]
    assert done.type == "done"
    tool_block = done.message.content[0]
    assert isinstance(tool_block, ToolCallBlock)
    # checklist 验收：arguments 是可 JSON 往返的解析结果
    assert json.loads(json.dumps(tool_block.arguments)) == {"command": "echo hi"}
    assert tool_block.id == "call-1"
    assert tool_block.name == "bash"
    assert done.message.stop_reason == "toolUse"


async def test_tool_call_partial_arguments_progressive() -> None:
    """参数增量期间部分解析可用（null 占位），终态完整。"""
    provider = load("openai_tool_calls")
    model = provider.get_models()[0]
    partials: list[dict] = []
    async for event in provider.stream(model, LlmContext()):
        if event.type == "toolcall_delta":
            partials.append(event.partial.content[0].arguments)  # type: ignore[union-attr]
    assert partials == [{}, {"command": None}, {"command": "echo hi"}]


async def test_reasoning_content_becomes_thinking_events() -> None:
    """delta 带 reasoning_content：thinking 三段事件 + 终态 thinking 块。"""
    provider = load("openai_reasoning")
    model = provider.get_models()[0]
    events = await collect(provider, model, LlmContext())
    assert [e.type for e in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_delta",
        "text_start",
        "text_delta",
        "thinking_end",
        "text_end",
        "done",
    ]
    done = events[-1]
    assert done.type == "done"
    thinking, text = done.message.content
    assert thinking == ThinkingBlock(thinking="先分析", thinking_signature="reasoning_content")
    assert text == TextBlock(text="答案")
    assert done.message.stop_reason == "stop"


async def test_payload_conversion_round_trip() -> None:
    """payload 构建：system/user/assistant(toolCall)/toolResult 的完整转换。"""
    provider = load("openai_text")
    model = provider.get_models()[0]
    context = make_full_context()
    await collect(provider, model, context)

    assert len(provider.requests) == 1
    payload = provider.requests[0]
    assert payload["model"] == "faux-gpt"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["messages"] == [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "列出目录"},
        {
            "role": "assistant",
            "content": "我用工具",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "file.txt"},
    ]
    tool = payload["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "bash"
    assert tool["function"]["description"] == "执行 shell 命令"
    assert tool["function"]["parameters"]["type"] == "object"


async def test_reasoning_effort_only_for_thinking_models() -> None:
    """thinking 级别仅在模型声明支持时下发 reasoning_effort。"""
    provider = load("openai_reasoning")
    plain_model = provider.get_models()[0]
    thinking_model = ModelInfo(
        id="faux-o1", provider="faux-openai", api="openai", supports_thinking=True
    )
    context = LlmContext(messages=[UserMessage(content="hi")])
    options = StreamOptions(thinking_level="high")

    await collect(provider, plain_model, context, options)
    assert "reasoning_effort" not in provider.requests[-1]

    await collect(provider, thinking_model, context, options)
    assert provider.requests[-1]["reasoning_effort"] == "high"


async def test_max_tokens_passed_to_payload() -> None:
    """显式输出上限下发 max_tokens。"""
    provider = load("openai_text")
    model = provider.get_models()[0]
    await collect(provider, model, LlmContext(), StreamOptions(max_tokens=512))
    assert provider.requests[-1]["max_tokens"] == 512
