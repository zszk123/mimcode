"""流契约测试：失败编码为流内 error 事件，不抛异常（checklist T3 验收）。

对齐 pi 的 StreamFn 契约（packages/agent/src/types.ts L19-32）：
请求/模型/运行期失败不抛出，而是产出 error 事件 +
stopReason 为 error/aborted 的终态消息。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mimcode.provider.faux import FauxAnthropicProvider, FauxFixture, FauxOpenAIProvider
from mimcode.types import AssistantStreamEvent, LlmContext, StreamError, UserMessage

FIXTURES = Path(__file__).parent / "fixtures"


def load_openai(name: str) -> FauxOpenAIProvider:
    return FauxOpenAIProvider.from_file(FIXTURES / f"{name}.json")


def load_anthropic(name: str) -> FauxAnthropicProvider:
    return FauxAnthropicProvider.from_file(FIXTURES / f"{name}.json")


async def collect_openai(provider: FauxOpenAIProvider, options=None) -> list[AssistantStreamEvent]:
    model = provider.get_models()[0]
    context = LlmContext(messages=[UserMessage(content="hi")])
    return [event async for event in provider.stream(model, context, options)]


async def collect_anthropic(
    provider: FauxAnthropicProvider, options=None
) -> list[AssistantStreamEvent]:
    model = provider.get_models()[0]
    context = LlmContext(messages=[UserMessage(content="hi")])
    return [event async for event in provider.stream(model, context, options)]


async def test_openai_connection_error_encoded_in_stream() -> None:
    """连接失败：不抛异常，仅产出 error 事件 + 终态错误消息。"""
    provider = load_openai("openai_error")
    events = await collect_openai(provider)
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, StreamError)
    assert error.reason == "error"
    assert error.error.stop_reason == "error"
    assert "faux connection error" in (error.error.error_message or "")


async def test_anthropic_connection_error_encoded_in_stream() -> None:
    """连接失败（anthropic）：不抛异常，仅产出 error 事件。"""
    provider = load_anthropic("anthropic_error")
    events = await collect_anthropic(provider)
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, StreamError)
    assert error.reason == "error"
    assert error.error.stop_reason == "error"
    assert "faux connection error" in (error.error.error_message or "")


async def test_openai_abort_before_first_chunk() -> None:
    """请求开始前中止：仅产出 aborted 终态事件。"""
    provider = load_openai("openai_text")
    signal = asyncio.Event()
    signal.set()
    events = await collect_openai(
        provider,
        options=__import__("mimcode.types", fromlist=["StreamOptions"]).StreamOptions(
            signal=signal
        ),
    )
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, StreamError)
    assert error.reason == "aborted"
    assert error.error.stop_reason == "aborted"


async def test_anthropic_abort_before_first_chunk() -> None:
    """请求开始前中止（anthropic）：仅产出 aborted 终态事件。"""
    provider = load_anthropic("anthropic_text")
    signal = asyncio.Event()
    signal.set()
    from mimcode.types import StreamOptions

    events = await collect_anthropic(provider, options=StreamOptions(signal=signal))
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, StreamError)
    assert error.reason == "aborted"


async def test_openai_abort_mid_stream() -> None:
    """流中途中止：已产出事件保留，以 aborted 终态收尾。"""
    provider = load_openai("openai_text")
    signal = asyncio.Event()
    from mimcode.types import StreamOptions

    events: list[AssistantStreamEvent] = []
    async for event in provider.stream(
        provider.get_models()[0], LlmContext(), StreamOptions(signal=signal)
    ):
        events.append(event)
        if event.type == "start":
            signal.set()
    assert events[0].type == "start"
    last = events[-1]
    assert isinstance(last, StreamError)
    assert last.reason == "aborted"
    assert last.error.stop_reason == "aborted"


async def test_openai_zero_chunks_is_protocol_error() -> None:
    """零 chunk：协议违约 → error 事件。"""
    provider = FauxOpenAIProvider(FauxFixture(protocol="openai", model="faux-gpt"))
    events = await collect_openai(provider)
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, StreamError)
    assert error.reason == "error"
    assert "Stream ended without any chunks" in (error.error.error_message or "")


async def test_anthropic_missing_stop_reason() -> None:
    """流结束但缺 stop reason：协议违约 → error 事件。"""
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "m",
                "model": "faux-claude",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    provider = FauxAnthropicProvider(
        FauxFixture(protocol="anthropic", model="faux-claude", chunks=chunks)
    )
    events = await collect_anthropic(provider)
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, StreamError)
    assert "without a stop reason" in (error.error.error_message or "")


async def test_anthropic_unknown_stop_reason() -> None:
    """未知 stop reason：协议违约 → error 事件（对齐 pi 的 throw 语义）。"""
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "m",
                "model": "faux-claude",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        },
        {"type": "message_delta", "delta": {"stop_reason": "weird_new_reason"}},
        {"type": "message_stop"},
    ]
    provider = FauxAnthropicProvider(
        FauxFixture(protocol="anthropic", model="faux-claude", chunks=chunks)
    )
    events = await collect_anthropic(provider)
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, StreamError)
    assert "Unhandled stop reason: weird_new_reason" in (error.error.error_message or "")


def test_zero_chunk_fixture_shape() -> None:
    """fixture 模型：空 chunks 合法构造，非法协议被校验拒绝。"""
    fixture = FauxFixture(protocol="openai", model="m")
    assert fixture.chunks == []
    assert fixture.error is None
    with pytest.raises(ValidationError):
        FauxFixture(protocol="google", model="m")
