"""T2 类型系统测试：模型结构、判别联合与 JSON 往返序列化。

checklist 对应项：
- 每类消息与事件 JSON 序列化→反序列化后与原对象相等
- 用量模型缺省为 0 且序列化稳定
- 事件命名与 pi 对齐（agent_start / turn_end / tool_execution_update 等）
"""

import time
from typing import cast

import pytest
from pydantic import BaseModel, ValidationError

from mimcode.types import (
    AgentEnd,
    AgentStart,
    AssistantMessage,
    ImageBlock,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    StreamDone,
    StreamError,
    StreamStart,
    StreamTextDelta,
    StreamTextEnd,
    StreamThinkingDelta,
    StreamToolCallEnd,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    ToolResult,
    ToolResultMessage,
    TurnEnd,
    TurnStart,
    Usage,
    UserMessage,
)

# 固定时间戳保证样本确定性
TS = 1730000000123


def make_assistant(**overrides: object) -> AssistantMessage:
    """构造覆盖全部四类内容块的助手消息样本。"""
    fields: dict[str, object] = {
        "content": [
            ThinkingBlock(thinking="先分析需求", thinking_signature="sig-1"),
            TextBlock(text="你好"),
            ImageBlock(data="aGk=", mime_type="image/png"),
            ToolCallBlock(id="call-1", name="bash", arguments={"command": "echo hi"}),
        ],
        "api": "anthropic",
        "provider": "anthropic",
        "model": "claude-sonnet",
        "usage": Usage(input=10, output=20, cache_read=5, cache_write=3, reasoning=7),
        "stop_reason": "toolUse",
        "timestamp": TS,
    }
    fields.update(overrides)
    return AssistantMessage.model_validate(fields)


def make_tool_result_message() -> ToolResultMessage:
    """构造工具结果消息样本（含 details 与 usage）。"""
    return ToolResultMessage(
        tool_call_id="call-1",
        tool_name="bash",
        content=[TextBlock(text="hi")],
        details={"exit_code": 0, "cwd": "/tmp"},
        usage=Usage(input=1, output=2),
        is_error=False,
        timestamp=TS,
    )


def make_user() -> UserMessage:
    return UserMessage(content="列出目录", timestamp=TS)


def roundtrip_samples() -> list[tuple[str, BaseModel]]:
    """全部消息/事件类型的往返序列化样本。"""
    assistant = make_assistant()
    user = make_user()
    tool_result_message = make_tool_result_message()
    tool_result = ToolResult(
        content=[TextBlock(text="hi")],
        details={"exit_code": 0},
        terminate=False,
    )
    return [
        # 消息与工具结果
        ("user", user),
        ("assistant", assistant),
        ("tool_result_message", tool_result_message),
        ("tool_result", tool_result),
        ("usage", Usage()),
        # 助手流事件
        ("stream_start", StreamStart(partial=assistant)),
        ("stream_text_delta", StreamTextDelta(content_index=1, delta="你", partial=assistant)),
        ("stream_text_end", StreamTextEnd(content_index=1, content="你好", partial=assistant)),
        (
            "stream_thinking_delta",
            StreamThinkingDelta(content_index=0, delta="先", partial=assistant),
        ),
        (
            "stream_toolcall_end",
            StreamToolCallEnd(
                content_index=3,
                tool_call=cast("ToolCallBlock", assistant.content[3]),
                partial=assistant,
            ),
        ),
        ("stream_done", StreamDone(reason="toolUse", message=assistant)),
        (
            "stream_error",
            StreamError(
                reason="error",
                error=make_assistant(stop_reason="error", error_message="连接失败"),
            ),
        ),
        # Agent 事件
        ("agent_start", AgentStart()),
        ("agent_end", AgentEnd(messages=[user, assistant, tool_result_message])),
        ("turn_start", TurnStart()),
        ("turn_end", TurnEnd(message=assistant, tool_results=[tool_result_message])),
        ("message_start", MessageStart(message=user)),
        (
            "message_update",
            MessageUpdate(
                message=assistant,
                assistant_event=StreamTextDelta(content_index=1, delta="你", partial=assistant),
            ),
        ),
        ("message_end", MessageEnd(message=assistant)),
        (
            "tool_execution_start",
            ToolExecutionStart(
                tool_call_id="call-1", tool_name="bash", args={"command": "echo hi"}
            ),
        ),
        (
            "tool_execution_update",
            ToolExecutionUpdate(
                tool_call_id="call-1",
                tool_name="bash",
                args={"command": "echo hi"},
                partial_result={"lines": 1},
            ),
        ),
        (
            "tool_execution_end",
            ToolExecutionEnd(
                tool_call_id="call-1",
                tool_name="bash",
                result=tool_result,
                is_error=False,
            ),
        ),
    ]


SAMPLES = roundtrip_samples()


@pytest.mark.parametrize(
    ("name", "sample"),
    SAMPLES,
    ids=[name for name, _ in SAMPLES],
)
def test_json_roundtrip(name: str, sample: BaseModel) -> None:
    """JSON 序列化→反序列化后与原对象相等。"""
    dumped = sample.model_dump_json()
    restored = type(sample).model_validate_json(dumped)
    assert restored == sample, f"{name} 往返不一致: {dumped}"


def test_usage_defaults_and_total() -> None:
    """用量缺省全 0，total_tokens 为四项之和，序列化键稳定。"""
    usage = Usage()
    assert usage.total_tokens == 0
    assert usage.model_dump() == {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning": 0,
    }
    assert Usage(input=100, output=50, cache_read=25, cache_write=25).total_tokens == 200


def test_event_type_tags_align_with_pi() -> None:
    """事件 type 字符串与 pi 命名逐一对齐。"""
    assistant = make_assistant()
    tool_result = ToolResult(content=[])
    events = [
        AgentStart(),
        AgentEnd(messages=[]),
        TurnStart(),
        TurnEnd(message=assistant, tool_results=[]),
        MessageStart(message=make_user()),
        MessageEnd(message=assistant),
        ToolExecutionStart(tool_call_id="c", tool_name="bash", args={}),
        ToolExecutionUpdate(tool_call_id="c", tool_name="bash", args={}, partial_result=None),
        ToolExecutionEnd(tool_call_id="c", tool_name="bash", result=tool_result, is_error=False),
    ]
    expected = [
        "agent_start",
        "agent_end",
        "turn_start",
        "turn_end",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
    ]
    assert [e.type for e in events] == expected


def test_stream_event_type_tags_align_with_pi() -> None:
    """流事件 type 字符串与 pi 命名逐一对齐。"""
    assistant = make_assistant()
    stream_events = [
        StreamStart(partial=assistant),
        StreamTextDelta(content_index=0, delta="x", partial=assistant),
        StreamTextEnd(content_index=0, content="x", partial=assistant),
        StreamThinkingDelta(content_index=0, delta="x", partial=assistant),
        StreamToolCallEnd(
            content_index=0,
            tool_call=ToolCallBlock(id="c", name="bash", arguments={}),
            partial=assistant,
        ),
        StreamDone(reason="stop", message=assistant),
        StreamError(reason="aborted", error=assistant),
    ]
    expected = [
        "start",
        "text_delta",
        "text_end",
        "thinking_delta",
        "toolcall_end",
        "done",
        "error",
    ]
    assert [e.type for e in stream_events] == expected


def test_discriminator_rejects_unknown_role() -> None:
    """未知 role 被判别联合拒绝。"""
    with pytest.raises(ValidationError):
        AssistantMessage.model_validate(
            {
                "role": "system",
                "api": "openai",
                "provider": "openai",
                "model": "gpt",
            }
        )


def test_discriminator_rejects_unknown_block_type() -> None:
    """未知内容块 type 被判别联合拒绝。"""
    with pytest.raises(ValidationError):
        AssistantMessage.model_validate(
            {
                "role": "assistant",
                "api": "openai",
                "provider": "openai",
                "model": "gpt",
                "content": [{"type": "audio", "text": "x"}],
            }
        )


def test_stream_error_carries_terminal_message() -> None:
    """error 流事件携带 stopReason 为 error/aborted 的终态消息（pi 流契约）。"""
    for reason in ("error", "aborted"):
        terminal = make_assistant(stop_reason=reason, error_message="boom")
        event = StreamError(reason=reason, error=terminal)
        assert event.error.stop_reason == reason
        assert event.error.error_message == "boom"


def test_user_message_accepts_string_or_blocks() -> None:
    """用户消息 content 兼容纯字符串与块列表两种形态。"""
    assert make_user().content == "列出目录"
    message = UserMessage(
        content=[TextBlock(text="看这个"), ImageBlock(data="aGk=", mime_type="image/png")],
        timestamp=TS,
    )
    assert isinstance(message.content, list)
    assert len(message.content) == 2
    assert message.content[1].type == "image"


def test_timestamp_defaults_to_now_ms() -> None:
    """未显式指定 timestamp 时默认取当前毫秒时间戳。"""
    before = int(time.time() * 1000)
    message = UserMessage(content="hi")
    after = int(time.time() * 1000)
    assert before <= message.timestamp <= after
