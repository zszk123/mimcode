"""T8 compaction 测试。

checklist 对应项：
- 触发阈值与默认值：reserve_tokens=16384 / keep_recent_tokens=20000 作为
  具名常量存在且被测试引用
- 构造上下文占用超过窗口 90% 的会话后 compact 触发，产出摘要消息且
  保留近期消息（keepRecent 语义）
- 无用量数据时按字符估算路径（中文按字符计数不丢失）
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mimcode.app.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    KEEP_RECENT_TOKENS,
    RESERVE_TOKENS,
    SUMMARIZATION_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    CompactionSettings,
    calculate_context_tokens,
    compact_result_from_entry,
    compact_session,
    estimate_context_tokens,
    estimate_tokens,
    find_cut_point,
    generate_summary,
    should_compact,
    summary_to_user_message,
)
from mimcode.app.session import SessionManager
from mimcode.types import (
    AssistantMessage,
    ImageBlock,
    LlmContext,
    StreamDone,
    StreamError,
    StreamOptions,
    StreamStart,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)

# ---------------------------------------------------------------------------
# 常量与估算
# ---------------------------------------------------------------------------


def test_default_settings_constants() -> None:
    """默认阈值具名常量（checklist：16384 / 20000）。"""
    assert RESERVE_TOKENS == 16384
    assert KEEP_RECENT_TOKENS == 20000
    assert DEFAULT_COMPACTION_SETTINGS.reserve_tokens == 16384
    assert DEFAULT_COMPACTION_SETTINGS.keep_recent_tokens == 20000
    assert DEFAULT_COMPACTION_SETTINGS.enabled is True


def test_calculate_context_tokens() -> None:
    """用量四项之和。"""
    usage = Usage(input=100, output=50, cache_read=25, cache_write=25)
    assert calculate_context_tokens(usage) == 200


def test_estimate_tokens_cjk_chars_counted() -> None:
    """字符估算：中文按字符计数（多字节不丢失、不截断）。"""
    # 8 个中文字符 → ceil(8/4) = 2 tokens
    assert estimate_tokens(UserMessage(content="一二三四五六七八")) == 2
    # 1 个中文字符 → ceil(1/4) = 1
    assert estimate_tokens(UserMessage(content="中")) == 1
    # 混合内容同样按字符数
    assert estimate_tokens(UserMessage(content="中文ab")) == math.ceil(4 / 4)


def test_estimate_tokens_message_kinds() -> None:
    """各类消息的估算：助手（text/thinking/toolCall）与工具结果。"""
    assistant = AssistantMessage(
        content=[TextBlock(text="abcd"), TextBlock(text="thinking-not-counted")],
        api="openai",
        provider="p",
        model="m",
    )
    # "abcd" 4 + "thinking-not-counted" 20 = 24 chars → 6
    assert estimate_tokens(assistant) == 6

    thinking_assistant = AssistantMessage(content=[], api="openai", provider="p", model="m")
    thinking_assistant.content.append(ThinkingBlock(thinking="x" * 8))
    thinking_assistant.content.append(
        ToolCallBlock(id="c", name="bash", arguments={"command": "ls"})
    )
    expected = 8 + len("bash") + len(str({"command": "ls"}))
    assert estimate_tokens(thinking_assistant) == math.ceil(expected / 4)

    result = ToolResultMessage(tool_call_id="c", tool_name="t", content=[TextBlock(text="x" * 12)])
    assert estimate_tokens(result) == 3

    image_user = UserMessage(
        content=[ImageBlock(data="aGk=", mime_type="image/png"), TextBlock(text="ab")]
    )
    # 图片 4800 + 2 文本字符
    assert estimate_tokens(image_user) == math.ceil(4802 / 4)


def test_estimate_context_tokens_usage_priority() -> None:
    """估算优先级：最后有效助手用量 + 其后字符估算。"""
    history: list[Any] = [
        UserMessage(content="早"),
        AssistantMessage(
            content=[TextBlock(text="答")],
            api="openai",
            provider="p",
            model="m",
            usage=Usage(input=50_000, output=2_000),
        ),
        UserMessage(content="尾部消息"),
        AssistantMessage(
            content=[TextBlock(text="尾答")],
            api="openai",
            provider="p",
            model="m",
        ),
    ]
    estimate = estimate_context_tokens(history)
    assert estimate.usage_tokens == 52_000
    trailing = estimate_tokens(history[2]) + estimate_tokens(history[3])
    assert estimate.trailing_tokens == trailing
    assert estimate.tokens == 52_000 + trailing
    assert estimate.last_usage_index == 1


def test_estimate_context_tokens_skips_invalid_usage() -> None:
    """aborted/error 的助手用量不可信：跳过。"""
    aborted = AssistantMessage(
        content=[],
        api="openai",
        provider="p",
        model="m",
        usage=Usage(input=99_000),
        stop_reason="aborted",
    )
    estimate = estimate_context_tokens([UserMessage(content="hi"), aborted])
    assert estimate.usage_tokens == 0
    assert estimate.last_usage_index is None


def test_estimate_context_tokens_no_usage_fallback() -> None:
    """无用量：全量字符估算。"""
    messages = [UserMessage(content="x" * 40), UserMessage(content="y" * 8)]
    estimate = estimate_context_tokens(messages)
    assert estimate.tokens == 10 + 2
    assert estimate.usage_tokens == 0


# ---------------------------------------------------------------------------
# 阈值
# ---------------------------------------------------------------------------


def test_should_compact_threshold() -> None:
    """阈值：tokens > window - reserve 触发；禁用不触发。"""
    settings = DEFAULT_COMPACTION_SETTINGS
    # 90% 占用（checklist 构造）：window=100000, tokens=92000
    assert should_compact(92_000, 100_000, settings) is True
    # 低于阈值
    assert should_compact(50_000, 100_000, settings) is False
    # 恰好等于阈值（不触发：严格大于）
    assert should_compact(100_000 - 16384, 100_000, settings) is False
    # 禁用
    disabled = CompactionSettings(enabled=False)
    assert should_compact(999_999, 100_000, disabled) is False


# ---------------------------------------------------------------------------
# 裁剪点
# ---------------------------------------------------------------------------


def make_pair(user_text: str, assistant_text: str) -> list[Any]:
    return [
        UserMessage(content=user_text),
        AssistantMessage(
            content=[TextBlock(text=assistant_text)],
            api="openai",
            provider="p",
            model="m",
        ),
    ]


def test_find_cut_point_keeps_recent_tail() -> None:
    """裁剪：保留尾部约 keepRecent，切在用户消息边界。"""
    messages: list[Any] = []
    for index in range(10):
        messages.extend(make_pair(f"问题{index}", "回答" * 10))
    # 尾部累计达不到 20000 → target=0 → 首个用户边界（index 2）
    cut = find_cut_point(messages, keep_recent_tokens=20000)
    assert cut == 2
    assert isinstance(messages[cut], UserMessage)


def test_find_cut_point_budget_neighborhood() -> None:
    """高 keep：从尾部累计，落点后的首个边界，保留段在预算邻域。"""
    messages: list[Any] = []
    for index in range(10):
        messages.extend(make_pair(f"问{index}", "答" * 400))  # 每答 400 字 → 100 tokens
    cut = find_cut_point(messages, keep_recent_tokens=300)
    assert isinstance(messages[cut], UserMessage)
    retained_estimate = sum(estimate_tokens(m) for m in messages[cut:])
    previous_boundary = max(b for b in range(2, cut + 1, 2) if b < cut)
    previous_retained = sum(estimate_tokens(m) for m in messages[previous_boundary:])
    # 保留段不超预算，往前一个边界就超（轮次粒度的邻域性）
    assert 200 <= retained_estimate < 304
    assert previous_retained > 300


def test_find_cut_point_no_boundary() -> None:
    """无用户消息边界（单条消息）→ None。"""
    assert find_cut_point([UserMessage(content="only")], 100) is None


def test_find_cut_point_never_splits_turn() -> None:
    """裁剪点必须是用户消息（不把 toolResult 与其助手拆开）。"""
    messages: list[Any] = []
    for index in range(6):
        messages.extend(make_pair(f"q{index}", "a"))
        messages.append(
            ToolResultMessage(
                tool_call_id=f"c{index}", tool_name="t", content=[TextBlock(text="r")]
            )
        )
    cut = find_cut_point(messages, keep_recent_tokens=1)
    assert isinstance(messages[cut], UserMessage)


# ---------------------------------------------------------------------------
# 摘要生成
# ---------------------------------------------------------------------------


def make_summary_stream(
    text: str, captured: list[tuple[Any, LlmContext, StreamOptions]] | None = None
):
    """固定文本摘要流（记录请求上下文）。"""

    async def stream_fn(model, context, options):
        if captured is not None:
            captured.append((model, context, options))
        partial = AssistantMessage(content=[], api="openai", provider="p", model="m")
        yield StreamStart(partial=partial)
        yield StreamDone(
            reason="stop",
            message=AssistantMessage(
                content=[TextBlock(text=text)],
                api="openai",
                provider="p",
                model="m",
                stop_reason="stop",
            ),
        )

    return stream_fn


def make_error_stream():
    """失败流（error 终态）。"""

    async def stream_fn(model, context, options):
        yield StreamError(
            reason="error",
            error=AssistantMessage(
                content=[],
                api="openai",
                provider="p",
                model="m",
                stop_reason="error",
                error_message="boomed",
            ),
        )

    return stream_fn


async def test_generate_summary_request_shape() -> None:
    """摘要请求：系统提示 + 历史 + 指令消息。"""
    captured: list[tuple[Any, LlmContext, StreamOptions]] = []
    stream_fn = make_summary_stream("结构化摘要", captured)
    history = [UserMessage(content="历史问题"), make_pair("q", "a")[1]]

    summary = await generate_summary(history, None, stream_fn)

    assert summary == "结构化摘要"
    context = captured[0][1]
    assert context.system_prompt == SUMMARIZATION_SYSTEM_PROMPT
    assert len(context.messages) == 3
    assert context.messages[-1].content == SUMMARIZATION_PROMPT
    assert context.messages[0].content == "历史问题"


async def test_generate_summary_failure_returns_none() -> None:
    """摘要流失败 → None（下次触发重试）。"""
    history = [UserMessage(content="x")]
    assert await generate_summary(history, None, make_error_stream()) is None


# ---------------------------------------------------------------------------
# 会话级端到端（checklist：90% 触发 + 保留尾部）
# ---------------------------------------------------------------------------


def make_assistant_with_usage(text: str, usage: Usage) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)],
        api="openai",
        provider="p",
        model="m",
        usage=usage,
        stop_reason="stop",
    )


def big_model() -> Any:
    from mimcode.types import ModelInfo

    return ModelInfo(id="m", provider="p", api="openai")


async def test_compact_session_end_to_end(tmp_path: Path) -> None:
    """端到端：超阈值会话压缩 → compaction 条目 → 上下文 = 摘要 + 保留尾。"""
    session = SessionManager.create(str(tmp_path), home=tmp_path)
    # 真实形态：最后一条助手的用量即当前上下文规模（超 90%）
    session.append_message(UserMessage(content="最初的问题"))
    session.append_message(make_assistant_with_usage("旧回答", Usage(input=2_000, output=100)))
    session.append_message(UserMessage(content="近期问题"))
    session.append_message(make_assistant_with_usage("近期回答", Usage(input=92_000, output=1_000)))

    captured: list[tuple[Any, LlmContext, StreamOptions]] = []
    stream_fn = make_summary_stream("这是压缩摘要", captured)

    entry_id = await compact_session(
        session,
        context_window=100_000,
        model=big_model(),
        stream_fn=stream_fn,
    )

    assert entry_id is not None

    # 摘要请求收到了被摘要的历史前缀 + 指令
    request_messages = captured[0][1].messages
    assert request_messages[0].content == "最初的问题"
    assert request_messages[-1].content == SUMMARIZATION_PROMPT

    # compaction 条目 extra 可还原
    entry = session.entry_by_id(entry_id)
    assert entry is not None and entry.type == "compaction"
    result = compact_result_from_entry(entry.extra)
    assert result is not None
    assert result.summary == "这是压缩摘要"
    assert result.tokens_before > 90_000

    # 压缩后上下文：[摘要用户消息] + [保留尾（近期问答）]
    context = session.build_context()
    texts = [
        m.content if isinstance(m, UserMessage) else m.content[0].text for m in context.messages
    ]
    assert len(context.messages) == 3
    assert "这是压缩摘要" in texts[0]
    assert texts[1:] == ["近期问题", "近期回答"]

    # 摘要消息含压缩标记与 token 量
    summary_message = context.messages[0]
    assert isinstance(summary_message, UserMessage)
    assert "[Compacted conversation summary]" in summary_message.content


async def test_compact_session_below_threshold_noop(tmp_path: Path) -> None:
    """低于阈值：不触发、无条目。"""
    session = SessionManager.create(str(tmp_path), home=tmp_path)
    session.append_message(UserMessage(content="小会话"))
    session.append_message(make_assistant_with_usage("答", Usage(input=1_000)))

    entry_id = await compact_session(
        session,
        context_window=200_000,
        model=big_model(),
        stream_fn=make_summary_stream("x"),
    )
    assert entry_id is None
    assert all(entry.type != "compaction" for entry in session.entries)


async def test_compact_session_summary_failure_noop(tmp_path: Path) -> None:
    """摘要失败：无条目（下次触发重试）。"""
    session = SessionManager.create(str(tmp_path), home=tmp_path)
    session.append_message(UserMessage(content="q1"))
    session.append_message(make_assistant_with_usage("a1", Usage(input=1_000)))
    session.append_message(UserMessage(content="q2"))
    session.append_message(make_assistant_with_usage("a2", Usage(input=92_000)))

    entry_id = await compact_session(
        session,
        context_window=100_000,
        model=big_model(),
        stream_fn=make_error_stream(),
    )
    assert entry_id is None
    assert all(entry.type != "compaction" for entry in session.entries)


async def test_compact_session_persists_and_reloads(tmp_path: Path) -> None:
    """compaction 持久化：重开会话仍感知压缩。"""
    session = SessionManager.create(str(tmp_path), home=tmp_path)
    session.append_message(UserMessage(content="旧问"))
    session.append_message(make_assistant_with_usage("旧答", Usage(input=1_000)))
    session.append_message(UserMessage(content="新问"))
    session.append_message(make_assistant_with_usage("新答", Usage(input=95_000)))

    entry_id = await compact_session(
        session,
        context_window=100_000,
        model=big_model(),
        stream_fn=make_summary_stream("持久摘要"),
    )
    assert entry_id is not None

    reopened = SessionManager.open(session.session_file)
    context = reopened.build_context()
    texts = [
        m.content if isinstance(m, UserMessage) else m.content[0].text for m in context.messages
    ]
    assert "持久摘要" in texts[0]
    assert texts[1:] == ["新问", "新答"]

    # 压缩后继续追加：新消息接在 compaction 条目后
    reopened.append_message(UserMessage(content="压缩后提问"))
    context2 = reopened.build_context()
    assert context2.messages[-1].content == "压缩后提问"


async def test_double_compaction(tmp_path: Path) -> None:
    """重复压缩：第二次以已压缩上下文为基准（旧前缀不再重复摘要）。"""
    session = SessionManager.create(str(tmp_path), home=tmp_path)
    session.append_message(UserMessage(content="第一轮问"))
    session.append_message(make_assistant_with_usage("第一轮答", Usage(input=1_000)))
    session.append_message(UserMessage(content="保留问"))
    session.append_message(make_assistant_with_usage("保留答", Usage(input=92_000)))

    first = await compact_session(
        session,
        context_window=100_000,
        model=big_model(),
        stream_fn=make_summary_stream("第一份摘要"),
    )
    assert first is not None

    # 再追加上下文（新末条助手用量反映增长后的规模，再次超阈值）
    session.append_message(UserMessage(content="第二轮问"))
    session.append_message(make_assistant_with_usage("第二轮答", Usage(input=95_000)))

    captured: list[tuple[Any, LlmContext, StreamOptions]] = []
    second = await compact_session(
        session,
        context_window=100_000,
        model=big_model(),
        stream_fn=make_summary_stream("第二份摘要", captured),
    )
    assert second is not None

    # 第二次摘要的输入不再含「第一轮问」（旧前缀已排除），
    # 而含上一份摘要（更新式摘要）
    request_texts = [
        m.content if isinstance(m, UserMessage) else m.content[0].text
        for m in captured[0][1].messages
    ]
    joined_request = "\n".join(request_texts)
    assert "第一轮问" not in joined_request
    assert "第一份摘要" in joined_request

    # 上下文：最新摘要 + 保留尾（第二轮问答在内）
    context = session.build_context()
    texts = [
        m.content if isinstance(m, UserMessage) else m.content[0].text for m in context.messages
    ]
    assert "第二份摘要" in texts[0]
    assert texts[-2:] == ["第二轮问", "第二轮答"]


def test_summary_to_user_message_format() -> None:
    """摘要消息格式：标记前缀 + token 量。"""
    message = summary_to_user_message("内容", 12345)
    assert isinstance(message, UserMessage)
    assert "[Compacted conversation summary]" in message.content
    assert "12345" in message.content
    assert "内容" in message.content
