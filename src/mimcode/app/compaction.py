"""上下文压缩（迁移自 pi packages/agent/src/harness/compaction/compaction.ts 的主路径）。

流程（对齐 pi）：
1. 估算上下文 token：优先用最后一条有效助手消息的 provider 用量，
   其后消息按字符启发式估算（估算值 = usage + trailing）
2. 阈值判断：``context_tokens > context_window - reserve_tokens``
3. 裁剪点选择：从尾部倒序累计 token 至 keep_recent_tokens，
   落到其后的首个用户消息边界（消息级切割，不拆半轮）
4. 摘要生成：独立流式请求（系统提示 + 历史 + 指令），失败返回 None
   （下次触发时重试，对齐 pi 的容错语义）
5. 会话落地：compaction 条目（summary / first_kept_entry_id / tokens_before），
   session.build_context 感知该条目后只保留 [摘要] + [保留尾] + [其后消息]

默认阈值（对齐 pi DEFAULT_COMPACTION_SETTINGS）：
- reserve_tokens = 16384（为摘要提示与输出预留）
- keep_recent_tokens = 20000（压缩后保留的近期上下文）
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING

from mimcode.types import (
    AssistantMessage,
    AssistantStreamEvent,
    LlmContext,
    StreamOptions,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    Usage,
    UserMessage,
)

if TYPE_CHECKING:
    from mimcode.app.session import SessionManager
    from mimcode.types import AgentMessage, ModelInfo

# ---------------------------------------------------------------------------
# 设置与常量（checklist：reserve_tokens=16384 / keep_recent_tokens=20000）
# ---------------------------------------------------------------------------

RESERVE_TOKENS = 16384
"""为摘要提示与输出预留的 token（对齐 pi reserveTokens）。"""

KEEP_RECENT_TOKENS = 20000
"""压缩后保留的近期上下文 token（对齐 pi keepRecentTokens）。"""

ESTIMATED_IMAGE_CHARS = 4800
"""单张图片的估算字符数（对齐 pi ESTIMATED_IMAGE_CHARS）。"""

_CHARS_PER_TOKEN = 4
"""保守字符/token 比（对齐 pi）。"""


@dataclasses.dataclass(frozen=True)
class CompactionSettings:
    """压缩设置（对齐 pi CompactionSettings）。"""

    enabled: bool = True
    reserve_tokens: int = RESERVE_TOKENS
    keep_recent_tokens: int = KEEP_RECENT_TOKENS


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()
"""默认压缩设置（对齐 pi DEFAULT_COMPACTION_SETTINGS）。"""


# ---------------------------------------------------------------------------
# token 估算（对齐 pi estimateTokens / estimateContextTokens）
# ---------------------------------------------------------------------------


def calculate_context_tokens(usage: Usage) -> int:
    """provider 用量 → 上下文 token 总数（四项之和，对齐 pi）。"""
    return usage.input + usage.output + usage.cache_read + usage.cache_write


def _valid_assistant_usage(message: AgentMessage) -> Usage | None:
    """助手消息的有效用量（aborted/error 的用量不可信，对齐 pi L168-179）。"""
    if isinstance(message, AssistantMessage):
        if (
            message.stop_reason not in ("aborted", "error")
            and calculate_context_tokens(message.usage) > 0
        ):
            return message.usage
    return None


def _text_and_image_chars(content: str | Sequence[object]) -> int:
    """文本/图片块内容的字符数（图片按 4800 计）。"""
    if isinstance(content, str):
        return len(content)
    chars = 0
    for block in content:
        if isinstance(block, TextBlock):
            chars += len(block.text)
        else:  # ImageBlock
            chars += ESTIMATED_IMAGE_CHARS
    return chars


def estimate_tokens(message: AgentMessage) -> int:
    """单条消息的保守字符启发式估算（chars/4，对齐 pi estimateTokens）。"""
    chars = 0
    if isinstance(message, UserMessage):
        chars = _text_and_image_chars(message.content)
    elif isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                chars += len(block.text)
            elif isinstance(block, ThinkingBlock):
                chars += len(block.thinking)
            elif isinstance(block, ToolCallBlock):
                chars += len(block.name) + len(str(block.arguments))
    else:  # ToolResultMessage
        chars = _text_and_image_chars(message.content)
    return -(-chars // _CHARS_PER_TOKEN)  # ceil


@dataclasses.dataclass(frozen=True)
class ContextUsageEstimate:
    """上下文 token 估算结果（对齐 pi ContextUsageEstimate）。"""

    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


def estimate_context_tokens(messages: Sequence[AgentMessage]) -> ContextUsageEstimate:
    """上下文 token 估算：provider 用量优先，尾部按字符估算。"""
    last_usage_index: int | None = None
    last_usage: Usage | None = None
    for index in range(len(messages) - 1, -1, -1):
        usage = _valid_assistant_usage(messages[index])
        if usage is not None:
            last_usage_index = index
            last_usage = usage
            break

    if last_usage is None or last_usage_index is None:
        estimated = sum(estimate_tokens(message) for message in messages)
        return ContextUsageEstimate(
            tokens=estimated, usage_tokens=0, trailing_tokens=estimated, last_usage_index=None
        )

    usage_tokens = calculate_context_tokens(last_usage)
    trailing_tokens = sum(estimate_tokens(message) for message in messages[last_usage_index + 1 :])
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing_tokens,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing_tokens,
        last_usage_index=last_usage_index,
    )


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    """阈值判断（对齐 pi shouldCompact）：超过 window - reserve 即触发。"""
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


# ---------------------------------------------------------------------------
# 裁剪点（对齐 pi findCutPoint 的消息级简化）
# ---------------------------------------------------------------------------


def find_cut_point(messages: Sequence[AgentMessage], keep_recent_tokens: int) -> int | None:
    """选择压缩裁剪点：保留约 keep_recent_tokens 的近期消息。

    语义（对齐 pi）：
    - 候选边界 = 用户消息起始位置（完整轮次，不拆半轮；
      切在 toolResult 之前会让模型看到孤立的助手轮）
    - 从尾部倒序累计估算 token，达到 keep_recent_tokens 后取其后的首个边界
    - 无有效边界（无用户消息可切）→ None（放弃本次压缩）

    Returns:
        保留段首条消息的下标；不可压缩返回 None。
    """
    boundaries = [
        index for index in range(1, len(messages)) if isinstance(messages[index], UserMessage)
    ]
    if not boundaries:
        return None

    accumulated = 0
    target_index = 0  # 无累计达标时取首个边界（保留最少，对齐 pi cutPoints[0] 兜底）
    for index in range(len(messages) - 1, 0, -1):
        accumulated += estimate_tokens(messages[index])
        if accumulated >= keep_recent_tokens:
            target_index = index
            break

    for boundary in boundaries:
        if boundary >= target_index:
            return boundary
    return boundaries[0]


# ---------------------------------------------------------------------------
# 摘要生成（对齐 pi SUMMARIZATION prompts + completeSimpleWithRetries）
# ---------------------------------------------------------------------------

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI assistant, then produce a structured summary following "
    "the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)
"""摘要请求的系统提示（对齐 pi SUMMARIZATION_SYSTEM_PROMPT）。"""

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. \
Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items \
if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""
"""摘要指令（对齐 pi SUMMARIZATION_PROMPT）。"""

COMPACTED_SUMMARY_PREFIX = "[Compacted conversation summary]"
"""摘要消息进入上下文时的前缀标记。"""


def summary_to_user_message(summary: str, tokens_before: int) -> UserMessage:
    """compaction 摘要 → 进入上下文的用户消息。"""
    content = (
        f"{COMPACTED_SUMMARY_PREFIX} "
        f"(earlier conversation compacted from ~{tokens_before} tokens):\n\n{summary}"
    )
    return UserMessage(content=content)


@dataclasses.dataclass(frozen=True)
class CompactResult:
    """压缩结果（对齐 pi CompactResult 的 v1 子集）。"""

    summary: str
    tokens_before: int
    history: list[AgentMessage]
    """被摘要替换的历史消息。"""

    retained_tail: list[AgentMessage]
    """保留的近期消息。"""


StreamFn = Callable[..., AsyncIterator[AssistantStreamEvent]]
"""流函数形态（Provider.stream 同构）。"""


async def generate_summary(
    history: Sequence[AgentMessage],
    model: ModelInfo,
    stream_fn: StreamFn,
    *,
    api_key: str | None = None,
) -> str | None:
    """独立摘要请求（对齐 pi completeSimpleWithRetries 的单次 v1 简化）。

    失败（流错误/中止/空文本）返回 None，调用方下次触发时重试。

    Args:
        history: 被摘要的消息序列。
        model: 摘要请求使用的模型。
        stream_fn: 流函数。
        api_key: 可选 key。

    Returns:
        摘要文本；失败返回 None。
    """
    context = LlmContext(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[*history, UserMessage(content=SUMMARIZATION_PROMPT)],
    )
    options = StreamOptions(api_key=api_key)

    final: AssistantMessage | None = None
    async for event in stream_fn(model, context, options):
        if event.type == "done":
            final = event.message
            break
        if event.type == "error":
            final = event.error
            break

    if final is None or final.stop_reason != "stop":
        return None
    text = "".join(block.text for block in final.content if isinstance(block, TextBlock))
    return text.strip() or None


# ---------------------------------------------------------------------------
# 会话级入口
# ---------------------------------------------------------------------------


async def compact_session(
    session: SessionManager,
    *,
    context_window: int,
    model: ModelInfo,
    stream_fn: StreamFn,
    settings: CompactionSettings = DEFAULT_COMPACTION_SETTINGS,
    api_key: str | None = None,
) -> str | None:
    """会话压缩入口：估算 → 阈值 → 裁剪 → 摘要 → 追加 compaction 条目。

    Args:
        session: 打开的会话。
        context_window: 当前模型的上下文窗口。
        model: 摘要请求使用的模型。
        stream_fn: 流函数。
        settings: 压缩设置。
        api_key: 可选 key。

    Returns:
        compaction 条目 id；未触发 / 不可切 / 摘要失败返回 None。
    """
    context = session.build_context()
    estimate = estimate_context_tokens(context.messages)
    if not should_compact(estimate.tokens, context_window, settings):
        return None

    # 裁剪基准 = 压缩感知后的上下文消息（重复压缩时已排除旧前缀）
    cut = find_cut_point(context.messages, settings.keep_recent_tokens)
    if cut is None:
        return None

    # 按对象身份把保留首消息映射回链条目（合成摘要消息无链身份，
    # 顺延到其后首个真实条目）
    chain = session.path_to_root()
    entry_ids_by_message: dict[int, str] = {
        id(entry.message): entry.id
        for entry in chain
        if entry.type == "message" and entry.message is not None
    }
    first_kept_entry_id: str | None = None
    for message in context.messages[cut:]:
        entry_id = entry_ids_by_message.get(id(message))
        if entry_id is not None:
            first_kept_entry_id = entry_id
            break

    history = context.messages[:cut]
    summary = await generate_summary(history, model, stream_fn, api_key=api_key)
    if summary is None:
        return None

    # compaction 条目挂当前叶（对齐 pi：整条链保留在树中，
    # 上下文裁剪由 build_context 的 first_kept 逻辑完成）
    return session.append_custom(
        "compaction",
        {
            "summary": summary,
            "first_kept_entry_id": first_kept_entry_id,
            "tokens_before": estimate.tokens,
        },
    )


def compact_result_from_entry(extra: dict[str, object]) -> CompactResult | None:
    """compaction 条目 extra → CompactResult（诊断/测试用）。"""
    summary = extra.get("summary")
    tokens_before = extra.get("tokens_before")
    if not isinstance(summary, str) or not isinstance(tokens_before, int):
        return None
    return CompactResult(
        summary=summary,
        tokens_before=tokens_before,
        history=[],
        retained_tail=[],
    )
