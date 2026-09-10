"""Claude 协议 provider（anthropic messages 流式）。

迁移自 pi packages/ai/src/api/anthropic-messages.ts 的 stream 主路径，
按 v1 裁剪：不做 OAuth 专有名转换、cache_control、fine-grained tool
streaming beta、deferred、服务端回退。

thinking 级别 → budget 映射迁移自 pi api/simple-options.ts：
DEFAULT_THINKING_BUDGETS / MIN_ANSWER_TOKENS。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

import anthropic
from anthropic import AsyncAnthropic, AsyncStream
from anthropic.types import RawMessageStreamEvent

from mimcode.provider._json import partial_json_loads
from mimcode.provider.base import (
    Provider,
    StreamProtocolError,
    identity_index,
    make_partial_message,
    terminal_error_event,
)
from mimcode.types import (
    AssistantMessage,
    AssistantStreamEvent,
    StreamDone,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamThinkingDelta,
    StreamThinkingEnd,
    StreamThinkingStart,
    StreamToolCallDelta,
    StreamToolCallEnd,
    StreamToolCallStart,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    Usage,
)
from mimcode.types.context import LlmContext, ModelInfo, StreamOptions, ThinkingLevel
from mimcode.types.messages import ContentBlock, ToolResultMessage, UserMessage

# --- thinking 预算（对齐 pi simple-options.ts L57-62 / L55） -----------------

MIN_ANSWER_TOKENS = 1024
"""思考预算与输出共享上限时，为回答保留的最少 token。"""

DEFAULT_THINKING_BUDGETS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}
"""各级别默认思考 token 预算（对齐 pi DEFAULT_THINKING_BUDGETS）。"""

_ANTHROPIC_MIN_THINKING_BUDGET = 1024
"""Anthropic API 对 budget_tokens 的下限；不足时不启用 thinking。"""


def thinking_budget_for_level(level: ThinkingLevel) -> int:
    """级别 → 默认预算 token 数。"""
    return DEFAULT_THINKING_BUDGETS[level]


def _map_stop_reason(
    reason: str, stop_details: dict[str, Any]
) -> tuple[Literal["stop", "length", "toolUse", "error"], str | None]:
    """Anthropic stop_reason → (stopReason, errorMessage)（对齐 pi mapStopReason）。

    未知取值抛协议错误（对齐 pi 的 throw → 流内 error 事件）。
    """
    if reason in ("end_turn", "pause_turn", "stop_sequence"):
        return ("stop", None)
    if reason == "max_tokens":
        return ("length", None)
    if reason == "tool_use":
        return ("toolUse", None)
    if reason == "refusal":
        return (
            "error",
            stop_details.get("explanation") or "The model refused to complete the request",
        )
    if reason == "sensitive":
        return ("error", "Provider stopped with: sensitive")
    raise StreamProtocolError(f"Unhandled stop reason: {reason}")


async def translate_anthropic_events(
    events: AsyncIterator[dict[str, Any]],
    output: AssistantMessage,
    signal: asyncio.Event | None = None,
) -> AsyncIterator[AssistantStreamEvent]:
    """把 anthropic messages 流事件（dict 形态）翻译为助手流事件。

    事件结构对齐 pi：content_block_start/delta/stop 携带块级 index，
    message_start/message_delta 携带用量与 stop reason。

    Args:
        events: dict 形态的事件序列（真实 SDK 的 model_dump 或 faux fixture）。
        output: 终态载体（就地更新；对齐 pi 的 partial 引用语义）。
        signal: 中止信号；置位后在事件间隙抛协议错误（归一为 aborted）。

    Yields:
        AssistantStreamEvent，以 StreamDone 结束。

    Raises:
        StreamProtocolError: 流违约（零事件 / 缺 stop reason / 未知取值 / 中止）。
    """
    started = False
    blocks_by_index: dict[int, ContentBlock] = {}
    partial_json: dict[int, str] = {}

    async for event in events:
        if signal is not None and signal.is_set():
            raise StreamProtocolError("Request was aborted")
        if not started:
            started = True
            yield StreamStart(partial=output)

        event_type = event.get("type")

        if event_type == "message_start":
            message = event.get("message") or {}
            usage = message.get("usage") or {}
            # 初始用量（即使中途中止也保住 input 计数，对齐 pi 注释语义）
            output.usage = Usage(
                input=usage.get("input_tokens") or 0,
                output=usage.get("output_tokens") or 0,
                cache_read=usage.get("cache_read_input_tokens") or 0,
                cache_write=usage.get("cache_creation_input_tokens") or 0,
            )

        elif event_type == "content_block_start":
            index = event.get("index", 0)
            block = event.get("content_block") or {}
            block_type = block.get("type")
            if block_type == "text":
                new_block: ContentBlock = TextBlock(text=block.get("text") or "")
                output.content.append(new_block)
                blocks_by_index[index] = new_block
                yield StreamTextStart(content_index=len(output.content) - 1, partial=output)
            elif block_type == "thinking":
                new_block = ThinkingBlock(
                    thinking=block.get("thinking") or "",
                    thinking_signature=block.get("signature") or "",
                )
                output.content.append(new_block)
                blocks_by_index[index] = new_block
                yield StreamThinkingStart(content_index=len(output.content) - 1, partial=output)
            elif block_type == "redacted_thinking":
                new_block = ThinkingBlock(
                    thinking="[Reasoning redacted]",
                    thinking_signature=block.get("data") or "",
                    redacted=True,
                )
                output.content.append(new_block)
                blocks_by_index[index] = new_block
                yield StreamThinkingStart(content_index=len(output.content) - 1, partial=output)
            elif block_type == "tool_use":
                new_block = ToolCallBlock(
                    id=block.get("id") or "",
                    name=block.get("name") or "",
                    arguments={},
                )
                output.content.append(new_block)
                blocks_by_index[index] = new_block
                partial_json[index] = ""
                yield StreamToolCallStart(content_index=len(output.content) - 1, partial=output)

        elif event_type == "content_block_delta":
            index = event.get("index")
            block = blocks_by_index.get(index) if index is not None else None
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if block is not None:
                content_index = identity_index(output.content, block)
                if delta_type == "text_delta" and isinstance(block, TextBlock):
                    text = delta.get("text") or ""
                    block.text += text
                    yield StreamTextDelta(content_index=content_index, delta=text, partial=output)
                elif delta_type == "thinking_delta" and isinstance(block, ThinkingBlock):
                    thinking = delta.get("thinking") or ""
                    block.thinking += thinking
                    yield StreamThinkingDelta(
                        content_index=content_index, delta=thinking, partial=output
                    )
                elif delta_type == "input_json_delta" and isinstance(block, ToolCallBlock):
                    fragment = delta.get("partial_json") or ""
                    partial_json[index] = partial_json.get(index, "") + fragment
                    block.arguments = partial_json_loads(partial_json[index])
                    yield StreamToolCallDelta(
                        content_index=content_index, delta=fragment, partial=output
                    )
                elif delta_type == "signature_delta" and isinstance(block, ThinkingBlock):
                    block.thinking_signature = (block.thinking_signature or "") + (
                        delta.get("signature") or ""
                    )

        elif event_type == "content_block_stop":
            index = event.get("index")
            block = blocks_by_index.get(index) if index is not None else None
            if block is not None:
                content_index = identity_index(output.content, block)
                if isinstance(block, TextBlock):
                    yield StreamTextEnd(
                        content_index=content_index, content=block.text, partial=output
                    )
                elif isinstance(block, ThinkingBlock):
                    yield StreamThinkingEnd(
                        content_index=content_index, content=block.thinking, partial=output
                    )
                else:
                    block.arguments = partial_json_loads(partial_json.get(index))
                    yield StreamToolCallEnd(
                        content_index=content_index, tool_call=block, partial=output
                    )

        elif event_type == "message_delta":
            delta = event.get("delta") or {}
            if delta.get("stop_reason"):
                stop_reason, error_message = _map_stop_reason(
                    delta["stop_reason"], delta.get("stop_details") or {}
                )
                output.stop_reason = stop_reason
                if error_message:
                    output.error_message = error_message
            # 只更新出现的字段（对齐 pi：代理可能省略 input_tokens）
            usage = event.get("usage") or {}
            if usage.get("input_tokens") is not None:
                output.usage.input = usage["input_tokens"]
            if usage.get("output_tokens") is not None:
                output.usage.output = usage["output_tokens"]
            if usage.get("cache_read_input_tokens") is not None:
                output.usage.cache_read = usage["cache_read_input_tokens"]
            if usage.get("cache_creation_input_tokens") is not None:
                output.usage.cache_write = usage["cache_creation_input_tokens"]
            # 思考 token 计入 output_tokens 的子集（对齐 pi L755-761）
            usage_details = usage.get("output_tokens_details") or {}
            if usage_details.get("thinking_tokens") is not None:
                output.usage.reasoning = usage_details["thinking_tokens"]

        # message_stop / ping 等事件无需处理

    if not started:
        raise StreamProtocolError("Stream ended without any events")
    if signal is not None and signal.is_set():
        raise StreamProtocolError("Request was aborted")
    if output.stop_reason == "pending":
        raise StreamProtocolError("Anthropic stream ended without a stop reason")
    if output.stop_reason in ("aborted", "error"):
        raise StreamProtocolError(output.error_message or "An unknown error occurred")
    if output.stop_reason in ("stop", "length", "toolUse"):
        yield StreamDone(reason=output.stop_reason, message=output)
    else:  # pragma: no cover - 上述分支已覆盖全部可终态
        raise StreamProtocolError(f"Unexpected terminal stop reason: {output.stop_reason}")


def _image_source(block: Any) -> dict[str, Any]:
    """ImageBlock → anthropic source 参数。"""
    return {"type": "base64", "media_type": block.mime_type, "data": block.data}


def _convert_user_message(message: UserMessage) -> dict[str, Any] | None:
    """用户消息 → MessageParam（空文本剔除，对齐 pi）。"""
    if isinstance(message.content, str):
        if not message.content.strip():
            return None
        return {"role": "user", "content": message.content}
    blocks: list[dict[str, Any]] = []
    for block in message.content:
        if block.type == "text":
            if block.text.strip():
                blocks.append({"type": "text", "text": block.text})
        else:
            blocks.append({"type": "image", "source": _image_source(block)})
    if not blocks:
        return None
    return {"role": "user", "content": blocks}


def _convert_assistant_message(message: AssistantMessage) -> dict[str, Any] | None:
    """助手消息 → MessageParam。

    thinking 块回传规则（对齐 pi L1217-1251）：
    - redacted → redacted_thinking（不透明载荷在签名槽）
    - 无签名的非空 thinking → 降级为 text（兼容端点可能给空签名）
    - 空且无签名 → 跳过
    """
    blocks: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            if block.text.strip():
                blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingBlock):
            if block.redacted:
                blocks.append({"type": "redacted_thinking", "data": block.thinking_signature or ""})
                continue
            has_signature = bool((block.thinking_signature or "").strip())
            if not block.thinking.strip() and not has_signature:
                continue
            if has_signature:
                blocks.append(
                    {
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": block.thinking_signature,
                    }
                )
            else:
                blocks.append({"type": "text", "text": block.thinking})
        else:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments or {},
                }
            )
    if not blocks:
        return None
    return {"role": "assistant", "content": blocks}


def _convert_tool_result_run(messages: list[Any], start: int) -> tuple[dict[str, Any] | None, int]:
    """把连续的 toolResult 消息组合为单条 user 消息（对齐 pi L1266-1291）。

    Anthropic 协议要求 tool_result 块位于 user 消息内；
    相邻分组合并兼容 z.ai 等端点。图片块作为 tool_result 后的兄弟块。

    Returns:
        (转换后的消息或 None, 消费到的下一条消息下标)。
    """
    tool_blocks: list[dict[str, Any]] = []
    sibling: list[dict[str, Any]] = []
    index = start
    while index < len(messages) and isinstance(messages[index], ToolResultMessage):
        message = messages[index]
        content: list[dict[str, Any]] = []
        for block in message.content:
            if block.type == "text":
                if block.text.strip():
                    content.append({"type": "text", "text": block.text})
            else:
                sibling.append({"type": "image", "source": _image_source(block)})
        entry: dict[str, Any] = {"type": "tool_result", "tool_use_id": message.tool_call_id}
        if content:
            entry["content"] = content
        if message.is_error:
            entry["is_error"] = True
        tool_blocks.append(entry)
        index += 1
    if not tool_blocks:
        return None, start
    return {"role": "user", "content": [*tool_blocks, *sibling]}, index


def convert_anthropic_messages(context: LlmContext) -> list[dict[str, Any]]:
    """LlmContext.messages → anthropic messages 参数。"""
    params: list[dict[str, Any]] = []
    index = 0
    while index < len(context.messages):
        message = context.messages[index]
        if isinstance(message, ToolResultMessage):
            converted, next_index = _convert_tool_result_run(context.messages, index)
            if converted is not None:
                params.append(converted)
            index = next_index
            continue
        if isinstance(message, UserMessage):
            converted = _convert_user_message(message)
        else:
            converted = _convert_assistant_message(message)
        if converted is not None:
            params.append(converted)
        index += 1
    return params


class AnthropicProtocolProvider(Provider):
    """Claude 协议 provider：官方 SDK + 自定义 base_url。"""

    def __init__(
        self,
        provider_id: str = "anthropic",
        name: str = "Claude 协议",
        *,
        base_url: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        models: list[ModelInfo] | None = None,
    ) -> None:
        super().__init__(
            provider_id, name, base_url=base_url, api_key_env=api_key_env, models=models
        )

    def _thinking_param(
        self, model: ModelInfo, options: StreamOptions, max_tokens: int
    ) -> dict[str, Any] | None:
        """计算 thinking 参数（对齐 pi adjustMaxTokensForThinking 的 v1 简化）。

        预算 = min(级别默认预算, max_tokens - MIN_ANSWER_TOKENS)；
        低于 Anthropic 下限（1024）时不启用。
        """
        if not options.thinking_level or not model.supports_thinking:
            return None
        budget = min(
            thinking_budget_for_level(options.thinking_level),
            max(0, max_tokens - MIN_ANSWER_TOKENS),
        )
        if budget < _ANTHROPIC_MIN_THINKING_BUDGET:
            return None
        return {"type": "enabled", "budget_tokens": budget}

    def _build_payload(
        self, model: ModelInfo, context: LlmContext, options: StreamOptions
    ) -> dict[str, Any]:
        """组装 messages.create 的参数（含 stream）。"""
        max_tokens = options.max_tokens or model.max_output_tokens
        payload: dict[str, Any] = {
            "model": model.id,
            "messages": convert_anthropic_messages(context),
            "max_tokens": max_tokens,
            "stream": True,
        }
        if context.system_prompt and context.system_prompt.strip():
            payload["system"] = context.system_prompt
        if context.tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters or {"type": "object", "properties": {}},
                }
                for tool in context.tools
            ]
        thinking = self._thinking_param(model, options, max_tokens)
        if thinking is not None:
            payload["thinking"] = thinking
        return payload

    async def _raw_events(
        self, payload: dict[str, Any], options: StreamOptions
    ) -> AsyncIterator[dict[str, Any]]:
        """真实 SDK 请求：事件以 dict 产出（与 faux fixture 同构）。"""
        client = AsyncAnthropic(
            api_key=self.resolve_api_key(options),
            base_url=self.base_url,
            max_retries=0,
        )
        # payload 含 stream=True；**kwargs 解包无法命中流式重载，显式收窄返回类型
        stream = cast(
            "AsyncStream[RawMessageStreamEvent]",
            await client.messages.create(**payload),
        )
        async for event in stream:
            yield event.model_dump()

    async def stream(
        self,
        model: ModelInfo,
        context: LlmContext,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantStreamEvent]:
        """流式调用（契约见 base 模块 docstring）。"""
        opts = options or StreamOptions()
        output = make_partial_message(model)
        try:
            payload = self._build_payload(model, context, opts)
            events = translate_anthropic_events(
                self._raw_events(payload, opts), output, opts.signal
            )
            async for event in events:
                yield event
        except (anthropic.AnthropicError, StreamProtocolError) as exc:
            yield terminal_error_event(output, exc, opts.signal)


__all__ = [
    "AnthropicProtocolProvider",
    "convert_anthropic_messages",
    "translate_anthropic_events",
    "DEFAULT_THINKING_BUDGETS",
    "MIN_ANSWER_TOKENS",
    "thinking_budget_for_level",
]
