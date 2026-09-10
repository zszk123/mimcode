"""OpenAI 协议 provider（chat.completions 流式）。

迁移自 pi packages/ai/src/api/openai-completions.ts 的 stream 主路径，
按 v1 裁剪：不做 OpenRouter 专有 reasoning_details 重放、cache_control、
grammars/自定义工具输入、deferred。

结构：
- ``_build_payload``：LlmContext → chat.completions 请求参数
- ``_raw_chunks``：SDK 流 → dict chunk（与 faux fixture 同构）
- ``translate_openai_chunks``：dict chunk → AssistantStreamEvent（纯翻译，
  真实与 faux 共用同一条代码路径）
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

import openai
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk

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
    StreamStart,
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
from mimcode.types.context import LlmContext, ModelInfo, StreamOptions
from mimcode.types.messages import UserMessage

# 兼容端点的推理字段（对齐 pi openai-completions.ts L586）：
# llama.cpp 用 reasoning_content，其他端点用 reasoning/reasoning_text；
# 取第一个非空者避免重复（chutes.ai 会同时返回两个字段）
_REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


def _first_not_none(*values: Any) -> Any:
    """返回第一个非 None 的值（对齐 TS 的 ?? 链）。"""
    return next((value for value in values if value is not None), None)


def _parse_chunk_usage(raw: dict[str, Any]) -> Usage:
    """解析 usage chunk（对齐 pi parseChunkUsage 的字段归一）。

    cache-read 的字段位置各家不一：OpenAI/OpenRouter 用
    prompt_tokens_details.cached_tokens，DeepSeek 用 prompt_cache_hit_tokens，
    Kimi 用顶层 usage.cached_tokens。
    """
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    return Usage(
        input=raw.get("prompt_tokens") or 0,
        output=raw.get("completion_tokens") or 0,
        cache_read=_first_not_none(
            prompt_details.get("cached_tokens"),
            raw.get("prompt_cache_hit_tokens"),
            raw.get("cached_tokens"),
        )
        or 0,
        cache_write=prompt_details.get("cache_write_tokens") or 0,
        reasoning=completion_details.get("reasoning_tokens") or 0,
    )


def _map_stop_reason(
    reason: str,
) -> tuple[Literal["stop", "length", "toolUse", "error"], str | None]:
    """finish_reason → (stopReason, errorMessage)（对齐 pi mapStopReason）。"""
    if reason in ("stop", "end"):
        return ("stop", None)
    if reason == "length":
        return ("length", None)
    if reason in ("function_call", "tool_calls"):
        return ("toolUse", None)
    if reason == "content_filter":
        return ("error", "Provider finish_reason: content_filter")
    if reason == "network_error":
        return ("error", "Provider finish_reason: network_error")
    return ("error", f"Provider finish_reason: {reason}")


async def translate_openai_chunks(
    chunks: AsyncIterator[dict[str, Any]],
    output: AssistantMessage,
    signal: asyncio.Event | None = None,
) -> AsyncIterator[AssistantStreamEvent]:
    """把 chat.completions 流式 chunk（dict 形态）翻译为助手流事件。

    状态模型对齐 pi：单文本块、单思考块（惰性创建），
    工具调用块按流内 index 归并、参数增量累积后尽力解析。

    Args:
        chunks: dict 形态的 chunk 序列（真实 SDK 的 model_dump 或 faux fixture）。
        output: 终态载体（就地更新；对齐 pi 的 partial 引用语义）。
        signal: 中止信号；置位后在 chunk 间隙抛协议错误（归一为 aborted）。

    Yields:
        AssistantStreamEvent，以 StreamDone 结束。

    Raises:
        StreamProtocolError: 流违约（零 chunk / error stopReason / 中止）。
    """
    started = False
    text_block: TextBlock | None = None
    thinking_block: ThinkingBlock | None = None
    tool_by_index: dict[int, ToolCallBlock] = {}
    partial_args: dict[int, str] = {}
    has_finish_reason = False

    async for chunk in chunks:
        if signal is not None and signal.is_set():
            raise StreamProtocolError("Request was aborted")
        if not started:
            started = True
            yield StreamStart(partial=output)

        if chunk.get("usage"):
            output.usage = _parse_chunk_usage(chunk["usage"])

        choices = chunk.get("choices") or []
        choice = choices[0] if choices else None
        if choice is None:
            continue

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            stop_reason, error_message = _map_stop_reason(finish_reason)
            output.stop_reason = stop_reason
            if error_message:
                output.error_message = error_message
            has_finish_reason = True

        delta = choice.get("delta") or {}

        # 文本增量
        content = delta.get("content")
        if isinstance(content, str) and content:
            if text_block is None:
                text_block = TextBlock(text="")
                output.content.append(text_block)
                yield StreamTextStart(
                    content_index=identity_index(output.content, text_block),
                    partial=output,
                )
            text_block.text += content
            yield StreamTextDelta(
                content_index=identity_index(output.content, text_block),
                delta=content,
                partial=output,
            )

        # 推理增量：兼容端点的 reasoning_content / reasoning / reasoning_text
        reasoning_delta = None
        reasoning_field: str | None = None
        for field in _REASONING_FIELDS:
            value = delta.get(field)
            if isinstance(value, str) and value:
                reasoning_delta = value
                reasoning_field = field
                break
        if reasoning_delta is not None and reasoning_field is not None:
            if thinking_block is None:
                # 签名槽存来源字段名（对齐 pi：用于回放侧辨识，v1 仅透传）
                thinking_block = ThinkingBlock(thinking="", thinking_signature=reasoning_field)
                output.content.append(thinking_block)
                yield StreamThinkingStart(
                    content_index=identity_index(output.content, thinking_block),
                    partial=output,
                )
            thinking_block.thinking += reasoning_delta
            yield StreamThinkingDelta(
                content_index=identity_index(output.content, thinking_block),
                delta=reasoning_delta,
                partial=output,
            )

        # 工具调用增量（参数 JSON 跨 chunk 拆分，按 index 归并）
        for tool_call in delta.get("tool_calls") or []:
            index = tool_call.get("index")
            if not isinstance(index, int):
                raise StreamProtocolError("tool_calls delta missing index")
            block = tool_by_index.get(index)
            if block is None:
                block = ToolCallBlock(
                    id=tool_call.get("id") or "",
                    name=(tool_call.get("function") or {}).get("name") or "",
                    arguments={},
                )
                tool_by_index[index] = block
                output.content.append(block)
                yield StreamToolCallStart(
                    content_index=identity_index(output.content, block),
                    partial=output,
                )
            if not block.id and tool_call.get("id"):
                block.id = tool_call["id"]
            function_delta = tool_call.get("function") or {}
            name = function_delta.get("name")
            if not block.name and name:
                block.name = name
            arguments_delta = function_delta.get("arguments")
            if isinstance(arguments_delta, str) and arguments_delta:
                partial_args[index] = partial_args.get(index, "") + arguments_delta
                block.arguments = partial_json_loads(partial_args[index])
            yield StreamToolCallDelta(
                content_index=identity_index(output.content, block),
                delta=arguments_delta if isinstance(arguments_delta, str) else "",
                partial=output,
            )

    if not started:
        raise StreamProtocolError("Stream ended without any chunks")

    # 终结各内容块（对齐 pi 的 finishBlock 循环）
    for final_block in output.content:
        content_index = identity_index(output.content, final_block)
        if isinstance(final_block, TextBlock):
            yield StreamTextEnd(
                content_index=content_index, content=final_block.text, partial=output
            )
        elif isinstance(final_block, ThinkingBlock):
            yield StreamThinkingEnd(
                content_index=content_index, content=final_block.thinking, partial=output
            )
        elif isinstance(final_block, ToolCallBlock):
            yield StreamToolCallEnd(
                content_index=content_index, tool_call=final_block, partial=output
            )

    if signal is not None and signal.is_set():
        raise StreamProtocolError("Request was aborted")
    if not has_finish_reason:
        # 宽松推断（对齐 pi 的 supportsFinishReason=false 分支）：
        # 协议即 provider 面向大量兼容端点，缺 finish_reason 时按内容推断
        output.stop_reason = (
            "toolUse"
            if any(isinstance(block, ToolCallBlock) for block in output.content)
            else "stop"
        )
    if output.stop_reason == "error":
        raise StreamProtocolError(output.error_message or "Provider returned an error stop reason")
    if output.stop_reason in ("stop", "length", "toolUse"):
        yield StreamDone(reason=output.stop_reason, message=output)
    else:  # pragma: no cover - 上述分支已覆盖全部可终态
        raise StreamProtocolError(f"Unexpected terminal stop reason: {output.stop_reason}")


def convert_openai_messages(context: LlmContext) -> list[dict[str, Any]]:
    """LlmContext.messages → chat.completions messages 参数。

    对齐 pi convertMessages 的 v1 子集：
    - 纯文本用户消息保持字符串形态；块形态转 content parts
    - 助手消息：thinking 块不回传（OpenAI 协议无通用回放通道）
    - 工具结果 → role:"tool"（携带 tool_call_id）
    """
    messages: list[dict[str, Any]] = []
    if context.system_prompt and context.system_prompt.strip():
        messages.append({"role": "system", "content": context.system_prompt})

    for message in context.messages:
        if isinstance(message, UserMessage):
            if isinstance(message.content, str):
                if message.content.strip():
                    messages.append({"role": "user", "content": message.content})
            else:
                parts: list[dict[str, Any]] = []
                for block in message.content:
                    if block.type == "text":
                        if block.text.strip():
                            parts.append({"type": "text", "text": block.text})
                    else:
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                            }
                        )
                if parts:
                    messages.append({"role": "user", "content": parts})
        elif isinstance(message, AssistantMessage):
            text = "".join(block.text for block in message.content if isinstance(block, TextBlock))
            tool_calls = [
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.arguments, ensure_ascii=False),
                    },
                }
                for block in message.content
                if isinstance(block, ToolCallBlock)
            ]
            entry: dict[str, Any] = {"role": "assistant"}
            if text:
                entry["content"] = text
            if tool_calls:
                entry["tool_calls"] = tool_calls
            if text or tool_calls:
                messages.append(entry)
        else:
            # ToolResultMessage：图片块在 OpenAI tool 消息中无法表达（v1 裁剪）
            text = "".join(block.text for block in message.content if isinstance(block, TextBlock))
            messages.append({"role": "tool", "tool_call_id": message.tool_call_id, "content": text})
    return messages


class OpenAIProtocolProvider(Provider):
    """OpenAI 协议 provider：官方 SDK + 自定义 base_url。"""

    def __init__(
        self,
        provider_id: str = "openai",
        name: str = "OpenAI 协议",
        *,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        models: list[ModelInfo] | None = None,
    ) -> None:
        super().__init__(
            provider_id, name, base_url=base_url, api_key_env=api_key_env, models=models
        )

    def _build_payload(
        self, model: ModelInfo, context: LlmContext, options: StreamOptions
    ) -> dict[str, Any]:
        """组装 chat.completions.create 的参数（含 stream/stream_options）。"""
        payload: dict[str, Any] = {
            "model": model.id,
            "messages": convert_openai_messages(context),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if context.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters or {"type": "object", "properties": {}},
                    },
                }
                for tool in context.tools
            ]
        if options.max_tokens is not None:
            payload["max_tokens"] = options.max_tokens
        if options.thinking_level and model.supports_thinking:
            # OpenAI reasoning_effort 枚举与 ThinkingLevel 取值一致
            payload["reasoning_effort"] = options.thinking_level
        return payload

    async def _raw_chunks(
        self, payload: dict[str, Any], options: StreamOptions
    ) -> AsyncIterator[dict[str, Any]]:
        """真实 SDK 请求：chunk 以 dict 产出（与 faux fixture 同构）。

        客户端构造缺 key 时抛 openai.OpenAIError → 由 stream 守卫归一。
        """
        client = AsyncOpenAI(
            api_key=self.resolve_api_key(options),
            base_url=self.base_url,
            max_retries=0,
        )
        # payload 含 stream=True；**kwargs 解包无法命中流式重载，显式收窄返回类型
        stream = cast(
            "AsyncStream[ChatCompletionChunk]",
            await client.chat.completions.create(**payload),
        )
        try:
            async for chunk in stream:
                yield chunk.model_dump()
        finally:
            await stream.close()
            await client.close()

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
            events = translate_openai_chunks(self._raw_chunks(payload, opts), output, opts.signal)
            async for event in events:
                yield event
        except (openai.OpenAIError, StreamProtocolError) as exc:
            yield terminal_error_event(output, exc, opts.signal)


__all__ = ["OpenAIProtocolProvider", "convert_openai_messages", "translate_openai_chunks"]
