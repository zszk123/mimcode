"""Provider 抽象层（对齐 pi packages/ai/src/models.ts 的 Provider 接口子集）。

「协议即 provider」：OpenAI 协议与 Claude 协议各一个实现类，
通过 base_url 覆盖吃下全部兼容端点。

流契约（对齐 pi 的 StreamFn，packages/agent/src/types.ts L19-32）：
- stream() 是异步生成器，产出 AssistantStreamEvent；
- 请求/模型/运行期失败不抛异常，编码为 StreamError 事件
  （终态消息 stopReason 为 error 或 aborted + errorMessage）。
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

from mimcode.types import AssistantMessage, AssistantStreamEvent, StreamError

if TYPE_CHECKING:
    from mimcode.types import ModelInfo
    from mimcode.types.context import LlmContext, StreamOptions


class StreamProtocolError(Exception):
    """流协议违约（如流结束但缺失 stop reason、未知 stop reason）。

    对齐 pi：这类错误在 TS 实现里以 throw 表达，被外层捕获后
    归一为流内 error 事件。mimcode 以本类型显式表达同一语义。
    """


def identity_index(blocks: list[Any], block: Any) -> int:
    """按对象身份（is）查找块下标。

    pydantic 的 list.index 依赖 __eq__（值相等），内容相同的块会误命中；
    事件 contentIndex 语义要求引用级定位（对齐 pi 的 indexOf）。
    """
    for index, candidate in enumerate(blocks):
        if candidate is block:
            return index
    return -1


def make_partial_message(model: ModelInfo) -> AssistantMessage:
    """构造流式起点消息（对齐 pi 的初始 output：stopReason=pending）。"""
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="pending",
    )


def terminal_error_event(
    output: AssistantMessage,
    exc: BaseException,
    signal: asyncio.Event | None,
) -> StreamError:
    """把异常归一为终态错误事件（对齐 pi 各协议 stream 的 catch 块）。

    signal 已置位 → aborted（用户主动中止）；否则 → error。
    """
    reason: Literal["aborted", "error"] = (
        "aborted" if signal is not None and signal.is_set() else "error"
    )
    output.stop_reason = reason
    output.error_message = str(exc) or type(exc).__name__
    return StreamError(reason=reason, error=output)


class Provider(ABC):
    """协议 provider 抽象基类。

    与 pi Provider 接口的对应：
    - id/name/base_url → 元数据（T4 目录使用）
    - auth → resolve_api_key（v1 仅 API key：选项 > 环境变量；T4 接入配置文件）
    - getModels → get_models（静态目录；动态刷新在 T4）
    - stream → stream（异步生成器，契约见模块 docstring）
    """

    def __init__(
        self,
        provider_id: str,
        name: str,
        *,
        base_url: str | None = None,
        api_key_env: str | None = None,
        models: list[ModelInfo] | None = None,
    ) -> None:
        self.id = provider_id
        self.name = name
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._models = list(models or [])

    def get_models(self) -> list[ModelInfo]:
        """当前已知模型目录（静态；动态刷新在 T4）。"""
        return list(self._models)

    def resolve_api_key(self, options: StreamOptions | None) -> str | None:
        """解析 API key：调用方显式传入 > 环境变量。

        T4 会在此链路接入配置文件来源。
        """
        if options is not None and options.api_key:
            return options.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or None
        return None

    @abstractmethod
    def stream(
        self,
        model: ModelInfo,
        context: LlmContext,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantStreamEvent]:
        """流式调用（契约见模块 docstring）。

        Args:
            model: 目标模型。
            context: LLM 上下文（系统提示/消息/工具）。
            options: 调用选项（key/中止信号/思考级别/输出上限）。

        Yields:
            AssistantStreamEvent 序列，以 StreamDone 或 StreamError 终结。
        """
