"""faux transport：JSON fixture 回放，供测试与 e2e（零真实网络）。

对标 pi 的 faux provider harness（packages/coding-agent/test/suite/harness.ts）。

设计：faux provider 继承真实协议 provider，只覆写「原始 chunk 源」——
payload 构建、流翻译、异常守卫全部复用真实代码路径。
fixture 以 dict chunk 驱动翻译器，与真实 SDK 的 model_dump 输出同构；
``error`` 字段模拟请求建立阶段的 SDK 异常（如连接失败），
用于验证流契约「失败编码为流内 error 事件」。

请求记录：``provider.requests`` 保存每次 stream 收到的完整 payload，
供断言模型切换/上下文注入等行为（T6/T14 使用）。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import anthropic
import httpx2
import openai
from pydantic import BaseModel, Field

from mimcode.provider.anthropic_protocol import AnthropicProtocolProvider
from mimcode.provider.openai_protocol import OpenAIProtocolProvider
from mimcode.types.context import ModelInfo, StreamOptions

_FAUX_OPENAI_URL = "https://faux.local/v1/chat/completions"
_FAUX_ANTHROPIC_URL = "https://faux.local/v1/messages"


class FauxFixture(BaseModel):
    """faux 回放数据。

    Attributes:
        protocol: 协议（openai / anthropic）。
        model: 回放侧宣称的模型 id。
        chunks: 原始 chunk/事件序列（dict，与 SDK model_dump 同构）。
        error: 错误类型标记（"connection"）；非空时请求建立阶段抛 SDK 连接异常。
    """

    protocol: Literal["openai", "anthropic"]
    model: str
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


def _fake_request(url: str) -> httpx2.Request:
    """构造抛异常用的最小请求对象（SDK 异常要求携带 request）。"""
    return httpx2.Request("POST", url)


def _raise_openai_connection_error(message: str) -> None:
    raise openai.APIConnectionError(request=_fake_request(_FAUX_OPENAI_URL), message=message)


def _raise_anthropic_connection_error(message: str) -> None:
    raise anthropic.APIConnectionError(request=_fake_request(_FAUX_ANTHROPIC_URL), message=message)


def load_fixture(path: Path | str) -> FauxFixture:
    """从 JSON 文件加载 fixture。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FauxFixture.model_validate(data)


class FauxOpenAIProvider(OpenAIProtocolProvider):
    """OpenAI 协议的 faux provider（复用真实翻译与守卫路径）。"""

    def __init__(self, fixture: FauxFixture) -> None:
        self.fixture = fixture
        self.requests: list[dict[str, Any]] = []
        super().__init__(
            provider_id="faux-openai",
            name="faux openai provider",
            models=[
                ModelInfo(
                    id=fixture.model,
                    provider="faux-openai",
                    api="openai",
                )
            ],
        )

    @classmethod
    def from_file(cls, path: Path | str) -> FauxOpenAIProvider:
        return cls(load_fixture(path))

    async def _raw_chunks(
        self, payload: dict[str, Any], options: StreamOptions
    ) -> AsyncIterator[dict[str, Any]]:
        """回放 fixture：记录 payload，按需抛连接异常，逐 chunk 产出。"""
        del options
        self.requests.append(payload)
        if self.fixture.error:
            _raise_openai_connection_error(f"faux connection error: {self.fixture.error}")
        for chunk in self.fixture.chunks:
            yield chunk


class FauxAnthropicProvider(AnthropicProtocolProvider):
    """Claude 协议的 faux provider（复用真实翻译与守卫路径）。"""

    def __init__(self, fixture: FauxFixture) -> None:
        super().__init__(
            provider_id="faux-anthropic",
            name="faux anthropic provider",
            models=[
                ModelInfo(
                    id=fixture.model,
                    provider="faux-anthropic",
                    api="anthropic",
                )
            ],
        )
        self.fixture = fixture
        self.requests: list[dict[str, Any]] = []

    @classmethod
    def from_file(cls, path: Path | str) -> FauxAnthropicProvider:
        return cls(load_fixture(path))

    async def _raw_events(
        self, payload: dict[str, Any], options: StreamOptions
    ) -> AsyncIterator[dict[str, Any]]:
        """回放 fixture：记录 payload，按需抛连接异常，逐事件产出。"""
        del options
        self.requests.append(payload)
        if self.fixture.error:
            _raise_anthropic_connection_error(f"faux connection error: {self.fixture.error}")
        for event in self.fixture.chunks:
            yield event


__all__ = [
    "FauxFixture",
    "FauxAnthropicProvider",
    "FauxOpenAIProvider",
    "load_fixture",
]
