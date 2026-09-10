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


# ---------------------------------------------------------------------------
# 目录驱动回放（faux:// base_url 接线：进程级 e2e）
# ---------------------------------------------------------------------------

FAUX_URL_SCHEME = "faux://"
"""faux transport 的 base_url scheme 前缀（目录回放，零真实网络）。"""

FIXTURES_FILENAME = "fixtures.json"
"""回放脚本文件名（FauxFixture 数组的 JSON 文件）。"""

REQUESTS_LOG_FILENAME = "requests.jsonl"
"""请求记录文件名（每次流式调用的 payload 逐行追加）。"""


def faux_dir_from_base_url(base_url: str | None) -> Path | None:
    """解析 faux:// base_url → 回放目录；非 faux scheme 返回 None。

    形态：``faux://<路径>``（绝对路径或相对 cwd 的路径；
    Windows 盘符写作 ``faux://E:/dir``）。
    """
    if base_url is None or not base_url.startswith(FAUX_URL_SCHEME):
        return None
    return Path(base_url[len(FAUX_URL_SCHEME) :])


def _initial_call_index(record_path: Path) -> int:
    """请求记录行数 → 轮转起点（跨进程续接回放序列）。"""
    try:
        with record_path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


class FauxReplayOpenAIProvider(OpenAIProtocolProvider):
    """目录驱动的多轮回放 faux（进程级 e2e 用）。

    与单 fixture 的 FauxOpenAIProvider 差异（fixture 与真实 SDK
    响应形态的同构性同 FauxFixture 约定）：
    - 回放脚本来自 ``<dir>/fixtures.json``，按流式调用序轮转，
      超出后重放最后一个
    - 轮转起点 = 请求记录文件的已有行数（进程重启后续接，
      支撑「第一次进程 → 第二次 -c 进程」的跨进程脚本序）
    - 每次调用的 payload 追加到 ``<dir>/requests.jsonl``，
      e2e 断言「faux 收到的消息序列」的数据源
    """

    def __init__(
        self,
        *,
        scripts: list[FauxFixture],
        record_path: Path,
        provider_id: str,
        name: str,
        models: list[ModelInfo],
    ) -> None:
        self._scripts = scripts
        self._record_path = record_path
        self._call_index = _initial_call_index(record_path)
        self.requests: list[dict[str, Any]] = []
        super().__init__(provider_id=provider_id, name=name, models=models)

    async def _raw_chunks(
        self, payload: dict[str, Any], options: StreamOptions
    ) -> AsyncIterator[dict[str, Any]]:
        """记录请求并回放当前轮脚本。"""
        del options
        self.requests.append(payload)
        self._append_record(payload)
        index = min(self._call_index, len(self._scripts) - 1)
        self._call_index += 1
        script = self._scripts[index]
        if script.error:
            _raise_openai_connection_error(f"faux connection error: {script.error}")
        for chunk in script.chunks:
            yield chunk

    def _append_record(self, payload: dict[str, Any]) -> None:
        """请求落盘（JSONL 追加）。"""
        self._record_path.parent.mkdir(parents=True, exist_ok=True)
        with self._record_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = [
    "FAUX_URL_SCHEME",
    "FIXTURES_FILENAME",
    "FauxAnthropicProvider",
    "FauxFixture",
    "FauxOpenAIProvider",
    "FauxReplayOpenAIProvider",
    "REQUESTS_LOG_FILENAME",
    "faux_dir_from_base_url",
    "load_fixture",
]
