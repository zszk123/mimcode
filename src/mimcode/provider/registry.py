"""端点注册表：端点条目 → provider 实例与模型解析。

「协议即 provider」的装配层：
- 端点条目（EndpointEntry）+ 模型条目 → 具体协议 provider 实例
- 模型解析：``--model`` 取值（裸模型 id 或 endpoint/model 形态）→ ModelInfo

模型 id 解析规则（对齐 pi 的 scoped model 语义子集）：
- ``endpoint/model``：指定端点下的模型
- ``model``：在全部端点中查找该模型 id；命中多个时默认端点优先

API key 不在注册表固化：provider 流式调用时经 StreamOptions.api_key
（agent 层从解析链取值后传入），保证环境变量变更即时生效。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from mimcode.config import Config, EndpointEntry, ModelEntry
from mimcode.provider.anthropic_protocol import AnthropicProtocolProvider
from mimcode.provider.base import Provider
from mimcode.provider.openai_protocol import OpenAIProtocolProvider
from mimcode.types import Api, ModelInfo

# 模型条目缺省值（ModelEntry 未覆盖时使用）
_DEFAULT_CONTEXT_WINDOW = 128000
_DEFAULT_MAX_OUTPUT_TOKENS = 8192

_DEFAULT_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def build_model_info(endpoint_name: str, endpoint: EndpointEntry, model_id: str) -> ModelInfo:
    """端点 + 模型条目 → ModelInfo（条目缺省字段落到默认值）。"""
    entry = (endpoint.models or {}).get(model_id) or ModelEntry()
    protocol = endpoint.protocol or "openai"
    return ModelInfo(
        id=model_id,
        provider=endpoint_name,
        api=protocol,
        name=entry.name,
        context_window=entry.context_window or _DEFAULT_CONTEXT_WINDOW,
        max_output_tokens=entry.max_output_tokens or _DEFAULT_MAX_OUTPUT_TOKENS,
        supports_thinking=entry.supports_thinking or False,
    )


def _require_protocol(endpoint_name: str, endpoint: EndpointEntry) -> Api:
    """三层合并后协议仍缺失 → 显式错误（而非静默回落）。"""
    if endpoint.protocol is None:
        raise ValueError(f"端点 '{endpoint_name}' 的协议未声明（配置与预置目录均未提供 protocol）")
    return endpoint.protocol


def _create_faux_replay(endpoint_name: str, endpoint: EndpointEntry, faux_dir: Path) -> Provider:
    """faux:// 端点 → 目录回放 provider（进程级 e2e 专用）。

    Raises:
        ValueError: 协议非 openai、回放脚本缺失/损坏/为空、未声明模型。
    """
    from mimcode.provider.faux import (
        FIXTURES_FILENAME,
        REQUESTS_LOG_FILENAME,
        FauxFixture,
        FauxReplayOpenAIProvider,
    )

    protocol = _require_protocol(endpoint_name, endpoint)
    if protocol != "openai":
        raise ValueError(
            f"faux transport 仅支持 openai 协议（端点 '{endpoint_name}' 声明 '{protocol}'）"
        )
    fixtures_path = faux_dir / FIXTURES_FILENAME
    if not fixtures_path.is_file():
        raise ValueError(f"faux 回放脚本不存在: {fixtures_path}")
    try:
        scripts = [
            FauxFixture.model_validate(item)
            for item in json.loads(fixtures_path.read_text(encoding="utf-8"))
        ]
    except ValueError as exc:  # JSON 解析与 pydantic 校验错误
        raise ValueError(f"faux 回放脚本不合法: {fixtures_path}: {exc}") from exc
    if not scripts:
        raise ValueError(f"faux 回放脚本为空: {fixtures_path}")
    models = [
        build_model_info(endpoint_name, endpoint, model_id) for model_id in (endpoint.models or {})
    ]
    if not models:
        raise ValueError(f"faux 端点 '{endpoint_name}' 未声明模型条目")
    return FauxReplayOpenAIProvider(
        scripts=scripts,
        record_path=faux_dir / REQUESTS_LOG_FILENAME,
        provider_id=endpoint_name,
        name=endpoint_name,
        models=models,
    )


def create_provider(
    endpoint_name: str,
    endpoint: EndpointEntry,
    *,
    models: list[ModelInfo] | None = None,
) -> Provider:
    """端点条目 → 协议 provider 实例（key 在流调用时经 options 传入）。

    Raises:
        ValueError: 三层合并后协议仍缺失。
    """
    from mimcode.provider.faux import faux_dir_from_base_url

    faux_dir = faux_dir_from_base_url(endpoint.base_url)
    if faux_dir is not None:
        return _create_faux_replay(endpoint_name, endpoint, faux_dir)
    protocol = _require_protocol(endpoint_name, endpoint)
    resolved_models = models or [
        build_model_info(endpoint_name, endpoint, model_id) for model_id in (endpoint.models or {})
    ]
    common_kwargs: dict = {
        "provider_id": endpoint_name,
        "name": endpoint_name,
        "base_url": endpoint.base_url,
        "models": resolved_models,
    }
    if protocol == "openai":
        return OpenAIProtocolProvider(
            api_key_env=endpoint.api_key_env or _DEFAULT_API_KEY_ENV["openai"],
            **common_kwargs,
        )
    return AnthropicProtocolProvider(
        api_key_env=endpoint.api_key_env or _DEFAULT_API_KEY_ENV["anthropic"],
        **common_kwargs,
    )


@dataclasses.dataclass
class ModelResolution:
    """模型解析结果。"""

    model: ModelInfo
    endpoint_name: str


class Registry:
    """端点注册表：配置合并目录的 provider/模型查询入口。"""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._providers: dict[str, Provider] = {
            name: create_provider(name, entry) for name, entry in config.endpoints.items()
        }

    @property
    def config(self) -> Config:
        """合并后的有效配置。"""
        return self._config

    def providers(self) -> dict[str, Provider]:
        """全部 provider 实例（按端点名索引）。"""
        return dict(self._providers)

    def get_provider(self, endpoint_name: str) -> Provider | None:
        """按端点名取 provider。"""
        return self._providers.get(endpoint_name)

    def all_models(self) -> list[ModelInfo]:
        """全部端点的全部模型。"""
        result: list[ModelInfo] = []
        for provider in self._providers.values():
            result.extend(provider.get_models())
        return result

    def resolve_model(self, spec: str) -> ModelResolution | None:
        """解析 --model 取值（``endpoint/model`` 或裸模型 id）。"""
        from mimcode.provider.catalog import default_endpoint_name

        if "/" in spec:
            endpoint_name, model_id = spec.split("/", 1)
            provider = self._providers.get(endpoint_name)
            if provider is None:
                return None
            for model in provider.get_models():
                if model.id == model_id:
                    return ModelResolution(model=model, endpoint_name=endpoint_name)
            return None

        candidates: list[ModelResolution] = []
        for endpoint_name, provider in self._providers.items():
            for model in provider.get_models():
                if model.id == spec:
                    candidates.append(ModelResolution(model=model, endpoint_name=endpoint_name))
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # 多端点同名模型：默认端点优先，其次按端点名稳定排序
        preferred = default_endpoint_name(self._config)
        for candidate in candidates:
            if candidate.endpoint_name == preferred:
                return candidate
        return sorted(candidates, key=lambda item: item.endpoint_name)[0]


def build_registry(user_config: Config) -> Registry:
    """用户配置 + 预置目录 → Registry（预置为底、用户覆盖同名条目）。

    Raises:
        ValueError: 某端点三层合并后协议仍缺失。
    """
    from mimcode.provider.catalog import merge_with_builtin

    merged = merge_with_builtin(user_config)
    return Registry(merged)
