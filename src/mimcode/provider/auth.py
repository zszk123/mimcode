"""API key 解析链：环境变量 > 配置文件。

对齐 checklist：``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` 及端点自定义
变量名（如 DEEPSEEK_API_KEY）；配置文件内联 api_key 仅作后备。

与 T3 的 Provider.resolve_api_key（选项 > 环境变量）衔接：
本模块补上「配置文件」一层，最终链为：
显式传入 > 环境变量（端点自定义名）> 配置内联 key。
"""

from __future__ import annotations

import os

from mimcode.config import EndpointEntry

DEFAULT_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
"""协议的默认环境变量名。"""


def resolve_api_key(
    entry: EndpointEntry,
    explicit: str | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> str | None:
    """解析端点的 API key。

    Args:
        entry: 端点条目（提供自定义环境变量名与内联 key）。
        explicit: 调用方显式传入的 key（最高优先级）。
        environ: 环境变量表（测试注入用；None 取 os.environ）。

    Returns:
        解析到的 key；全链未命中返回 None。
    """
    if explicit:
        return explicit
    env = environ if environ is not None else os.environ
    protocol_env = DEFAULT_API_KEY_ENV.get(entry.protocol or "openai")
    key_env = entry.api_key_env or protocol_env or DEFAULT_API_KEY_ENV["openai"]
    value = env.get(key_env)
    if value:
        return value
    return entry.api_key or None


def api_key_env_name(entry: EndpointEntry) -> str:
    """端点实际使用的环境变量名（诊断与提示用）。"""
    protocol_env = DEFAULT_API_KEY_ENV.get(entry.protocol or "openai")
    return entry.api_key_env or protocol_env or DEFAULT_API_KEY_ENV["openai"]
