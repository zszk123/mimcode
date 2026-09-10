"""预置端点与模型目录（「协议即 provider」的内置条目）。

对齐 pi providers/all.ts 的角色（内置 provider 注册表），但形态收敛为
「端点条目表」：每个条目描述一个可经 OpenAI/Claude 协议访问的端点，
用户可在 config.toml 中覆盖或新增同名条目（config 合并逻辑）。

预置目录覆盖（checklist）：openai 官方、anthropic 官方、deepseek、
zhipu（GLM）、kimi（moonshot）共 5 组。
模型元数据为常用公开值，用户可经配置逐字段覆盖。
"""

from __future__ import annotations

from mimcode.config import Config, EndpointEntry, ModelEntry

# 各端点条目：协议、base_url、默认环境变量名与模型表
_BUILTIN_ENDPOINTS: dict[str, EndpointEntry] = {
    "openai": EndpointEntry(
        protocol="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default=True,
        models={
            "gpt-5": ModelEntry(
                name="GPT-5",
                context_window=400000,
                max_output_tokens=128000,
                supports_thinking=True,
            ),
            "gpt-5-mini": ModelEntry(
                name="GPT-5 mini",
                context_window=400000,
                max_output_tokens=128000,
                supports_thinking=True,
            ),
        },
    ),
    "anthropic": EndpointEntry(
        protocol="anthropic",
        base_url="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
        models={
            "claude-sonnet-4-5": ModelEntry(
                name="Claude Sonnet 4.5",
                context_window=200000,
                max_output_tokens=64000,
                supports_thinking=True,
            ),
            "claude-opus-4-1": ModelEntry(
                name="Claude Opus 4.1",
                context_window=200000,
                max_output_tokens=32000,
                supports_thinking=True,
            ),
        },
    ),
    "deepseek": EndpointEntry(
        protocol="openai",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        models={
            "deepseek-chat": ModelEntry(
                name="DeepSeek Chat (V3)",
                context_window=128000,
                max_output_tokens=8192,
            ),
            "deepseek-reasoner": ModelEntry(
                name="DeepSeek Reasoner (R1)",
                context_window=128000,
                max_output_tokens=65536,
                supports_thinking=True,
            ),
        },
    ),
    "zhipu": EndpointEntry(
        protocol="openai",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="ZHIPU_API_KEY",
        models={
            "glm-4.6": ModelEntry(
                name="GLM-4.6",
                context_window=200000,
                max_output_tokens=32768,
                supports_thinking=True,
            ),
        },
    ),
    "kimi": EndpointEntry(
        protocol="openai",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="KIMI_API_KEY",
        models={
            "kimi-k2-turbo-preview": ModelEntry(
                name="Kimi K2 Turbo",
                context_window=256000,
                max_output_tokens=32768,
            ),
        },
    ),
}


def builtin_config() -> Config:
    """预置目录的 Config 形态（作为合并链的最底层）。"""
    return Config(
        endpoints={name: entry.model_copy(deep=True) for name, entry in _BUILTIN_ENDPOINTS.items()}
    )


def builtin_endpoint(name: str) -> EndpointEntry | None:
    """按名取预置端点条目（不存在返回 None）。"""
    entry = _BUILTIN_ENDPOINTS.get(name)
    return entry.model_copy(deep=True) if entry is not None else None


def default_endpoint_name(user_config: Config) -> str | None:
    """解析默认端点名：用户显式 default 优先，否则取预置 openai。

    Returns:
        端点名；配置无端点时返回 None。
    """
    for name, entry in user_config.endpoints.items():
        if entry.default:
            return name
    if "openai" in user_config.endpoints:
        return "openai"
    return next(iter(user_config.endpoints), None)


def merge_with_builtin(user_config: Config) -> Config:
    """预置目录 + 用户配置合并（用户条目字段级覆盖同名预置条目）。

    合并前先清空用户配置中与预置冲突的 default 标记之外的逻辑不变：
    仅做字段覆盖；若用户没有任何端点声明 default，则预置 openai
    的 default=True 继续生效（与 default_endpoint_name 的回落一致）。
    """
    from mimcode.config import _merge_configs

    merged = _merge_configs(builtin_config(), user_config)
    # 用户已显式指定 default 端点时，预置 openai 的 default 标记失效
    user_default = any(entry.default for entry in user_config.endpoints.values())
    if user_default:
        openai_entry = merged.endpoints.get("openai")
        if (
            openai_entry is not None
            and openai_entry.default
            and "openai" not in user_config.endpoints
        ):
            merged.endpoints["openai"] = openai_entry.model_copy(update={"default": None})
    return merged
