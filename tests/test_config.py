"""T4 配置层测试：TOML 两级合并、目录、key 解析链、模型解析。

checklist 对应项：
- 项目级 .mimcode/config.toml 覆盖全局同名端点条目（合并顺序断言）
- API key 解析顺序：环境变量（含端点自定义变量名）> 配置文件内联
- 预置目录至少含 openai/anthropic/deepseek/zhipu/kimi 5 组端点
- 配置文件新增自定义条目后出现在 --list-models 输出
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mimcode.config import (
    Config,
    EndpointEntry,
    load_config,
    project_config_path,
)
from mimcode.provider.auth import api_key_env_name, resolve_api_key
from mimcode.provider.catalog import (
    builtin_config,
    builtin_endpoint,
    default_endpoint_name,
    merge_with_builtin,
)
from mimcode.provider.registry import build_registry


def write_toml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_config_paths(tmp_path: Path) -> None:
    """路径约定：~/.mimcode/config.toml 与 cwd/.mimcode/config.toml。"""
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    from mimcode.config import global_config_path

    assert global_config_path(home) == home / ".mimcode" / "config.toml"
    assert project_config_path(cwd) == cwd / ".mimcode" / "config.toml"


def test_load_config_missing_files(tmp_path: Path) -> None:
    """两级文件都不存在：空配置。"""
    config = load_config(cwd=tmp_path / "cwd", home=tmp_path / "home")
    assert config.endpoints == {}


def test_load_global_only(tmp_path: Path) -> None:
    """仅全局配置。"""
    home = tmp_path / "home"
    write_toml(
        home / ".mimcode" / "config.toml",
        '[endpoints.my-endpoint]\nprotocol = "openai"\nbase_url = "https://x"\n',
    )
    config = load_config(cwd=tmp_path / "cwd", home=home)
    assert config.endpoints["my-endpoint"].protocol == "openai"
    assert config.endpoints["my-endpoint"].base_url == "https://x"


def test_invalid_toml_raises_value_error(tmp_path: Path) -> None:
    """TOML 语法错误 → ValueError（含文件路径）。"""
    home = tmp_path / "home"
    write_toml(home / ".mimcode" / "config.toml", "not [valid toml")
    with pytest.raises(ValueError, match="config.toml"):
        load_config(cwd=tmp_path / "cwd", home=home)


def test_unknown_protocol_rejected(tmp_path: Path) -> None:
    """非法协议取值被 schema 拒绝。"""
    home = tmp_path / "home"
    write_toml(
        home / ".mimcode" / "config.toml",
        '[endpoints.x]\nprotocol = "google"\n',
    )
    with pytest.raises(ValueError, match="不合法"):
        load_config(cwd=tmp_path / "cwd", home=home)


def test_extra_field_rejected(tmp_path: Path) -> None:
    """未知顶层键被拒绝（extra=forbid 防拼写错误静默丢失）。"""
    home = tmp_path / "home"
    write_toml(
        home / ".mimcode" / "config.toml",
        '[endpoints.x]\nprotocol = "openai"\nunknown_field = 1\n',
    )
    with pytest.raises(ValueError, match="不合法"):
        load_config(cwd=tmp_path / "cwd", home=home)


# ---------------------------------------------------------------------------
# 预置目录
# ---------------------------------------------------------------------------


def test_builtin_catalog_covers_five_endpoints() -> None:
    """预置目录含 5 组端点（checklist）。"""
    endpoints = builtin_config().endpoints
    for name in ("openai", "anthropic", "deepseek", "zhipu", "kimi"):
        assert name in endpoints, f"缺少预置端点 {name}"
    assert builtin_endpoint("deepseek") is not None
    assert builtin_endpoint("nonexistent") is None


def test_builtin_endpoint_protocols() -> None:
    """协议正确：anthropic 官方为 claude 协议，其余为 openai 协议。"""
    endpoints = builtin_config().endpoints
    assert endpoints["anthropic"].protocol == "anthropic"
    for name in ("openai", "deepseek", "zhipu", "kimi"):
        assert endpoints[name].protocol == "openai", name


def test_default_endpoint_resolution() -> None:
    """默认端点：用户显式 default 优先，否则回落 openai。"""
    empty = Config()
    assert default_endpoint_name(empty) is None
    assert default_endpoint_name(builtin_config()) == "openai"

    user = Config(
        endpoints={
            "mine": EndpointEntry(protocol="openai", default=True),
        }
    )
    assert default_endpoint_name(user) == "mine"


def test_merge_with_builtin_overrides() -> None:
    """用户覆盖预置：字段级覆盖 + 新增端点并存。"""
    user = Config(
        endpoints={
            "deepseek": EndpointEntry(protocol="openai", base_url="https://my-proxy"),
            "custom": EndpointEntry(protocol="openai", base_url="https://custom"),
        }
    )
    merged = merge_with_builtin(user)
    assert merged.endpoints["deepseek"].base_url == "https://my-proxy"
    # 预置默认 key 环境名保留
    assert merged.endpoints["deepseek"].api_key_env == "DEEPSEEK_API_KEY"
    assert "custom" in merged.endpoints
    assert "openai" in merged.endpoints


# ---------------------------------------------------------------------------
# API key 解析链
# ---------------------------------------------------------------------------


def test_api_key_explicit_wins() -> None:
    """显式传入 > 环境变量 > 内联。"""
    entry = EndpointEntry(protocol="openai", api_key_env="MY_KEY", api_key="inline-key")
    resolved = resolve_api_key(entry, explicit="explicit-key", environ={"MY_KEY": "env-key"})
    assert resolved == "explicit-key"


def test_api_key_env_var_wins_over_inline() -> None:
    """环境变量 > 配置内联（checklist）。"""
    entry = EndpointEntry(protocol="openai", api_key_env="MY_KEY", api_key="inline-key")
    assert resolve_api_key(entry, environ={"MY_KEY": "env-key"}) == "env-key"


def test_api_key_custom_env_name() -> None:
    """端点自定义变量名（如 DEEPSEEK_API_KEY）生效。"""
    entry = EndpointEntry(protocol="openai", api_key_env="DEEPSEEK_API_KEY")
    assert resolve_api_key(entry, environ={"DEEPSEEK_API_KEY": "ds-key"}) == "ds-key"


def test_api_key_default_env_fallback() -> None:
    """未自定义变量名时回落协议默认（OPENAI_API_KEY / ANTHROPIC_API_KEY）。"""
    openai_entry = EndpointEntry(protocol="openai")
    anthropic_entry = EndpointEntry(protocol="anthropic")
    assert resolve_api_key(openai_entry, environ={"OPENAI_API_KEY": "o"}) == "o"
    assert resolve_api_key(anthropic_entry, environ={"ANTHROPIC_API_KEY": "a"}) == "a"
    assert api_key_env_name(openai_entry) == "OPENAI_API_KEY"
    assert api_key_env_name(anthropic_entry) == "ANTHROPIC_API_KEY"


def test_api_key_inline_when_no_env() -> None:
    """环境变量缺失时回落配置内联 key。"""
    entry = EndpointEntry(protocol="openai", api_key_env="MY_KEY", api_key="inline")
    assert resolve_api_key(entry, environ={}) == "inline"
    assert resolve_api_key(entry, environ={"MY_KEY": ""}) == "inline"


def test_api_key_none_when_unset() -> None:
    """全链未命中返回 None。"""
    entry = EndpointEntry(protocol="openai")
    assert resolve_api_key(entry, environ={}) is None


# ---------------------------------------------------------------------------
# 注册表与模型解析
# ---------------------------------------------------------------------------


def test_registry_from_builtin() -> None:
    """预置目录构建：5 个 provider、模型可枚举。"""
    registry = build_registry(Config())
    names = set(registry.providers())
    assert {"openai", "anthropic", "deepseek", "zhipu", "kimi"} <= names
    models = registry.all_models()
    assert len(models) >= 7
    model_ids = {m.id for m in models}
    assert "gpt-5" in model_ids
    assert "claude-sonnet-4-5" in model_ids


def test_registry_resolves_scoped_and_bare() -> None:
    """模型解析：endpoint/model 与裸 id 两种形态。"""
    registry = build_registry(Config())
    scoped = registry.resolve_model("deepseek/deepseek-chat")
    assert scoped is not None
    assert scoped.model.id == "deepseek-chat"
    assert scoped.endpoint_name == "deepseek"

    bare = registry.resolve_model("glm-4.6")
    assert bare is not None
    assert bare.endpoint_name == "zhipu"


def test_registry_resolve_miss() -> None:
    """未命中（未知模型/未知端点）返回 None。"""
    registry = build_registry(Config())
    assert registry.resolve_model("no-such-model") is None
    assert registry.resolve_model("no-endpoint/model") is None


def test_registry_custom_endpoint_appears() -> None:
    """用户自定义端点 + 自定义模型进入注册表。"""
    user = Config(
        endpoints={
            "my-proxy": EndpointEntry(
                protocol="openai",
                base_url="https://my-proxy/v1",
                api_key_env="MY_PROXY_KEY",
                models={"my-model": {}},
            )
        }
    )
    registry = build_registry(user)
    resolution = registry.resolve_model("my-proxy/my-model")
    assert resolution is not None
    assert resolution.model.id == "my-model"
    assert registry.get_provider("my-proxy") is not None
