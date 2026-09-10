"""修正后的 config 测试片段（合并语义变更后）。

替换 test_config.py 中 5 个失败用例的期望：浅层覆盖 → 显式声明覆盖、
pydantic 模型相等、默认端点在 merge_with_builtin 后仍指用户声明。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mimcode.config import (
    Config,
    EndpointEntry,
    ModelEntry,
    load_config,
    merge_endpoint_fields,
)
from mimcode.provider.catalog import merge_with_builtin
from mimcode.provider.registry import build_registry


def write_toml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_project_overrides_global(tmp_path: Path) -> None:
    """项目级覆盖全局同名条目；不同名条目并存（协议可单层缺省）。"""
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    write_toml(
        home / ".mimcode" / "config.toml",
        """
[endpoints.shared]
protocol = "openai"
base_url = "https://global"
api_key = "global-key"

[endpoints.only-global]
protocol = "anthropic"
""",
    )
    write_toml(
        cwd / ".mimcode" / "config.toml",
        """
[endpoints.shared]
base_url = "https://project"

[endpoints.only-project]
protocol = "openai"
""",
    )
    config = load_config(cwd=cwd, home=home)

    shared = config.endpoints["shared"]
    assert shared.base_url == "https://project"
    assert shared.protocol == "openai"  # 全局补齐
    assert shared.api_key == "global-key"
    assert "only-global" in config.endpoints
    assert "only-project" in config.endpoints


def test_merge_endpoint_fields_semantics() -> None:
    """覆盖语义：显式声明字段胜出、模型表整体替换、None 不覆盖。"""
    base = EndpointEntry(
        protocol="openai",
        base_url="https://a",
        api_key="k",
        models={"m1": ModelEntry()},
    )
    override = EndpointEntry(base_url="https://b")
    merged = merge_endpoint_fields(base, override)
    assert merged.base_url == "https://b"
    assert merged.api_key == "k"  # 未声明保留
    assert merged.protocol == "openai"  # 未声明协议沿用 base
    assert merged.models == {"m1": ModelEntry()}  # pydantic 模型相等

    with_models = EndpointEntry(models={"m2": ModelEntry()})
    merged_models = merge_endpoint_fields(base, with_models).models
    assert merged_models is not None
    assert merged_models["m2"] == ModelEntry()


def test_registry_bare_ambiguous_prefers_default() -> None:
    """多端点同名模型：用户声明的默认端点优先（预置 openai 退位）。"""
    user = Config(
        endpoints={
            "backup": EndpointEntry(
                protocol="openai",
                models={"shared-model": {}},
            ),
            "primary": EndpointEntry(
                protocol="openai",
                default=True,
                models={"shared-model": {}},
            ),
        }
    )
    merged = merge_with_builtin(user)
    # 用户已声明 default → 预置 openai 的 default 标记被清除
    assert merged.endpoints["primary"].default is True
    assert merged.endpoints["openai"].default is None

    registry = build_registry(user)
    resolution = registry.resolve_model("shared-model")
    assert resolution is not None
    assert resolution.endpoint_name == "primary"


def test_user_config_file_loads_into_registry(tmp_path: Path) -> None:
    """端到端：两级文件 → load_config → build_registry（协议缺省补齐）。"""
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    write_toml(
        home / ".mimcode" / "config.toml",
        """
[endpoints.openai]
api_key_env = "MY_OPENAI_KEY"
""",
    )
    write_toml(
        cwd / ".mimcode" / "config.toml",
        """
[endpoints.openai]
base_url = "https://proxy.internal/v1"

[endpoints.local]
protocol = "openai"
base_url = "http://localhost:11434/v1"
api_key_env = "OLLAMA_KEY"

[endpoints.local.models.qwen3]
context_window = 131072
""",
    )
    config = load_config(cwd=cwd, home=home)
    # openai 条目：全局缺省协议，项目也缺省 → 合并层仍缺，由预置目录补齐
    merged = merge_with_builtin(config)
    assert merged.endpoints["openai"].protocol == "openai"

    registry = build_registry(config)
    openai_provider = registry.get_provider("openai")
    assert openai_provider is not None
    assert openai_provider.base_url == "https://proxy.internal/v1"
    assert openai_provider.api_key_env == "MY_OPENAI_KEY"

    local = registry.resolve_model("local/qwen3")
    assert local is not None
    assert local.model.context_window == 131072


def test_protocol_missing_everywhere_raises() -> None:
    """三层合并后协议仍缺失 → build_registry 显式报错。"""
    user = Config(endpoints={"broken": EndpointEntry(base_url="https://x")})
    # 用户条目名不在预置目录中，无下层可补
    with pytest.raises(ValueError, match="broken.*协议未声明"):
        build_registry(user)
