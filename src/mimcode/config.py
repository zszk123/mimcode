"""配置文件层：TOML 两级配置（全局 > 项目）的发现、解析与合并。

对齐 pi 的配置层次思路（settings-manager.ts + defaults.ts），
按 v1 裁剪：仅端点与模型条目，不含主题/键位等 TUI 配置。

文件位置（checklist 约定）：
- 全局：``~/.mimcode/config.toml``
- 项目：``<cwd>/.mimcode/config.toml``

合并语义（两级 + 预置目录三层，逐层覆盖）：
- 同名端点条目：项目级字段覆盖全局，全局覆盖预置；
- 协议字段（protocol）允许在单层缺省，由下层合并补齐；
  最终（三层合并后）仍缺失时在注册表层校验报错。

TOML 形态::

    [endpoints.openai]
    protocol = "openai"           # "openai" | "anthropic"（可缺省，由下层补齐）
    base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"
    api_key = "sk-..."            # 可选；环境变量优先
    default = true                # 可选：默认端点

    [endpoints.openai.models.gpt-5]
    context_window = 400000
    supports_thinking = true
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mimcode.types import Api

MIMCODE_DIR = ".mimcode"
"""项目级配置目录名。"""

CONFIG_FILENAME = "config.toml"
"""配置文件名。"""


class ModelEntry(BaseModel):
    """配置文件中的模型条目（字段均可选，缺省用目录/默认值）。"""

    name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_thinking: bool | None = None


class EndpointEntry(BaseModel):
    """配置文件中的端点条目。

    Attributes:
        protocol: 协议（openai / anthropic）。允许缺省（None），
            由合并链下层补齐；最终仍缺失时注册表层报错。
        base_url: 端点 URL；None 时用协议默认官方地址。
        api_key_env: 环境变量名；缺省用协议默认变量名。
        api_key: 配置文件内联 key（环境变量优先于它）。
        default: 是否默认端点。None 表示未声明（不参与覆盖）。
        models: 模型条目表（key 为模型 id）；声明即整体替换下层同名表。
    """

    model_config = ConfigDict(extra="forbid")

    protocol: Api | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    default: bool | None = None
    models: dict[str, ModelEntry] | None = None


class Config(BaseModel):
    """单层（或合并后的）配置。"""

    model_config = ConfigDict(extra="forbid")

    endpoints: dict[str, EndpointEntry] = Field(default_factory=dict)


def global_config_dir(home: Path | None = None) -> Path:
    """全局配置目录（~/.mimcode）。"""
    return (home or Path.home()) / ".mimcode"


def global_config_path(home: Path | None = None) -> Path:
    """全局配置文件路径。"""
    return global_config_dir(home) / CONFIG_FILENAME


def project_config_path(cwd: Path | None = None) -> Path:
    """项目级配置文件路径（cwd/.mimcode/config.toml）。"""
    return (cwd or Path.cwd()) / MIMCODE_DIR / CONFIG_FILENAME


def _load_toml(path: Path) -> dict[str, Any] | None:
    """读取单个 TOML 文件为原始 dict；文件不存在返回 None。

    Raises:
        ValueError: TOML 语法错误。
    """
    if not path.is_file():
        return None
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"配置文件 TOML 解析失败: {path}: {exc}") from exc


def _validate(data: dict[str, Any], path: Path) -> Config:
    """校验原始 dict 的结构并转 Config。

    Raises:
        ValueError: 结构不符合 schema（含未知键、非法枚举）。
    """
    try:
        return Config.model_validate(data)
    except ValueError as exc:
        raise ValueError(f"配置文件结构不合法: {path}: {exc}") from exc


def _entry_overrides(entry: EndpointEntry) -> dict[str, Any]:
    """提取端点条目中「显式声明」的字段（None 视为未声明）。"""
    return {key: value for key, value in entry.model_dump().items() if value is not None}


def merge_endpoint_fields(base: EndpointEntry, override: EndpointEntry) -> EndpointEntry:
    """端点条目覆盖合并：override 显式声明的字段胜出，其余保留 base。

    None 与缺省都视为「未声明」；协议字段与模型表同样按此语义
    （override 未声明协议时沿用 base 的协议声明）。
    """
    merged = _entry_overrides(override)
    combined = base.model_dump() | merged
    return EndpointEntry.model_validate(combined)


def _merge_configs(base: Config, override: Config) -> Config:
    """配置级合并：同名端点按字段覆盖，新端点直接并入。"""
    merged_endpoints = {name: entry.model_copy(deep=True) for name, entry in base.endpoints.items()}
    for name, entry in override.endpoints.items():
        existing = merged_endpoints.get(name)
        merged_endpoints[name] = (
            merge_endpoint_fields(existing, entry) if existing is not None else entry
        )
    return Config(endpoints=merged_endpoints)


def load_config(
    cwd: Path | None = None,
    home: Path | None = None,
) -> Config:
    """加载并合并两级配置（项目覆盖全局）。

    Args:
        cwd: 项目目录（None 取当前目录）。
        home: 用户主目录（None 取 Path.home()）。

    Returns:
        合并后的有效配置；两级均不存在时返回空配置。
        单层协议缺省是合法的（可被另一层补齐）。

    Raises:
        ValueError: 任一文件 TOML 语法错误或结构不合法（含文件路径）。
    """
    raw_global = _load_toml(global_config_path(home))
    raw_project = _load_toml(project_config_path(cwd))

    global_config = _validate(raw_global, global_config_path(home)) if raw_global else None
    project_config = _validate(raw_project, project_config_path(cwd)) if raw_project else None

    if global_config is None:
        return project_config or Config()
    if project_config is None:
        return global_config
    return _merge_configs(global_config, project_config)
