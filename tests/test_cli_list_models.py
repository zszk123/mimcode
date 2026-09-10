"""--list-models 与 T4 冒烟修正（测试辅助）。"""

from __future__ import annotations

import io
from pathlib import Path

from mimcode.cli import main
from mimcode.cli.list_models import format_model_line, list_models
from mimcode.config import Config, EndpointEntry, ModelEntry
from mimcode.provider.registry import build_registry


def test_format_model_line() -> None:
    """单行格式：端点（协议）+ 模型 id + 窗口。"""
    line = format_model_line("deepseek", "openai", "deepseek-chat", 128000)
    assert line == "deepseek (openai)\tdeepseek-chat\t128000"


def test_list_models_contains_builtin_five() -> None:
    """预置 5 组端点全部出现。"""
    registry = build_registry(Config())
    stream = io.StringIO()
    list_models(registry, stream=stream)
    output = stream.getvalue()
    for expected in (
        "openai (openai)",
        "anthropic (anthropic)",
        "deepseek (openai)",
        "zhipu (openai)",
        "kimi (openai)",
    ):
        assert expected in output, expected
    assert "gpt-5" in output
    assert "128000" in output


def test_list_models_custom_endpoint_visible() -> None:
    """自定义端点条目出现在列表。"""
    user = Config(
        endpoints={
            "my-proxy": EndpointEntry(
                protocol="openai",
                base_url="https://my-proxy/v1",
                models={"my-model": {}},
            )
        }
    )
    registry = build_registry(user)
    stream = io.StringIO()
    list_models(registry, stream=stream)
    assert "my-proxy (openai)\tmy-model" in stream.getvalue()


def test_main_list_models_exit_zero(tmp_path: Path, monkeypatch) -> None:
    """main(['--list-models'])：正常退出（空用户配置 + 预置目录）。"""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    assert main(["--list-models"]) == 0


def test_main_list_models_invalid_config(tmp_path: Path, monkeypatch, capsys) -> None:
    """配置文件损坏：退出码 2 + stderr 错误提示。"""
    home = tmp_path / "home"
    config_dir = home / ".mimcode"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text("broken [ [", encoding="utf-8")
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    assert main(["--list-models"]) == 2
    assert "配置错误" in capsys.readouterr().err


def test_model_entry_table_default() -> None:
    """模型表 None 语义：未声明模型表时不影响 provider 构建。"""
    entry = EndpointEntry(protocol="openai", models=None)
    assert entry.models is None
    entry_with = EndpointEntry(protocol="openai", models={"m": ModelEntry(context_window=1000)})
    assert entry_with.models is not None
    assert entry_with.models["m"].context_window == 1000
