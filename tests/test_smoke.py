"""T1 冒烟测试：包可导入、CLI 参数声明与 --version/--help 行为。"""

import re

import pytest

import mimcode
from mimcode.cli import build_parser, main

# checklist 要求 --help 输出必须出现的关键参数
REQUIRED_FLAGS = ["-p", "--model", "--list-models", "-c", "--fork"]


def test_version_format() -> None:
    """版本号符合语义化版本格式（mimcode X.Y.Z）。"""
    assert re.fullmatch(r"\d+\.\d+\.\d+", mimcode.__version__)


def test_help_exits_zero() -> None:
    """--help 退出码为 0。"""
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--help"])
    assert exc_info.value.code == 0


@pytest.mark.parametrize("flag", REQUIRED_FLAGS)
def test_help_text_mentions_flags(flag: str, capsys: pytest.CaptureFixture[str]) -> None:
    """帮助文本逐项包含 -p/--model/--list-models/-c/--fork。"""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    assert flag in capsys.readouterr().out


def test_main_version(capsys: pytest.CaptureFixture[str]) -> None:
    """main(['--version']) 输出形如 mimcode 0.1.0 且退出码 0。"""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert f"mimcode {mimcode.__version__}" in capsys.readouterr().out


def test_main_unimplemented_paths_exit_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """未实现模式统一返回非零退出码，且 stderr 有提示而非崩溃。"""
    assert main(["-p", "你好"]) == 2
    assert main(["--list-models"]) == 2
    assert main([]) == 2
    assert "尚未实现" in capsys.readouterr().err


def test_parse_accepts_combined_flags() -> None:
    """-p 与 -c/--session 等参数可被解析并存（语义接线在后续任务）。"""
    args = build_parser().parse_args(
        ["-p", "-c", "--session", "abc", "--model", "deepseek-chat", "--thinking", "high", "你好"],
    )
    assert args.print_mode is True
    assert args.continue_session is True
    assert args.session == "abc"
    assert args.model == "deepseek-chat"
    assert args.thinking == "high"
    assert args.prompt == ["你好"]
