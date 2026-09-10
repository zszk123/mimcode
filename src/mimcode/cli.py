"""mimcode 命令行入口。

T1 阶段仅负责参数声明与 --help/--version；
其余参数已按 pi 的语义占位（pi: src/main.ts L109-120 的模式分发语义），
后续任务（T4/T7/T14）逐步接入实际行为。
"""

from __future__ import annotations

import argparse
import sys

from mimcode import __version__


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。

    参数语义对齐 pi：
    - ``-p``           print 模式：一次性执行提示词并输出结果后退出
    - ``-c/--continue`` 恢复当前目录最近会话
    - ``--fork``       从最近（或 --session 指定）会话分叉新会话
    - ``--session``    指定会话 ID（配合 -c/--fork 使用）
    - ``--model``      指定模型（形如 provider/model 或裸模型 id）
    - ``--thinking``   思考级别（off/low/medium/high）
    - ``--list-models`` 列出可用模型目录后退出
    """
    parser = argparse.ArgumentParser(
        prog="mimcode",
        description="mimcode - 终端 AI 编程助手（pi 的 Python 迁移版）",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="初始提示词（多段文本以空格拼接）",
    )
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        dest="print_mode",
        help="print 模式：一次性执行提示词并输出结果",
    )
    parser.add_argument(
        "-c",
        "--continue",
        action="store_true",
        dest="continue_session",
        help="恢复当前目录最近会话",
    )
    parser.add_argument(
        "--fork",
        action="store_true",
        help="从最近（或 --session 指定）会话分叉新会话",
    )
    parser.add_argument(
        "--session",
        metavar="ID",
        default=None,
        help="指定会话 ID（配合 -c/--fork 使用）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="指定模型（如 openai/gpt-5 或 deepseek-chat）",
    )
    parser.add_argument(
        "--thinking",
        default=None,
        choices=["off", "low", "medium", "high"],
        help="思考级别（默认跟随模型能力）",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出可用模型目录后退出",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mimcode {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。

    Args:
        argv: 命令行参数列表；``None`` 时取 ``sys.argv[1:]``。

    Returns:
        进程退出码。T1 阶段 ``--help``/``--version`` 可用，
        其余模式统一返回 2 并提示接入的任务编号。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # --version 由 argparse 的 version 动作直接处理，不会执行到这里
    if args.list_models:
        print("尚未实现：模型目录（T4 接入）", file=sys.stderr)
        return 2
    if args.print_mode or args.prompt:
        print("尚未实现：print 模式（T14 接入）", file=sys.stderr)
        return 2
    if args.continue_session or args.fork or args.session:
        print("尚未实现：会话管理（T7/T14 接入）", file=sys.stderr)
        return 2

    print("尚未实现：交互模式（T12 接入）", file=sys.stderr)
    return 2
