"""mimcode 命令行入口（T14 全参数接线）。

模式分发（对齐 pi main.ts L109-120）：
- ``--list-models``/``--version``/``--help``：直出
- ``-p``/非 TTY：print 模式
- TTY：interactive 模式
- 会话 flag（-c/--fork/--session）在两种模式下语义一致
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mimcode import __version__


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器（对齐 pi 参数语义）。"""
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
        choices=["off", "minimal", "low", "medium", "high"],
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
    """CLI 主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        from mimcode.cli.list_models import list_models
        from mimcode.config import load_config
        from mimcode.provider.registry import build_registry

        try:
            config = load_config()
        except ValueError as exc:
            print(f"配置错误: {exc}", file=sys.stderr)
            return 2
        registry = build_registry(config)
        list_models(registry)
        return 0

    prompt = " ".join(args.prompt).strip() if args.prompt else ""

    if args.print_mode or prompt or not sys.stdin.isatty():
        # print 模式（-p / 带提示词 / 非 TTY，对齐 pi）
        if not prompt:
            print('print 模式需要提示词（mimcode -p "提示词"）', file=sys.stderr)
            return 2
        return _run_print(args, prompt)
    return _run_interactive(args, prompt)


def _run_print(args: argparse.Namespace, prompt: str) -> int:
    """print 模式执行。"""
    import asyncio

    from mimcode.app.agent_session import AgentSession, AgentSessionError

    cwd = str(Path.cwd())

    async def _execute() -> int:
        session = AgentSession(
            cwd=cwd,
            model_spec=args.model,
            thinking_level=args.thinking,
            continue_session=args.continue_session,
            fork=args.fork,
            session_id=args.session,
        )
        from mimcode.modes.print_mode import run_print_mode

        return await run_print_mode(session, prompt)

    try:
        return asyncio.run(_execute())
    except AgentSessionError as exc:
        print(f"会话错误: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2


def _run_interactive(args: argparse.Namespace, prompt: str) -> int:
    """interactive 模式执行。"""
    import asyncio

    from mimcode.app.agent_session import AgentSession, AgentSessionError
    from mimcode.tui.app import InteractiveApp

    cwd = str(Path.cwd())

    async def _execute() -> int:
        session = AgentSession(
            cwd=cwd,
            model_spec=args.model,
            thinking_level=args.thinking,
            continue_session=args.continue_session,
            fork=args.fork,
            session_id=args.session,
        )
        agent_context = session.build_agent_context()
        app = InteractiveApp(
            cwd=cwd,
            agent_context=agent_context,
            config_factory=lambda: session.make_loop_config(),
            command_context_factory=session.command_context,
        )
        return await app.run_async()

    try:
        return asyncio.run(_execute())
    except AgentSessionError as exc:
        print(f"会话错误: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    del prompt
