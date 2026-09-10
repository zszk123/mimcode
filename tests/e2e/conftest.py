"""T15 进程级 e2e 共享工具（checklist E2E-1/2/3/4/5）。

faux transport 接线形态（与真实 SDK 的差异，对齐会话纪律标注）：
- 端点在项目级 ``.mimcode/config.toml`` 声明 ``base_url = "faux://<dir>"``，
  registry 构建目录回放 provider（payload 构建、流翻译走真实代码路径，
  仅原始 chunk 源被替换）
- 回放脚本 ``<dir>/fixtures.json`` 按流式调用序轮转；轮转起点 =
  请求记录已有行数，跨进程续接（E2E-2 的 -c 第二进程接第一进程的序）
- 请求 payload 逐行追加 ``<dir>/requests.jsonl``，断言
  「faux 收到的消息序列」读此文件（而非内存 provider.requests）
- 子进程统一 ``PYTHONIOENCODING=utf-8``（Windows 控制台默认 GBK，
  不设则中文流式输出在管道侧被替换/报错）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

CLI_RUNNER = "import sys; from mimcode.cli import main; sys.exit(main(sys.argv[1:]))"
"""子进程 CLI 入口（-c 形态；参数经 sys.argv 透传）。"""

FAUX_KEY_ENV = "FAUX_E2E_KEY"
"""faux 端点的 key 环境变量名（e2e 统一注入）。"""

CLI_TIMEOUT = 90.0
"""子进程超时上限（秒）。"""


def run_cli(
    args: list[str],
    *,
    cwd: Path,
    home: Path,
    timeout: float = CLI_TIMEOUT,
) -> subprocess.CompletedProcess[bytes]:
    """以子进程跑 CLI（print 路径；stdin 管道 → 非 TTY）。

    USERPROFILE 重定向到 home（隔离会话目录与全局配置）。
    """
    env = dict(os.environ)
    env["USERPROFILE"] = str(home)
    env["PYTHONIOENCODING"] = "utf-8"
    env[FAUX_KEY_ENV] = "faux-e2e-key"
    return subprocess.run(
        [sys.executable, "-c", CLI_RUNNER, *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def run_driver(
    script: Path, *args: str, cwd: Path, home: Path
) -> subprocess.CompletedProcess[bytes]:
    """以子进程跑驱动脚本（headless interactive 等场景）。"""
    env = dict(os.environ)
    env["USERPROFILE"] = str(home)
    env["PYTHONIOENCODING"] = "utf-8"
    env[FAUX_KEY_ENV] = "faux-e2e-key"
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        timeout=CLI_TIMEOUT,
        check=False,
    )


def write_faux_endpoint(
    cwd: Path,
    fixtures: list[dict[str, Any]],
    *,
    context_window: int = 200_000,
) -> Path:
    """写项目级 faux 端点配置 + 回放脚本；返回 faux 目录。

    端点名为 faux-test、模型 id 为 faux-model（E2E checklist 约定）。
    """
    faux_dir = cwd / ".mimcode" / "faux"
    faux_dir.mkdir(parents=True, exist_ok=True)
    (faux_dir / "fixtures.json").write_text(
        json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = (
        "[endpoints.faux-test]\n"
        'protocol = "openai"\n'
        f'base_url = "faux://{faux_dir.as_posix()}"\n'
        f'api_key_env = "{FAUX_KEY_ENV}"\n'
        "default = true\n"
        "\n"
        '[endpoints.faux-test.models."faux-model"]\n'
        f"context_window = {context_window}\n"
    )
    config_dir = cwd / ".mimcode"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.toml").write_text(config, encoding="utf-8")
    return faux_dir


def read_requests(faux_dir: Path) -> list[dict[str, Any]]:
    """读请求记录（每次流式调用的 payload 列表）。"""
    path = faux_dir / "requests.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def session_files(home: Path) -> list[Path]:
    """home 下的会话文件列表。"""
    root = home / ".mimcode" / "sessions"
    return sorted(root.rglob("*.jsonl")) if root.is_dir() else []


def read_entries(path: Path) -> list[dict[str, Any]]:
    """会话文件 → JSONL 条目列表。"""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# ---------------------------------------------------------------------------
# openai 协议 chunk 构造（与真实 SDK 流式响应形态同构）
# ---------------------------------------------------------------------------


def text_chunks(text: str) -> list[dict[str, Any]]:
    """纯文本回复 chunk 序列。"""
    return [
        {
            "id": "chatcmpl-e2e-text",
            "model": "faux-model",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}],
        },
        {
            "id": "chatcmpl-e2e-text",
            "model": "faux-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {"id": "chatcmpl-e2e-text", "model": "faux-model", "choices": []},
    ]


def tool_call_chunks(call_id: str, name: str, arguments: str) -> list[dict[str, Any]]:
    """单工具调用 chunk 序列（参数单chunk直发）。"""
    return [
        {
            "id": f"chatcmpl-{call_id}",
            "model": "faux-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": f"chatcmpl-{call_id}",
            "model": "faux-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": arguments}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {"id": f"chatcmpl-{call_id}", "model": "faux-model", "choices": []},
    ]


def fixture(model: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """openai 协议 fixture dict（写入 fixtures.json 的元素）。"""
    return {"protocol": "openai", "model": model, "chunks": chunks}
