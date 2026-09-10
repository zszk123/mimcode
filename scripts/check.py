"""mimcode 单命令质量门禁。

等价于 pi 仓库的 ``npm run check``：一条命令依次完成
ruff（lint + format 检查）→ mypy（类型检查）→ pytest（单元测试），
任一环节失败立即终止并返回非零退出码。

用法::

    uv run python scripts/check.py            # 全部门禁（含 e2e）
    uv run python scripts/check.py --fast     # 跳过 e2e（pytest -m "not e2e"）
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# 项目根目录（scripts/ 的上一级）
ROOT = Path(__file__).resolve().parent.parent


def run_step(name: str, command: list[str]) -> bool:
    """执行单个检查步骤。

    Args:
        name: 步骤名称（用于终端提示）。
        command: 以 ``sys.executable -m`` 形式调用，确保使用当前 venv 的工具。

    Returns:
        步骤是否成功（退出码为 0）。
    """
    print(f"==> {name}: {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT)
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    """门禁主入口：全部步骤通过返回 0，任一失败返回 1。

    Args:
        argv: 命令行参数（--fast 跳过 e2e 测试）。
    """
    fast = "--fast" in (argv if argv is not None else sys.argv[1:])
    pytest_command = [sys.executable, "-m", "pytest"]
    if fast:
        pytest_command += ["-m", "not e2e"]
    steps: list[tuple[str, list[str]]] = [
        ("ruff lint", [sys.executable, "-m", "ruff", "check", "."]),
        ("ruff format", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        ("mypy", [sys.executable, "-m", "mypy"]),
        ("pytest", pytest_command),
    ]
    for name, command in steps:
        if not run_step(name, command):
            print(f"check 失败于: {name}", file=sys.stderr)
            return 1
    print("check 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
