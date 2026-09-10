"""插件加载器：目录扫描 + register(ctx) 约定 + 故障隔离（对齐 pi loader.ts 角色）。

扫描位置（checklist）：
- 全局：``~/.mimcode/extensions/``
- 项目：``<cwd>/.mimcode/extensions/``

约定：目录内每个 ``*.py`` 文件是一个插件，须暴露模块级
``register(ctx: ExtensionContext)``。

故障隔离（checklist）：单个插件加载失败（语法错误/导入错误/
register 抛异常/无 register 函数）→ 记入诊断、跳过该插件，
不影响其余插件与启动流程。插件文件以唯一模块名导入
（importlib.util.spec_from_file_location），不污染 sys.modules
命名空间（同文件名共存于两级目录）。

安全边界：插件代码即用户代码（与 pi TS 插件同级信任）；
项目级插件在首次加载时由 T14 首次启动引导提示确认（v1 语义，
trust 流程在 T14 接线）。
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from mimcode.app.extensions.types import ExtensionContext
from mimcode.config import MIMCODE_DIR

EXTENSIONS_DIRNAME = "extensions"
"""两级目录下的插件目录名。"""


@dataclasses.dataclass(frozen=True)
class LoadedExtension:
    """已加载的插件。"""

    name: str
    source: str  # "user" | "project"
    file_path: Path
    context: ExtensionContext


@dataclasses.dataclass(frozen=True)
class ExtensionDiagnostic:
    """插件加载诊断（失败/告警）。"""

    type: str  # "warning" | "error"
    extension: str
    message: str
    file_path: Path


@dataclasses.dataclass(frozen=True)
class LoadExtensionsResult:
    """插件加载结果。"""

    extensions: list[LoadedExtension]
    diagnostics: list[ExtensionDiagnostic]


def _import_plugin_module(file_path: Path, unique_name: str) -> ModuleType:
    """以独立模块名导入插件文件（不入 sys.modules 全局表）。"""
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {file_path} 创建模块规格")
    module = importlib.util.module_from_spec(spec)
    # 插件相对导入自洽：临时挂表（加载完即摘）
    sys.modules[unique_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(unique_name, None)
    return module


def _load_single_plugin(
    file_path: Path, source: str, index: int
) -> tuple[LoadedExtension | None, ExtensionDiagnostic | None]:
    """加载单个插件文件。

    Returns:
        (插件或 None, 失败诊断或 None)。
    """
    name = file_path.stem
    # 唯一模块名：防两级目录同名插件冲突
    unique_name = f"_mimcode_ext_{source}_{index}_{name}"
    try:
        module = _import_plugin_module(file_path, unique_name)
    except Exception as exc:  # noqa: BLE001 - 插件加载失败的统一隔离边界
        return None, ExtensionDiagnostic(
            type="error",
            extension=name,
            message=f"导入失败: {exc}",
            file_path=file_path,
        )

    register = getattr(module, "register", None)
    if not callable(register):
        return None, ExtensionDiagnostic(
            type="warning",
            extension=name,
            message="缺少模块级 register(ctx) 函数",
            file_path=file_path,
        )

    context = ExtensionContext(name)
    try:
        register(context)
    except Exception as exc:  # noqa: BLE001 - register 异常隔离（不中断其余插件）
        return None, ExtensionDiagnostic(
            type="error",
            extension=name,
            message=f"register() 抛出异常: {exc}",
            file_path=file_path,
        )

    # 基本健全性：注册面非空（或至少没有空跑失败）——空注册也允许
    # （纯日志型插件），但 ctx 必须是本插件自己的
    if context.name != name:
        return None, ExtensionDiagnostic(
            type="error",
            extension=name,
            message="register(ctx) 收到的上下文不一致",
            file_path=file_path,
        )
    return LoadedExtension(name=name, source=source, file_path=file_path, context=context), None


def _scan_dir(
    directory: Path, source: str, start_index: int
) -> tuple[list[LoadedExtension], list[ExtensionDiagnostic], int]:
    """扫描一个插件目录（按文件名排序，保证加载顺序稳定）。"""
    extensions: list[LoadedExtension] = []
    diagnostics: list[ExtensionDiagnostic] = []
    if not directory.is_dir():
        return extensions, diagnostics, start_index

    files = sorted(
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix == ".py" and not entry.name.startswith("_")
    )
    index = start_index
    for file_path in files:
        loaded, diagnostic = _load_single_plugin(file_path, source, index)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
        if loaded is not None:
            extensions.append(loaded)
        index += 1
    return extensions, diagnostics, index


def load_extensions(cwd: str, home: Path | None = None) -> LoadExtensionsResult:
    """加载全局与项目两级插件（先全局后项目）。

    Args:
        cwd: 项目目录。
        home: 用户主目录（测试注入）。

    Returns:
        全部加载成功的插件与诊断（失败插件被隔离）。
    """
    user_dir = (home or Path.home()) / ".mimcode" / EXTENSIONS_DIRNAME
    project_dir = Path(cwd) / MIMCODE_DIR / EXTENSIONS_DIRNAME

    all_extensions: list[LoadedExtension] = []
    all_diagnostics: list[ExtensionDiagnostic] = []
    next_index = 0
    for directory, source in ((user_dir, "user"), (project_dir, "project")):
        extensions, diagnostics, next_index = _scan_dir(directory, source, next_index)
        all_extensions.extend(extensions)
        all_diagnostics.extend(diagnostics)

    return LoadExtensionsResult(extensions=all_extensions, diagnostics=all_diagnostics)


def apply_extensions(
    result: LoadExtensionsResult,
    *,
    tool_registry: Any,
    command_registry: Any,
) -> dict[str, list]:
    """把插件注册面挂接到运行时注册表。

    Args:
        result: load_extensions 结果。
        tool_registry: agent 工具注册表（register(tool)）。
        command_registry: slash 命令注册表（register(SlashCommand)）。

    Returns:
        {"tools": [...], "commands": [...], "event_handlers": [...]}
        （T12/T14 事件循环接线的输入）。
    """
    applied_tools: list = []
    applied_commands: list = []
    applied_handlers: list = []

    for extension in result.extensions:
        for tool in extension.context.tools:
            tool_registry.register(tool, replace=True)  # 插件工具覆盖内置（后注册胜）
            applied_tools.append(tool)
        for command in extension.context.commands:
            command_registry.register(command)
            applied_commands.append(command)
        applied_handlers.extend(extension.context.event_handlers)

    return {
        "tools": applied_tools,
        "commands": applied_commands,
        "event_handlers": applied_handlers,
    }


def extension_names(result: LoadExtensionsResult) -> list[str]:
    """已加载插件名清单（/extensions 与 CommandContext.skill 同源）。"""
    return [extension.name for extension in result.extensions]
