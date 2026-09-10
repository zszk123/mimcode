"""忽略规则匹配（.gitignore/.ignore/.fdignore，对齐 pi utils 的 ignore 语义）。

用 pathspec 实现 gitignore 风格匹配：
- 规则文件从根到条目所在目录逐级累积（下级覆盖上级语义
  由 pathspec 的 negate 规则自然表达）
- 无任何规则文件时全通过
"""

from __future__ import annotations

from pathlib import Path

import pathspec

IGNORE_FILENAMES = (".gitignore", ".ignore", ".fdignore")
"""参与匹配的规则文件名（对齐 pi IGNORE_FILE_NAMES）。"""


def load_ignore_specs(root: Path) -> list[tuple[Path, pathspec.PathSpec]]:
    """从 root 起扫描各级目录的忽略规则文件。

    Returns:
        (规则文件所在目录, spec) 列表，按目录层级排序（浅在前）。
        规则匹配以条目相对该目录的路径进行。
    """
    specs: list[tuple[Path, pathspec.PathSpec]] = []
    if not root.is_dir():
        return specs
    for current_dir in _walk_dirs(root):
        for filename in IGNORE_FILENAMES:
            candidate = current_dir / filename
            if candidate.is_file():
                lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
                # 过滤空行与注释（pathspec 自身可处理，显式过滤减少误配）
                patterns = [line for line in lines if line.strip() and not line.startswith("#")]
                if patterns:
                    specs.append((current_dir, pathspec.GitIgnoreSpec.from_lines(patterns)))
    return specs


def _walk_dirs(root: Path) -> list[Path]:
    """列出 root 及其全部子目录（不含符号环），忽略 .git 目录。"""
    dirs = [root]
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir() and entry.name != ".git" and not entry.is_symlink():
                dirs.append(entry)
                stack.append(entry)
    return sorted(dirs, key=lambda d: len(d.parts))


def is_ignored(path: Path, root: Path, specs: list[tuple[Path, pathspec.PathSpec]]) -> bool:
    """判断相对 root 的条目是否命中任一规则。

    匹配语义：条目路径相对「规则文件所在目录」计算，
    任一层规则命中即忽略；negate（!）规则由 pathspec 处理。
    """
    for spec_dir, spec in specs:
        try:
            relative = path.relative_to(spec_dir)
        except ValueError:
            continue
        posix = relative.as_posix()
        # pathspec 约定：目录规则（尾斜杠）匹配目录条目时需带尾斜杠
        candidates = [posix]
        if path.is_dir():
            candidates.append(posix + "/")
        for candidate in candidates:
            if spec.match_file(candidate):
                return True
        # 目录命中时其下条目也忽略（父目录命中判定）
        for parent in relative.parents:
            parent_posix = parent.as_posix()
            if spec.match_file(parent_posix) or spec.match_file(parent_posix + "/"):
                return True
    return False
