"""技能加载（迁移自 pi core/skills.ts 的主路径）。

发现规则（对齐 pi loadSkillsFromDir）：
- 目录含 SKILL.md → 视为技能根，加载后不再递归
- 否则加载根目录的直接 .md 子文件（平铺技能）
- 递归子目录寻找 SKILL.md；跳过隐藏条目与忽略命中
- 名称冲突：先注册者胜 + collision 诊断（全局先于项目，对齐 pi 顺序）

校验（对齐 Agent Skills 规范）：
- 名称：≤64 字符、小写字母/数字/连字符、不以连字符开头结尾、无连续连字符
- 描述：必填、≤1024 字符
- 校验失败的技能不加载并记入诊断（与 pi 的差异：pi 仅警告仍加载；
  mimcode 按 checklist 验收语义拒绝，防超长字段破坏提示格式）

注入格式：XML（对齐 pi formatSkillsForPrompt，agentskills.io 标准）；
``disable-model-invocation`` 的技能不注入（仅显式调用，T10 接线）。

与 pi 的实现差异：frontmatter 解析为受限 YAML 子集（标量
``key: value``，含引号剥离与布尔），不引入 PyYAML 依赖。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pathspec

from mimcode.agent.tools.ignore import IGNORE_FILENAMES, is_ignored, load_ignore_specs
from mimcode.config import MIMCODE_DIR

SKILL_FILENAME = "SKILL.md"
"""技能清单文件名（对齐 pi）。"""

MAX_NAME_LENGTH = 64
"""名称长度上限（对齐 pi MAX_NAME_LENGTH）。"""

MAX_DESCRIPTION_LENGTH = 1024
"""描述长度上限（对齐 pi MAX_DESCRIPTION_LENGTH）。"""

_SKILL_NAME_PATTERN = "abcdefghijklmnopqrstuvwxyz0123456789-"


@dataclasses.dataclass(frozen=True)
class Skill:
    """已加载的技能。"""

    name: str
    description: str
    file_path: Path
    base_dir: Path
    source: str  # "user" | "project" | "explicit"
    disable_model_invocation: bool = False


@dataclasses.dataclass(frozen=True)
class SkillDiagnostic:
    """技能加载诊断（warning = 校验失败；collision = 重名）。"""

    type: str  # "warning" | "collision"
    message: str
    path: Path
    winner_path: Path | None = None


@dataclasses.dataclass(frozen=True)
class LoadSkillsResult:
    """技能加载结果。"""

    skills: list[Skill]
    diagnostics: list[SkillDiagnostic]


# ---------------------------------------------------------------------------
# frontmatter 解析（受限 YAML 子集）
# ---------------------------------------------------------------------------


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """解析 ``---`` 分隔的 frontmatter（标量 key: value 子集）。

    Returns:
        (frontmatter 字段表, 正文)；无 frontmatter 时字段表为空。
    """
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if normalized.startswith("\ufeff"):  # BOM
        normalized = normalized[1:]  # BOM
    if not normalized.startswith("---"):
        return {}, normalized
    end = normalized.find("\n---", 3)
    if end == -1:
        return {}, normalized

    yaml_text = normalized[4:end]
    body = normalized[end + 4 :].strip()

    frontmatter: dict[str, Any] = {}
    for line in yaml_text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, _, raw_value = stripped.partition(":")
        frontmatter[key.strip()] = _parse_scalar(raw_value.strip())
    return frontmatter, body


def _parse_scalar(raw: str) -> Any:
    """标量值解析：引号剥离与布尔字面量。"""
    if raw in ("true", "True"):
        return True
    if raw in ("false", "False"):
        return False
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1]
    return raw


# ---------------------------------------------------------------------------
# 校验（对齐 pi validateName / validateDescription）
# ---------------------------------------------------------------------------


def validate_name(name: str) -> list[str]:
    """技能名校验（对齐 Agent Skills 规范）。"""
    errors: list[str] = []
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if any(char not in _SKILL_NAME_PATTERN for char in name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def validate_description(description: object) -> list[str]:
    """技能描述校验。"""
    if not isinstance(description, str) or not description.strip():
        return ["description is required"]
    if len(description) > MAX_DESCRIPTION_LENGTH:
        return [f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})"]
    return []


# ---------------------------------------------------------------------------
# 文件加载
# ---------------------------------------------------------------------------


def _load_skill_from_file(
    file_path: Path, source: str
) -> tuple[Skill | None, list[SkillDiagnostic]]:
    """加载单个技能文件（SKILL.md 或平铺 .md）。

    Returns:
        (技能或 None, 诊断列表)。
    """
    diagnostics: list[SkillDiagnostic] = []
    is_declared = file_path.name == SKILL_FILENAME

    try:
        raw_content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        diagnostics.append(SkillDiagnostic(type="warning", message=str(exc), path=file_path))
        return None, diagnostics

    frontmatter, _body = parse_frontmatter(raw_content)

    description = frontmatter.get("description")
    has_description = isinstance(description, str) and description.strip() != ""
    # 平铺 .md 技能必须自带描述（无目录名兜底语义，对齐 pi）
    if not is_declared and not has_description:
        return None, diagnostics

    parent_name = file_path.parent.name
    frontmatter_name = frontmatter.get("name")
    name = (
        frontmatter_name if isinstance(frontmatter_name, str) and frontmatter_name else parent_name
    )

    # 校验失败 → 拒绝 + 诊断（checklist 语义）
    for error in validate_name(name):
        diagnostics.append(SkillDiagnostic(type="warning", message=error, path=file_path))
    for error in validate_description(description):
        diagnostics.append(SkillDiagnostic(type="warning", message=error, path=file_path))
    if diagnostics:
        return None, diagnostics
    assert isinstance(description, str)

    return (
        Skill(
            name=name,
            description=description,
            file_path=file_path,
            base_dir=file_path.parent,
            source=source,
            disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
        ),
        diagnostics,
    )


def _load_skills_from_dir(
    directory: Path,
    source: str,
    include_root_files: bool,
    specs: list | None = None,
    root_dir: Path | None = None,
) -> LoadSkillsResult:
    """目录发现（对齐 pi loadSkillsFromDirInternal 的递归语义）。"""
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    if not directory.is_dir():
        return LoadSkillsResult(skills, diagnostics)

    root = root_dir or directory
    if specs is None:
        specs = load_ignore_specs(root)

    try:
        entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return LoadSkillsResult(skills, diagnostics)

    # 优先：本目录即技能根（SKILL.md）
    for entry in entries:
        if entry.name != SKILL_FILENAME:
            continue
        if is_ignored(entry, root, specs):
            continue
        skill, file_diagnostics = _load_skill_from_file(entry, source)
        diagnostics.extend(file_diagnostics)
        if skill is not None:
            skills.append(skill)
        return LoadSkillsResult(skills, diagnostics)

    # 否则：子目录递归 + 根平铺 .md
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if is_ignored(entry, root, specs):
            continue

        if entry.is_dir() and not entry.is_symlink():
            sub = _load_skills_from_dir(entry, source, False, specs, root)
            skills.extend(sub.skills)
            diagnostics.extend(sub.diagnostics)
            continue

        if entry.is_file() and include_root_files and entry.name.endswith(".md"):
            skill, file_diagnostics = _load_skill_from_file(entry, source)
            diagnostics.extend(file_diagnostics)
            if skill is not None:
                skills.append(skill)

    return LoadSkillsResult(skills, diagnostics)


def _local_ignore_specs(directory: Path) -> list[tuple[Path, Any]]:
    """读取 directory 本级（不递归）的 ignore 规则文件。"""
    specs: list[tuple[Path, Any]] = []
    if not directory.is_dir():
        return specs
    for filename in IGNORE_FILENAMES:
        candidate = directory / filename
        if not candidate.is_file():
            continue
        lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
        patterns = [line for line in lines if line.strip() and not line.startswith("#")]
        if patterns:
            specs.append((directory, pathspec.GitIgnoreSpec.from_lines(patterns)))
    return specs


# ---------------------------------------------------------------------------
# 顶层装配（全局 + 项目两级，对齐 pi loadSkills）
# ---------------------------------------------------------------------------


def load_skills(cwd: str, home: Path | None = None) -> LoadSkillsResult:
    """加载全局与项目两级技能（先全局后项目；重名先注册者胜）。

    Args:
        cwd: 项目目录。
        home: 用户主目录（测试注入）。

    Returns:
        合并后的技能与诊断。
    """
    user_dir = (home or Path.home()) / ".mimcode" / "skills"
    project_root = Path(cwd).resolve()
    project_dir = project_root / MIMCODE_DIR / "skills"

    by_name: dict[str, Skill] = {}
    diagnostics: list[SkillDiagnostic] = []

    # project 级规则 = project 根的 .gitignore/.ignore/.fdignore +
    # skills 树内各级规则（条目路径相对规则目录计算，git 的
    # 无斜杠规则天然匹配任意层级目录名）
    project_specs = _local_ignore_specs(project_root) + load_ignore_specs(project_dir)
    for directory, source, specs in (
        (user_dir, "user", load_ignore_specs(user_dir)),
        (project_dir, "project", project_specs),
    ):
        result = _load_skills_from_dir(directory, source, True, specs)
        diagnostics.extend(result.diagnostics)
        for skill in result.skills:
            existing = by_name.get(skill.name)
            if existing is not None:
                diagnostics.append(
                    SkillDiagnostic(
                        type="collision",
                        message=f'name "{skill.name}" collision',
                        path=skill.file_path,
                        winner_path=existing.file_path,
                    )
                )
            else:
                by_name[skill.name] = skill

    return LoadSkillsResult(list(by_name.values()), diagnostics)


# ---------------------------------------------------------------------------
# 系统提示注入（对齐 pi formatSkillsForPrompt）
# ---------------------------------------------------------------------------


def escape_xml(text: str) -> str:
    """XML 特殊字符转义（对齐 pi escapeXml）。"""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_skills_for_prompt(skills: list[Skill]) -> str:
    """技能清单 → 系统提示片段（XML 格式，agentskills.io 标准）。

    ``disable_model_invocation`` 的技能不注入（对齐 pi）。
    """
    visible = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible:
        return ""

    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill "
        "directory (parent of SKILL.md / dirname of the path) and use that absolute "
        "path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{escape_xml(str(skill.file_path))}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)
