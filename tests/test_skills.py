"""T9 skills 测试。

checklist 对应项：
- SKILL.md 校验：名称 > 64 字符或描述 > 1024 字符时该技能被拒并进入
  诊断列表（两个边界各 1 例）
- .gitignore 命中的技能目录不参与发现
- 格式化注入：加载 2 个技能后系统提示含两个技能名
"""

from __future__ import annotations

from pathlib import Path

from mimcode.app.skills import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    format_skills_for_prompt,
    load_skills,
    parse_frontmatter,
    validate_description,
    validate_name,
)


def write_skill(directory: Path, name: str, description: str, *, body: str = "技能正文") -> Path:
    """写入一个技能目录（SKILL.md）。"""
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    file = skill_dir / "SKILL.md"
    file.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return file


def write_user_skill(home: Path, name: str, description: str) -> Path:
    return write_skill(home / ".mimcode" / "skills", name, description)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------


def test_validate_name_boundaries() -> None:
    """名称校验：合法形态与各类违规。"""
    assert validate_name("pdf-tools") == []
    assert validate_name("a") == []
    assert validate_name("data-v2") == []

    # 超长（>64）
    long_name = "a" * (MAX_NAME_LENGTH + 1)
    errors = validate_name(long_name)
    assert any("exceeds" in error for error in errors)

    # 大写非法
    assert any("invalid characters" in error for error in validate_name("My-Skill"))
    # 下划线非法
    assert any("invalid characters" in error for error in validate_name("my_skill"))
    # 连字符开头/结尾
    assert any("start or end" in error for error in validate_name("-lead"))
    assert any("start or end" in error for error in validate_name("trail-"))
    # 连续连字符
    assert any("consecutive" in error for error in validate_name("double--dash"))


def test_validate_description_boundaries() -> None:
    """描述校验：必填与长度上限。"""
    assert validate_description("做事说明") == []
    assert validate_description(None) == ["description is required"]
    assert validate_description("   ") == ["description is required"]
    long_description = "x" * (MAX_DESCRIPTION_LENGTH + 1)
    errors = validate_description(long_description)
    assert len(errors) == 1
    assert "exceeds" in errors[0]


# ---------------------------------------------------------------------------
# 发现与加载
# ---------------------------------------------------------------------------


def test_load_skills_two_levels(tmp_path: Path) -> None:
    """两级发现：全局 + 项目技能都加载。"""
    home = tmp_path / "home"
    project = tmp_path / "proj"
    write_user_skill(home, "global-skill", "全局技能描述")
    write_skill(project / ".mimcode" / "skills", "project-skill", "项目技能描述")

    result = load_skills(str(project), home=home)
    names = {skill.name for skill in result.skills}
    assert names == {"global-skill", "project-skill"}
    sources = {skill.name: skill.source for skill in result.skills}
    assert sources["global-skill"] == "user"
    assert sources["project-skill"] == "project"


def test_skill_name_defaults_to_directory(tmp_path: Path) -> None:
    """frontmatter 缺 name 时回落目录名（对齐 pi）。"""
    skills_dir = tmp_path / "home" / ".mimcode" / "skills"
    skill_dir = skills_dir / "dir-named-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\ndescription: 只有描述\n---\n正文\n", encoding="utf-8")
    result = load_skills(str(tmp_path / "proj"), home=tmp_path / "home")
    assert [skill.name for skill in result.skills] == ["dir-named-skill"]


def test_name_too_long_rejected_with_diagnostic(tmp_path: Path) -> None:
    """checklist：名称 > 64 → 拒绝 + 诊断。"""
    home = tmp_path / "home"
    long_name = "a" * (MAX_NAME_LENGTH + 1)
    write_user_skill(home, long_name, "描述")

    result = load_skills(str(tmp_path / "proj"), home=home)
    assert result.skills == []
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.type == "warning"
    assert "exceeds" in diagnostic.message


def test_description_too_long_rejected_with_diagnostic(tmp_path: Path) -> None:
    """checklist：描述 > 1024 → 拒绝 + 诊断。"""
    home = tmp_path / "home"
    write_user_skill(home, "ok-name", "x" * (MAX_DESCRIPTION_LENGTH + 1))

    result = load_skills(str(tmp_path / "proj"), home=home)
    assert result.skills == []
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.type == "warning"
    assert "exceeds" in diagnostic.message


def test_gitignored_skill_dir_not_discovered(tmp_path: Path) -> None:
    """checklist：.gitignore 命中的技能目录不参与发现。"""
    project = tmp_path / "proj"
    skills_dir = project / ".mimcode" / "skills"
    project.mkdir(parents=True, exist_ok=True)
    (project / ".gitignore").write_text("secret-skill/\n", encoding="utf-8")

    write_skill(skills_dir, "visible-skill", "可见技能")
    write_skill(skills_dir, "secret-skill", "被忽略的技能")

    result = load_skills(str(project), home=tmp_path / "nowhere-home")
    names = {skill.name for skill in result.skills}
    assert names == {"visible-skill"}


def test_flat_md_skill_requires_description(tmp_path: Path) -> None:
    """平铺 .md 技能：必须有描述（无目录名兜底，对齐 pi）。"""
    skills_dir = tmp_path / "home" / ".mimcode" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "with-desc.md").write_text(
        "---\nname: flat-one\ndescription: 平铺技能\n---\n正文\n", encoding="utf-8"
    )
    (skills_dir / "no-desc.md").write_text("只有正文无 frontmatter\n", encoding="utf-8")

    result = load_skills(str(tmp_path / "proj"), home=tmp_path / "home")
    assert [skill.name for skill in result.skills] == ["flat-one"]


def test_skill_root_not_recursed(tmp_path: Path) -> None:
    """SKILL.md 目录视为技能根：内部子目录不再扫描。"""
    skills_dir = tmp_path / "home" / ".mimcode" / "skills"
    outer = skills_dir / "outer-skill"
    write_skill(skills_dir, "outer-skill", "外层技能")
    write_skill(outer, "inner-skill", "内层技能（不应被发现）")

    result = load_skills(str(tmp_path / "proj"), home=tmp_path / "home")
    names = {skill.name for skill in result.skills}
    assert names == {"outer-skill"}


def test_name_collision_first_wins(tmp_path: Path) -> None:
    """重名冲突：先注册者胜 + collision 诊断（全局先于项目，对齐 pi）。"""
    home = tmp_path / "home"
    project = tmp_path / "proj"
    winner = write_user_skill(home, "same-name", "全局版本")
    loser = write_skill(project / ".mimcode" / "skills", "same-name", "项目版本")

    result = load_skills(str(project), home=home)
    assert len(result.skills) == 1
    skill = result.skills[0]
    assert skill.source == "user"
    assert skill.file_path == winner

    collisions = [d for d in result.diagnostics if d.type == "collision"]
    assert len(collisions) == 1
    assert collisions[0].path == loser
    assert collisions[0].winner_path == winner


def test_missing_description_rejected(tmp_path: Path) -> None:
    """SKILL.md 无描述：拒绝 + 诊断。"""
    skills_dir = tmp_path / "home" / ".mimcode" / "skills"
    skill_dir = skills_dir / "no-desc-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: no-desc-skill\n---\n正文\n", encoding="utf-8")

    result = load_skills(str(tmp_path / "proj"), home=tmp_path / "home")
    assert result.skills == []
    assert any(
        d.type == "warning" and "description is required" in d.message for d in result.diagnostics
    )


def test_empty_and_missing_dirs(tmp_path: Path) -> None:
    """目录不存在 / 为空：空结果无诊断。"""
    result = load_skills(str(tmp_path / "proj"), home=tmp_path / "home")
    assert result.skills == []
    assert result.diagnostics == []


# ---------------------------------------------------------------------------
# 提示注入（checklist：2 个技能 → 系统提示含两个技能名）
# ---------------------------------------------------------------------------


def test_format_skills_for_prompt_two_skills(tmp_path: Path) -> None:
    """checklist：两个技能注入后提示含两个技能名。"""
    home = tmp_path / "home"
    write_user_skill(home, "pdf-tools", "处理 PDF 文档")
    write_user_skill(home, "web-search", "联网搜索资料")

    result = load_skills(str(tmp_path / "proj"), home=home)
    prompt = format_skills_for_prompt(result.skills)

    assert "<available_skills>" in prompt
    assert "</available_skills>" in prompt
    assert "<name>pdf-tools</name>" in prompt
    assert "<name>web-search</name>" in prompt
    assert "处理 PDF 文档" in prompt
    assert "SKILL.md" in prompt  # location 含文件路径


def test_format_skills_empty() -> None:
    """无技能：注入片段为空字符串。"""
    assert format_skills_for_prompt([]) == ""


def test_format_skills_excludes_disabled(tmp_path: Path) -> None:
    """disable-model-invocation 的技能不注入。"""
    home = tmp_path / "home"
    skills_dir = home / ".mimcode" / "skills"
    normal = write_skill(skills_dir, "normal-skill", "普通技能")
    hidden_dir = skills_dir / "hidden-skill"
    hidden_dir.mkdir(parents=True)
    (hidden_dir / "SKILL.md").write_text(
        "---\n"
        "name: hidden-skill\n"
        "description: 仅显式调用\n"
        "disable-model-invocation: true\n"
        "---\n正文\n",
        encoding="utf-8",
    )

    result = load_skills(str(tmp_path / "proj"), home=home)
    assert len(result.skills) == 2
    prompt = format_skills_for_prompt(result.skills)
    assert "normal-skill" in prompt
    assert "hidden-skill" not in prompt
    assert normal.is_file()


def test_format_skills_escapes_xml() -> None:
    """描述中的 XML 特殊字符被转义。"""
    from mimcode.app.skills import Skill

    skill = Skill(
        name="escape-test",
        description='含 <标签> & "引号"',
        file_path=Path("/tmp/s/SKILL.md"),
        base_dir=Path("/tmp/s"),
        source="user",
    )
    prompt = format_skills_for_prompt([skill])
    assert "<标签>" not in prompt
    assert "&lt;标签&gt;" in prompt
    assert "&amp;" in prompt
    assert "&quot;引号&quot;" in prompt


# ---------------------------------------------------------------------------
# frontmatter 解析
# ---------------------------------------------------------------------------


def test_parse_frontmatter_forms() -> None:
    """frontmatter 解析：标准/引号/布尔/缺失。"""
    front, body = parse_frontmatter("---\nname: x\ndescription: 描述\n---\n\n正文内容\n")
    assert front == {"name": "x", "description": "描述"}
    assert body == "正文内容"

    quoted = parse_frontmatter('---\nname: "quoted"\n---\nbody')
    assert quoted[0]["name"] == "quoted"

    flag = parse_frontmatter("---\ndisable-model-invocation: true\n---\nb")
    assert flag[0]["disable-model-invocation"] is True

    none = parse_frontmatter("无 frontmatter 的正文")
    assert none[0] == {}
    assert none[1] == "无 frontmatter 的正文"

    unclosed = parse_frontmatter("---\nname: x\n未闭合")
    assert unclosed[0] == {}
