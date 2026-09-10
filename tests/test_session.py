"""T7 会话持久化测试。

checklist 对应项：
- 会话目录 ~/.mimcode/sessions/<cwd 编码名>/，文件名含时间戳与会话 id，.jsonl
- append-only：新增消息只追加行，已写入行不被改写（记录行哈希后追加再比对）
- 分叉：fork 后新会话首行以外的条目与源逐行相等，源文件未被修改
- 发现容错：混入损坏 jsonl 与超大文件后列表仍返回其余合法会话
- --continue 语义：恢复最近会话，上下文含上次全部消息
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mimcode.app.session import SessionManager
from mimcode.app.session_store import (
    SessionFormatError,
    append_entry,
    encode_cwd_to_dir_name,
    fork_session_file,
    list_sessions,
    load_session,
    new_session_id,
    session_file_path,
    sessions_root,
)
from mimcode.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage, UserMessage


def make_user(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=1)


def make_assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)],
        api="openai",
        provider="faux-openai",
        model="faux-gpt",
        stop_reason="stop",
        timestamp=2,
    )


def make_tool_call_assistant() -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCallBlock(id="c1", name="echo", arguments={"text": "hi"}, type="toolCall")],
        api="openai",
        provider="faux-openai",
        model="faux-gpt",
        stop_reason="toolUse",
        timestamp=3,
    )


def make_tool_result() -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="c1",
        tool_name="echo",
        content=[TextBlock(text="echo:hi")],
        timestamp=4,
    )


# ---------------------------------------------------------------------------
# 布局与命名
# ---------------------------------------------------------------------------


def test_cwd_encoding(tmp_path: Path) -> None:
    """cwd → 目录名编码（分隔符转 -，-- 包裹）。"""
    encoded = encode_cwd_to_dir_name(str(tmp_path))
    assert encoded.startswith("--") and encoded.endswith("--")
    assert ":" not in encoded
    assert "\\" not in encoded

    root = sessions_root(tmp_path)
    assert root == tmp_path / ".mimcode" / "sessions"


def test_session_file_layout(tmp_path: Path) -> None:
    """目录与文件命名（checklist）。"""
    manager = SessionManager.create(str(tmp_path), home=tmp_path)
    file = manager.session_file
    assert file.parent.parent == sessions_root(tmp_path)
    assert file.suffix == ".jsonl"
    assert manager.session_id in file.name
    # 时间戳前缀（: 与 . 已替换）
    prefix = file.stem.split("_")[0]
    assert ":" not in prefix and "." not in prefix


def test_session_id_validation() -> None:
    """非法会话 id 被拒。"""
    with pytest.raises(SessionFormatError):
        SessionManager.create(".", session_id="bad id with spaces")
    with pytest.raises(SessionFormatError):
        SessionManager.create(".", session_id="-leading-dash")


# ---------------------------------------------------------------------------
# append-only（checklist：行哈希比对）
# ---------------------------------------------------------------------------


async def test_append_only_never_rewrites(tmp_path: Path) -> None:
    """新增消息只追加行；已写入行不改写（记录哈希后追加再比对）。"""
    manager = SessionManager.create(str(tmp_path), home=tmp_path)
    manager.append_message(make_user("第一"))
    manager.append_message(make_assistant("回复一"))
    manager.append_message(make_user("第二"))

    lines_before = manager.session_file.read_text(encoding="utf-8").splitlines()
    hashes_before = [hash(line) for line in lines_before]
    count_before = len(lines_before)

    manager.append_message(make_assistant("回复二"))
    lines_after = manager.session_file.read_text(encoding="utf-8").splitlines()

    assert len(lines_after) == count_before + 1
    assert [hash(line) for line in lines_after[:count_before]] == hashes_before


def test_reopen_sees_all_entries(tmp_path: Path) -> None:
    """重新打开：条目完整还原（消息经 pydantic 往返）。"""
    manager = SessionManager.create(str(tmp_path), home=tmp_path)
    manager.append_message(make_user("hi"))
    manager.append_message(make_tool_call_assistant())
    manager.append_message(make_tool_result())
    manager.append_model_change("deepseek", "deepseek-chat")
    manager.append_thinking_level_change("high")

    reopened = SessionManager.open(manager.session_file)
    context = reopened.build_context()

    assert [message.role for message in context.messages] == [
        "user",
        "assistant",
        "toolResult",
    ]
    assert context.settings.provider == "deepseek"
    assert context.settings.model_id == "deepseek-chat"
    assert context.settings.thinking_level == "high"
    # toolCall 块往返保真
    assistant = context.messages[1]
    assert isinstance(assistant, AssistantMessage)
    block = assistant.content[0]
    assert isinstance(block, ToolCallBlock)
    assert block.arguments == {"text": "hi"}


# ---------------------------------------------------------------------------
# 树与分支
# ---------------------------------------------------------------------------


def test_branch_changes_context(tmp_path: Path) -> None:
    """同文件分支：从历史节点分叉后，上下文切换到分支链。"""
    manager = SessionManager.create(str(tmp_path), home=tmp_path)
    first = manager.append_message(make_user("话题A"))
    manager.append_message(make_assistant("答A"))
    second = manager.append_message(make_user("话题B"))
    manager.append_message(make_assistant("答B"))

    # 默认叶链：A→答A→B→答B
    assert len(manager.build_context().messages) == 4

    # 从「话题B」分叉（模拟重新生成答B）
    manager.branch_from(second)
    manager.append_message(make_assistant("答B-v2"))

    context = manager.build_context()
    texts = []
    for message in context.messages:
        if isinstance(message, UserMessage):
            texts.append(message.content)
        elif isinstance(message, AssistantMessage):
            texts.append(message.content[0].text if message.content else "")
    assert texts == ["话题A", "答A", "话题B", "答B-v2"]
    assert first  # 条目 id 存在


def test_branch_from_unknown_entry_rejected(tmp_path: Path) -> None:
    """未知条目 id 分叉被拒。"""
    manager = SessionManager.create(str(tmp_path), home=tmp_path)
    with pytest.raises(SessionFormatError):
        manager.branch_from("no-such-entry")


def test_path_to_root_explicit_leaf(tmp_path: Path) -> None:
    """显式 leaf：只取该链（旧分支仍可读）。"""
    manager = SessionManager.create(str(tmp_path), home=tmp_path)
    manager.append_message(make_user("A"))
    fork_point = manager.append_message(make_assistant("old"))
    manager.append_message(make_user("B"))
    manager.branch_from(fork_point)
    manager.append_message(make_assistant("new"))

    old_chain = manager.path_to_root(fork_point)
    assert [entry.type for entry in old_chain] == ["message", "message"]

    current = manager.build_context()
    texts = [
        m.content if isinstance(m, UserMessage) else m.content[0].text for m in current.messages
    ]
    # 链 = A → old（分叉点）→ new（branch 锚点不产生消息）
    assert texts == ["A", "old", "new"]


# ---------------------------------------------------------------------------
# fork（checklist：逐行相等 + 源未修改）
# ---------------------------------------------------------------------------


async def test_fork_copies_entries_and_leaves_source_intact(tmp_path: Path) -> None:
    """fork：新会话条目与源逐行相等；源文件未被修改（checklist）。"""
    source = SessionManager.create(str(tmp_path), home=tmp_path)
    source.append_message(make_user("问题"))
    source.append_message(make_assistant("回答"))
    source.append_model_change("zhipu", "glm-4.6")

    source_lines = source.session_file.read_text(encoding="utf-8").splitlines()
    source_hash = hash(source.session_file.read_bytes())

    forked = SessionManager.fork_from(source.session_file, str(tmp_path), home=tmp_path)

    # 源未修改
    assert hash(source.session_file.read_bytes()) == source_hash

    # 新会话：header 不同（新 id、parent_session 指向源），其余条目逐行相等
    forked_lines = forked.session_file.read_text(encoding="utf-8").splitlines()
    assert forked_lines[0] != source_lines[0]
    assert forked_lines[1:] == source_lines[1:]
    assert forked.session_id != source.session_id
    assert forked.cwd == str(tmp_path.resolve())

    # 上下文等价
    assert len(forked.build_context().messages) == 2
    assert forked.build_context().settings.model_id == "glm-4.6"


def test_fork_invalid_source_rejected(tmp_path: Path) -> None:
    """fork 源无效：报错。"""
    bad = tmp_path / "bad.jsonl"
    bad.write_text("", encoding="utf-8")
    with pytest.raises(SessionFormatError, match="empty or invalid"):
        fork_session_file(bad, tmp_path)


# ---------------------------------------------------------------------------
# 发现容错（checklist：损坏 + 超大不阻塞）
# ---------------------------------------------------------------------------


async def test_list_sessions_tolerates_corrupt_and_oversized(tmp_path: Path) -> None:
    """混入损坏与超大文件：列表仍返回其余合法会话（checklist）。"""
    cwd = str(tmp_path / "proj")
    good1 = SessionManager.create(cwd, home=tmp_path)
    good1.append_message(make_user("会话一"))

    good2 = SessionManager.create(cwd, home=tmp_path)
    good2.append_message(make_user("会话二"))
    # 让 good2 更新（后修改时间 → 排序在前）
    import time

    time.sleep(0.05)
    good2.append_message(make_assistant("回复"))

    directory = good1.session_file.parent
    assert directory == good2.session_file.parent

    # 损坏文件（部分行合法 JSON 但整体无 header / 全坏行）
    corrupt = directory / "corrupt.jsonl"
    corrupt.write_text("not json\nalso not json\n", encoding="utf-8")
    # 伪装合法 header 但 body 损坏
    corrupt2 = directory / "corrupt2.jsonl"
    corrupt2.write_text(
        json.dumps({"type": "session", "id": "x", "timestamp": "t", "cwd": "c"}) + "\n{broken\n",
        encoding="utf-8",
    )
    # 超大文件
    oversized = directory / "oversized.jsonl"
    oversized.write_text('{"type":"session","id":"big"}\n' + "x" * 2000, encoding="utf-8")

    sessions = list_sessions(directory, max_bytes=1000)
    ids = {info.id for info in sessions}
    paths = {info.path.name for info in sessions}
    assert good1.session_id in ids
    assert good2.session_id in ids
    # 全坏文件（无合法 header）与超大文件被跳过；
    # corrupt2 有合法 header（坏 body 行被跳过）算空会话——与 pi 一致
    assert "corrupt.jsonl" not in paths
    assert "oversized.jsonl" not in paths
    assert "corrupt2.jsonl" in paths
    # 排序：最近修改在前
    assert sessions[0].id == good2.session_id
    # 列表元数据
    info = next(info for info in sessions if info.id == good1.session_id)
    assert info.first_message == "会话一"
    assert info.message_count == 1


def test_load_session_returns_none_for_invalid(tmp_path: Path) -> None:
    """空文件/无 header/不存在 → None。"""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    no_header = tmp_path / "no_header.jsonl"
    no_header.write_text('{"type":"message"}\n', encoding="utf-8")
    assert load_session(empty) is None
    assert load_session(no_header) is None
    assert load_session(tmp_path / "missing.jsonl") is None


def test_parse_skips_malformed_lines() -> None:
    """解析跳过损坏行（对齐 pi parseSessionEntries）。"""
    from mimcode.app.session_store import parse_session_content

    content = '{"type":"session"}\nbad line\n{"type":"message"}\n\n'
    rows = parse_session_content(content)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# --continue 语义（checklist：最近会话恢复）
# ---------------------------------------------------------------------------


async def test_latest_recovers_most_recent(tmp_path: Path) -> None:
    """latest：恢复最近会话，上下文含全部历史消息（--continue 语义）。"""
    cwd = str(tmp_path / "work")

    first = SessionManager.create(cwd, home=tmp_path)
    first.append_message(make_user("旧会话"))

    import time

    time.sleep(0.05)
    second = SessionManager.create(cwd, home=tmp_path)
    second.append_message(make_user("新会话问题"))
    second.append_message(make_assistant("新会话回答"))

    recovered = SessionManager.latest(cwd, home=tmp_path)
    assert recovered is not None
    assert recovered.session_id == second.session_id

    context = recovered.build_context()
    texts = [
        m.content if isinstance(m, UserMessage) else m.content[0].text for m in context.messages
    ]
    assert texts == ["新会话问题", "新会话回答"]


def test_latest_no_sessions(tmp_path: Path) -> None:
    """无会话：latest 返回 None。"""
    assert SessionManager.latest(str(tmp_path / "empty-cwd"), home=tmp_path) is None


# ---------------------------------------------------------------------------
# 线程池写文件（asyncio.to_thread 路径）
# ---------------------------------------------------------------------------


async def test_concurrent_appends_serialized(tmp_path: Path) -> None:
    """并发追加：行完整性（无交错损坏）。"""
    import asyncio

    manager = SessionManager.create(str(tmp_path), home=tmp_path)

    async def append(index: int) -> None:
        manager.append_message(make_user(f"m{index}"))

    await asyncio.gather(*(append(i) for i in range(10)))

    lines = manager.session_file.read_text(encoding="utf-8").splitlines()
    # header + 10 条消息
    assert len(lines) == 11
    # 每行都是合法 JSON
    for line in lines:
        json.loads(line)
    reopened = SessionManager.open(manager.session_file)
    assert len(reopened.entries) == 10


def test_append_entry_low_level(tmp_path: Path) -> None:
    """底层 append_entry：条目序列化字段保真。"""
    from mimcode.app.session_store import SessionEntry, SessionHeader, write_header

    file = session_file_path(tmp_path, "abc")
    write_header(file, SessionHeader(id="abc", timestamp="t", cwd="c"))
    entry = SessionEntry(
        type="message",
        id="e1",
        parent_id=None,
        timestamp="t",
        message=make_user("x"),
    )
    append_entry(file, entry)
    loaded = load_session(file)
    assert loaded is not None
    assert loaded.entries[0].message is not None
    assert loaded.entries[0].message.role == "user"


def test_entry_id_generation() -> None:
    """条目 id：8 hex 且不冲突。"""
    ids = {new_session_id() for _ in range(50)}
    assert len(ids) == 50
