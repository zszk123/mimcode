"""会话存储层：JSONL append-only 树的文件操作（对齐 pi session-manager.ts 的存储职责）。

文件布局（checklist）：
- 目录：``~/.mimcode/sessions/--<编码后的-cwd>--/``
- 文件名：``<ISO时间戳(:.→-)>_<会话id>.jsonl``

JSONL 结构（对齐 pi FileEntry 形态，snake_case）：
- 首行 header：``{"type": "session", "version": 1, "id", "timestamp", "cwd", "parent_session"?}``
- 其后每行一个树节点：``{"type", "id", "parent_id", "timestamp", ...}``
  - ``message``：携带 AgentMessage（v1 的主要载荷）
  - ``model_change`` / ``thinking_level_change``：设置变更（恢复会话时还原）
  - ``compaction``：T8 接入（占位识别）

树语义：``id``/``parent_id`` 链；leaf = 最后追加的条目（默认上下文走
leaf→root 回溯）；同文件分叉 = 新条目的 parent_id 指向已有条目。

发现容错（对齐 pi）：损坏行跳过、空文件/无 header 跳过、
超大文件跳过（不阻塞其他会话的发现）。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from mimcode.types import AgentMessage

_MESSAGE_ADAPTER: TypeAdapter[AgentMessage] = TypeAdapter(AgentMessage)
"""AgentMessage 判别联合的验证适配器（会话消息行还原）。"""

SESSION_VERSION = 1
"""mimcode 会话格式版本（从 1 起步；pi 当前为 3，演进历史不迁移）。"""

MAX_SESSION_FILE_BYTES = 50 * 1024 * 1024
"""发现时的文件大小上限（超大文件不是会话，对齐 pi 容错语义）。"""

_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


class SessionFormatError(Exception):
    """会话文件格式错误（fork 源无效等）。"""


def new_session_id() -> str:
    """新会话 id（uuid4 hex，对齐 pi 的 uuid 角色）。"""
    return uuid.uuid4().hex


def new_entry_id() -> str:
    """新条目 id（短 hex，对齐 pi generateId）。"""
    return uuid.uuid4().hex[:8]


def assert_valid_session_id(session_id: str) -> None:
    """校验会话 id 字符集（对齐 pi assertValidSessionId）。

    Raises:
        SessionFormatError: id 含非法字符。
    """
    if not _SESSION_ID_PATTERN.match(session_id):
        raise SessionFormatError(
            "Session id must be non-empty, contain only alphanumeric characters, "
            "'-', '_', and '.', and start and end with an alphanumeric character"
        )


def encode_cwd_to_dir_name(cwd: str) -> str:
    """cwd → 安全目录名（对齐 pi：``--<路径分隔符替换为->--``）。"""
    resolved = str(Path(cwd).resolve())
    stripped = resolved.replace("\\", "/").lstrip("/")
    safe = stripped.replace("/", "-").replace(":", "-")
    return f"--{safe}--"


def sessions_root(home: Path | None = None) -> Path:
    """会话根目录：``~/.mimcode/sessions``。"""
    return (home or Path.home()) / ".mimcode" / "sessions"


def session_dir_for(cwd: str, home: Path | None = None) -> Path:
    """cwd 对应的会话目录（不存在时创建）。"""
    directory = sessions_root(home) / encode_cwd_to_dir_name(cwd)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _file_timestamp(moment: datetime | None = None) -> str:
    """文件名用时间戳（ISO 格式，:. 替换为 -，对齐 pi）。"""
    now = moment or datetime.now(UTC)
    return now.isoformat().replace(":", "-").replace(".", "-")


def session_file_path(session_dir: Path, session_id: str, moment: datetime | None = None) -> Path:
    """会话文件路径：``<时间戳>_<id>.jsonl``。"""
    return session_dir / f"{_file_timestamp(moment)}_{session_id}.jsonl"


# ---------------------------------------------------------------------------
# 条目模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionHeader:
    """会话首行。"""

    type: str = "session"
    version: int = SESSION_VERSION
    id: str = ""
    timestamp: str = ""
    cwd: str = ""
    parent_session: str | None = None


@dataclass(frozen=True)
class SessionEntry:
    """树节点条目（v1：message / model_change / thinking_level_change / compaction）。"""

    type: str
    id: str
    parent_id: str | None
    timestamp: str

    message: AgentMessage | None = None
    """type=message 时的载荷。"""

    provider: str | None = None
    model_id: str | None = None
    """type=model_change。"""

    thinking_level: str | None = None
    """type=thinking_level_change。"""

    extra: dict[str, Any] = field(default_factory=dict)
    """compaction 等类型的原样字段（T8 使用）。"""


def header_to_dict(header: SessionHeader) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": "session",
        "version": header.version,
        "id": header.id,
        "timestamp": header.timestamp,
        "cwd": header.cwd,
    }
    if header.parent_session is not None:
        data["parent_session"] = header.parent_session
    return data


def entry_to_dict(entry: SessionEntry) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": entry.type,
        "id": entry.id,
        "parent_id": entry.parent_id,
        "timestamp": entry.timestamp,
    }
    if entry.message is not None:
        data["message"] = entry.message.model_dump()
    if entry.provider is not None:
        data["provider"] = entry.provider
    if entry.model_id is not None:
        data["model_id"] = entry.model_id
    if entry.thinking_level is not None:
        data["thinking_level"] = entry.thinking_level
    data.update(entry.extra)
    return data


def _entry_from_dict(data: dict[str, Any]) -> SessionEntry | None:
    """dict → SessionEntry（message 载荷经 pydantic 还原；失败返回 None）。"""
    entry_type = data.get("type")
    if entry_type is None or entry_type == "session":
        return None
    message: AgentMessage | None = None
    raw_message = data.get("message")
    if raw_message is not None:
        try:
            message = _MESSAGE_ADAPTER.validate_python(raw_message)
        except ValidationError:  # 损坏条目跳过（对齐 pi 跳过坏行）
            return None
    known_keys = {
        "type",
        "id",
        "parent_id",
        "timestamp",
        "message",
        "provider",
        "model_id",
        "thinking_level",
    }
    extra = {key: value for key, value in data.items() if key not in known_keys}
    return SessionEntry(
        type=str(entry_type),
        id=str(data.get("id", "")),
        parent_id=data.get("parent_id"),
        timestamp=str(data.get("timestamp", "")),
        message=message,
        provider=data.get("provider"),
        model_id=data.get("model_id"),
        thinking_level=data.get("thinking_level"),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 文件读写
# ---------------------------------------------------------------------------


def write_header(path: Path, header: SessionHeader) -> None:
    """创建会话文件并写入首行（文件已存在则失败，对齐 pi 'wx' 语义）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(header_to_dict(header), ensure_ascii=False) + "\n")


def append_entry(path: Path, entry: SessionEntry) -> None:
    """追加一行条目（append-only，永不改写已有行）。"""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry_to_dict(entry), ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class LoadedSession:
    """一个会话文件的解析结果。"""

    header: SessionHeader
    entries: list[SessionEntry]

    @property
    def leaf_id(self) -> str | None:
        """最后追加的条目 id（默认上下文叶）。"""
        return self.entries[-1].id if self.entries else None


def parse_session_content(content: str) -> list[dict[str, Any]]:
    """解析 JSONL 内容（跳过损坏行，对齐 pi parseSessionEntries）。"""
    parsed: list[dict[str, Any]] = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
    return parsed


def load_session(path: Path, *, max_bytes: int = MAX_SESSION_FILE_BYTES) -> LoadedSession | None:
    """加载会话文件。

    Returns:
        解析结果；空文件 / 无 header / 超大 / 无有效条目 → None（容错跳过）。
    """
    try:
        if path.stat().st_size > max_bytes:
            return None
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    rows = parse_session_content(content)
    if not rows:
        return None
    header_row = rows[0]
    if header_row.get("type") != "session":
        return None
    header = SessionHeader(
        id=str(header_row.get("id", "")),
        timestamp=str(header_row.get("timestamp", "")),
        cwd=str(header_row.get("cwd", "")),
        parent_session=header_row.get("parent_session"),
    )

    entries: list[SessionEntry] = []
    for row in rows[1:]:
        entry = _entry_from_dict(row)
        if entry is not None:
            entries.append(entry)
    return LoadedSession(header=header, entries=entries)


# ---------------------------------------------------------------------------
# 发现与 fork
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionInfo:
    """会话列表条目（对齐 pi SessionInfo 角色）。"""

    id: str
    path: Path
    created: str
    modified: float
    message_count: int
    first_message: str
    parent_session: str | None = None


def _first_message_text(entries: list[SessionEntry]) -> str:
    """首条用户消息的文本预览（列表展示用）。"""
    for entry in entries:
        if entry.type == "message" and entry.message is not None:
            message = entry.message
            if message.role == "user":
                content = message.content
                return (
                    content
                    if isinstance(content, str)
                    else next((block.text for block in content if block.type == "text"), "")
                )
    return ""


def list_sessions(directory: Path, *, max_bytes: int = MAX_SESSION_FILE_BYTES) -> list[SessionInfo]:
    """列出目录下的会话（按修改时间倒序；容错跳过坏文件）。

    一个损坏/超大文件不影响其他会话的发现（checklist）。
    """
    if not directory.is_dir():
        return []
    infos: list[SessionInfo] = []
    for path in directory.iterdir():
        if not path.name.endswith(".jsonl") or not path.is_file():
            continue
        loaded = load_session(path, max_bytes=max_bytes)
        if loaded is None:
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        infos.append(
            SessionInfo(
                id=loaded.header.id or path.stem,
                path=path,
                created=loaded.header.timestamp,
                modified=modified,
                message_count=sum(1 for entry in loaded.entries if entry.type == "message"),
                first_message=_first_message_text(loaded.entries),
                parent_session=loaded.header.parent_session,
            )
        )
    infos.sort(key=lambda info: info.modified, reverse=True)
    return infos


def fork_session_file(
    source: Path,
    target_dir: Path,
    new_id: str | None = None,
    *,
    target_cwd: str | None = None,
) -> Path:
    """从源会话分叉：新文件 + 复制全部非 header 条目（对齐 pi forkFrom）。

    Args:
        source: 源会话文件。
        target_dir: 新会话所在目录。
        new_id: 新会话 id（缺省生成）。
        target_cwd: 新会话的 cwd（缺省沿用源 header）。

    Returns:
        新会话文件路径。

    Raises:
        SessionFormatError: 源为空 / 无 header。
    """
    loaded = load_session(source)
    if loaded is None:
        raise SessionFormatError(f"Cannot fork: source session file is empty or invalid: {source}")

    session_id = new_id or new_session_id()
    assert_valid_session_id(session_id)
    timestamp = datetime.now(UTC).isoformat()
    target = session_file_path(target_dir, session_id)
    header = SessionHeader(
        id=session_id,
        timestamp=timestamp,
        cwd=target_cwd or loaded.header.cwd,
        parent_session=str(source),
    )
    write_header(target, header)
    for entry in loaded.entries:
        append_entry(target, entry)
    return target


def exclusive_create(path: Path) -> None:
    """独占创建（竞态保护，对齐 pi 'wx' flag）。"""
    os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
