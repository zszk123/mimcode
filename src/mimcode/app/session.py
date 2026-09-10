"""会话运行时：append-only 树上的追加、上下文构建与分支（对齐 pi SessionManager 角色）。

职责边界：
- ``SessionManager``：一个打开的会话文件——追加条目（消息/设置变更）、
  维护叶指针、构建 leaf→root 上下文
- session_store：文件级操作（解析/发现/fork）

上下文构建（对齐 pi buildSessionContext）：
- 消息条目 → AgentMessage 列表（leaf→root 链上）
- 设置条目（model_change/thinking_level_change）沿链取最后值
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mimcode.app import session_store as store
from mimcode.types import AgentMessage


@dataclass
class SessionSettings:
    """从会话条目流恢复的运行设置。"""

    provider: str | None = None
    model_id: str | None = None
    thinking_level: str | None = None


@dataclass
class SessionContext:
    """恢复会话的完整上下文（对齐 pi SessionContext 角色）。"""

    messages: list[AgentMessage] = field(default_factory=list)
    settings: SessionSettings = field(default_factory=SessionSettings)


class SessionManager:
    """打开的会话：追加条目、上下文构建、同文件分叉。"""

    def __init__(
        self,
        session_file: Path,
        header: store.SessionHeader,
        entries: list[store.SessionEntry],
    ) -> None:
        self._file = session_file
        self._header = header
        self._entries = list(entries)

    # --- 工厂 ---

    @classmethod
    def create(
        cls,
        cwd: str,
        *,
        home: Path | None = None,
        session_id: str | None = None,
        parent_session: str | None = None,
    ) -> SessionManager:
        """新建会话（新文件 + header）。"""
        resolved_id = session_id or store.new_session_id()
        store.assert_valid_session_id(resolved_id)
        directory = store.session_dir_for(cwd, home)
        path = store.session_file_path(directory, resolved_id)
        timestamp = datetime.now(UTC).isoformat()
        header = store.SessionHeader(
            id=resolved_id,
            timestamp=timestamp,
            cwd=str(Path(cwd).resolve()),
            parent_session=parent_session,
        )
        store.write_header(path, header)
        return cls(path, header, [])

    @classmethod
    def open(cls, session_file: Path) -> SessionManager:
        """打开已有会话。

        Raises:
            store.SessionFormatError: 文件无效（空/无 header/损坏）。
        """
        loaded = store.load_session(session_file)
        if loaded is None:
            raise store.SessionFormatError(f"Invalid session file: {session_file}")
        return cls(session_file, loaded.header, loaded.entries)

    @classmethod
    def fork_from(
        cls,
        source: Path,
        target_cwd: str,
        *,
        home: Path | None = None,
        new_id: str | None = None,
    ) -> SessionManager:
        """从源会话分叉新会话（新文件复制条目，对齐 pi forkFrom）。"""
        resolved_id = new_id or store.new_session_id()
        directory = store.session_dir_for(target_cwd, home)
        target = store.fork_session_file(source, directory, resolved_id, target_cwd=target_cwd)
        return cls.open(target)

    # --- 查询 ---

    @property
    def session_file(self) -> Path:
        return self._file

    @property
    def session_id(self) -> str:
        return self._header.id

    @property
    def cwd(self) -> str:
        return self._header.cwd

    @property
    def entries(self) -> list[store.SessionEntry]:
        """全部条目（引用视图；调用方不得修改）。"""
        return self._entries

    def leaf_id(self) -> str | None:
        """当前叶（最后追加条目）。"""
        return self._entries[-1].id if self._entries else None

    def entry_by_id(self, entry_id: str) -> store.SessionEntry | None:
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    def path_to_root(self, leaf_id: str | None = None) -> list[store.SessionEntry]:
        """leaf→root 的条目链（正序；对齐 pi buildSessionPath）。

        Args:
            leaf_id: 起始叶（缺省当前叶）。
        """
        start = self.leaf_id() if leaf_id is None else leaf_id
        if not self._entries or not start:
            return []
        index = {entry.id: entry for entry in self._entries}
        chain: list[store.SessionEntry] = []
        current = index.get(start)
        while current is not None:
            chain.append(current)
            current = index.get(current.parent_id) if current.parent_id is not None else None
        chain.reverse()
        return chain

    # --- 上下文 ---

    def build_context(self, leaf_id: str | None = None) -> SessionContext:
        """构建 LLM 上下文（链上消息 + 设置，对齐 pi buildSessionContext）。"""
        chain = self.path_to_root(leaf_id)
        messages: list[AgentMessage] = []
        settings = SessionSettings()
        for entry in chain:
            if entry.type == "message" and entry.message is not None:
                messages.append(entry.message)
            elif entry.type == "model_change":
                settings.provider = entry.provider
                settings.model_id = entry.model_id
            elif entry.type == "thinking_level_change":
                settings.thinking_level = entry.thinking_level
        return SessionContext(messages=messages, settings=settings)

    # --- 追加（append-only）---

    def _append(self, entry: store.SessionEntry) -> store.SessionEntry:
        store.append_entry(self._file, entry)
        self._entries.append(entry)
        return entry

    def _now_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def append_message(self, message: AgentMessage) -> str:
        """追加消息条目（父 = 当前叶）。

        Returns:
            条目 id。
        """
        entry_id = store.new_entry_id()
        self._append(
            store.SessionEntry(
                type="message",
                id=entry_id,
                parent_id=self.leaf_id(),
                timestamp=self._now_iso(),
                message=message,
            )
        )
        return entry_id

    def append_model_change(
        self, provider: str, model_id: str, *, parent_id: str | None = None
    ) -> str:
        """追加模型变更条目（parent_id 显式时可从历史节点分叉同文件分支）。"""
        entry_id = store.new_entry_id()
        self._append(
            store.SessionEntry(
                type="model_change",
                id=entry_id,
                parent_id=parent_id if parent_id is not None else self.leaf_id(),
                timestamp=self._now_iso(),
                provider=provider,
                model_id=model_id,
            )
        )
        return entry_id

    def append_thinking_level_change(self, level: str, *, parent_id: str | None = None) -> str:
        """追加思考级别变更条目。"""
        entry_id = store.new_entry_id()
        self._append(
            store.SessionEntry(
                type="thinking_level_change",
                id=entry_id,
                parent_id=parent_id if parent_id is not None else self.leaf_id(),
                timestamp=self._now_iso(),
                thinking_level=level,
            )
        )
        return entry_id

    def append_custom(
        self, entry_type: str, extra: dict[str, Any], *, parent_id: str | None = None
    ) -> str:
        """追加自定义类型条目（compaction 等，T8 使用）。"""
        entry_id = store.new_entry_id()
        self._append(
            store.SessionEntry(
                type=entry_type,
                id=entry_id,
                parent_id=parent_id if parent_id is not None else self.leaf_id(),
                timestamp=self._now_iso(),
                extra=extra,
            )
        )
        return entry_id

    def branch_from(self, from_entry_id: str) -> str:
        """从历史节点开新分支：追加一个分叉锚点条目（同文件分叉语义）。

        Returns:
            新分支叶 id。
        """
        if self.entry_by_id(from_entry_id) is None:
            raise store.SessionFormatError(f"Unknown entry id: {from_entry_id}")
        entry_id = store.new_entry_id()
        self._append(
            store.SessionEntry(
                type="branch",
                id=entry_id,
                parent_id=from_entry_id,
                timestamp=self._now_iso(),
            )
        )
        return entry_id

    # --- 最近会话发现 ---

    @staticmethod
    def latest(cwd: str, *, home: Path | None = None) -> SessionManager | None:
        """恢复 cwd 最近会话（无会话返回 None；--continue 语义）。"""
        directory = store.sessions_root(home) / store.encode_cwd_to_dir_name(cwd)
        sessions = store.list_sessions(directory)
        if not sessions:
            return None
        return SessionManager.open(sessions[0].path)
