"""会话用例门面：Controller 只通过这里读写会话上下文。"""
from __future__ import annotations

from repo.context_store import (
    delete_session,
    ensure_session_id,
    get_recent,
    list_sessions,
    rollback_last_user_message,
)


def normalize_session_id(session_id) -> str:
    """非法/缺失 session_id 统一补成合法 UUID。"""
    return ensure_session_id(session_id)


def is_valid_session_id(session_id) -> bool:
    return isinstance(session_id, str) and normalize_session_id(session_id) == session_id


def all_sessions() -> list:
    return list_sessions()


def recent_messages(session_id: str, n: int = 1000) -> list:
    return get_recent(session_id, n=n)


def remove_session(session_id: str) -> bool:
    return delete_session(session_id)


def rollback_last_user(session_id: str) -> None:
    rollback_last_user_message(session_id)
