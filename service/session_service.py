"""会话用例门面：Controller 只通过这里读写会话上下文。"""
from __future__ import annotations

import re

from mapper.context_store import (
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


def generate_session_title(query: str, answer: str) -> str:
    """调用轻量模型生成会话标题；失败回退到 query 截断。"""
    fallback_title = (query.strip()[:20] or "新会话")
    try:
        from gateway.llm_tool import get_chat_model, get_small_model_name
        from utils.prompts_tool import render
        from langchain_core.messages import HumanMessage

        prompt = render("session_title", query=query, answer=answer[:120])
        if prompt is None:
            return fallback_title
        llm = get_chat_model(model=get_small_model_name(), temperature=0.3)
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw_title = getattr(resp, "content", "") or ""
        cleaned = re.sub(r'["\'《》“”‘’\n\r\t。，！？,.!?]', '', raw_title).strip()
        if cleaned:
            cleaned = re.sub(r'^(?:会话|对话)?标题[：:]\s*', '', cleaned).strip()
            if cleaned:
                return cleaned[:20]
    except Exception:
        pass
    return fallback_title
