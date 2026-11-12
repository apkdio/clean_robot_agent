"""一轮对话的请求体：跨入口传递用户输入与重答标记。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatRequest:
    query: str
    session_id: str
    retry: bool = False

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query 必须是非空字符串")
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id 必须是非空字符串")
        if not isinstance(self.retry, bool):
            raise ValueError("retry 必须是 bool")
