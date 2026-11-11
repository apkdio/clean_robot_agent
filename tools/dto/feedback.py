"""反馈用例的请求与结果：Controller 负责 HTTP，service 只接收/返回 DTO。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackRequest:
    query: str
    session_id: str
    rating: str
    reason: str = ""
    note: str = ""
    date: str | None = None
    turn_index: int | None = None

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query 必须是非空字符串")
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id 必须是非空字符串")
        if not isinstance(self.rating, str):
            raise ValueError("rating 必须是字符串")
        if not isinstance(self.reason, str):
            raise ValueError("reason 必须是字符串")
        if not isinstance(self.note, str):
            raise ValueError("note 必须是字符串")
        if self.date is not None and (not isinstance(self.date, str) or not self.date.strip()):
            raise ValueError("date 必须是非空字符串")
        if self.turn_index is not None:
            if isinstance(self.turn_index, bool) or not isinstance(self.turn_index, int):
                raise ValueError("turn_index 必须是整数")
            if self.turn_index < 0:
                raise ValueError("turn_index 不能为负")
        if (self.date is None) != (self.turn_index is None):
            raise ValueError("date 与 turn_index 必须同时提供")


_FEEDBACK_STATUSES = ("ok", "error", "no_turn")


@dataclass(frozen=True)
class FeedbackResult:
    status: str
    message: str = ""
    date: str | None = None
    turn_index: int | None = None
    updated: bool = False

    def __post_init__(self):
        if self.status not in _FEEDBACK_STATUSES:
            raise ValueError("FeedbackResult.status 不在闭集内：%r" % (self.status,))

    def payload(self) -> dict:
        body = {"status": self.status}
        if self.message:
            body["message"] = self.message
        if self.status == "ok":
            body.update(date=self.date, turn_index=self.turn_index, updated=self.updated)
        return body
