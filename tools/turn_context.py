"""一轮对话的业务状态与阶段返回体的兼容入口。

TurnContext 只放业务状态，不放 trace、不攒日志；阶段返回 Continue / Finish，
不直接向用户输出。返回体本身定义在 tools/dto/。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tools.dto.request import ChatRequest
from tools.dto.stage import Continue, Finish, StageOutcome

__all__ = ["TurnContext", "Continue", "Finish", "StageOutcome"]


@dataclass
class TurnContext:
    """本轮状态单：阶段之间只通过它传递业务事实。"""

    request: ChatRequest
    query: str
    q: str
    stop_check: Callable[[], bool] | None = None

    intent: str | None = None
    margin: float = 0.0
    low_conf: bool = False
    intent_retry: dict | None = None
    prefer_op: str | None = None
    enter_declined_turn: bool = False
    op_note: dict = field(default_factory=dict)
    chunks: list | None = None

    @classmethod
    def from_request(cls, request: ChatRequest,
                     stop_check: Callable[[], bool] | None = None) -> "TurnContext":
        return cls(request=request, query=request.query, q=request.query,
                   stop_check=stop_check)

    @property
    def session_id(self) -> str:
        return self.request.session_id

    @property
    def lng(self) -> float | None:
        return self.request.lng

    @property
    def lat(self) -> float | None:
        return self.request.lat

    @property
    def retry(self) -> bool:
        return self.request.retry
