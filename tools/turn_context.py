"""一轮对话的业务状态与阶段返回体的兼容入口。

TurnContext 只放业务状态，不放 trace、不攒日志；阶段返回 Continue / Finish，
不直接向用户输出。返回体本身定义在 tools/dto/。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tools.dto.stage import Continue, Finish, StageOutcome

__all__ = ["TurnContext", "Continue", "Finish", "StageOutcome"]


@dataclass
class TurnContext:
    """本轮状态单：阶段之间只通过它传递业务事实。"""

    session_id: str
    query: str
    q: str
    lng: float | None = None
    lat: float | None = None
    retry: bool = False
    stop_check: Callable[[], bool] | None = None

    intent: str | None = None
    margin: float = 0.0
    low_conf: bool = False
    intent_retry: dict | None = None
    prefer_op: str | None = None
    enter_declined_turn: bool = False
    op_note: dict = field(default_factory=dict)
    chunks: list | None = None
