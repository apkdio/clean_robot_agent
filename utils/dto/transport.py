"""会话交付事件：webapp 只做 SSE 序列化。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


_KINDS = ("delta", "error", "stopped", "meta", "done")


@dataclass(frozen=True)
class TurnEvent:
    """一轮对话对外交付的事件；kind 是闭集。"""

    kind: Literal["delta", "error", "stopped", "meta", "done"]
    text: str = ""
    meta: dict | None = None

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError("TurnEvent.kind 不在闭集内：%r" % (self.kind,))
        if not isinstance(self.text, str):
            raise ValueError("TurnEvent.text 必须是 str")
        if self.kind in ("delta", "error") and not self.text:
            raise ValueError("%s 事件必须带 text" % self.kind)
        if self.kind == "meta" and not isinstance(self.meta, dict):
            raise ValueError("meta 事件必须带 dict meta")
