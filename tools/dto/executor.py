"""操作执行体的返回契约：接管并回话，或不是我的。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reply:
    """接管本轮；text 为空表示接管但无话术。"""

    text: str = ""

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise ValueError("Reply.text 必须是 str")


@dataclass(frozen=True)
class NotMine:
    """不是该执行体负责，交回派发器试下一个。"""


ExecOutcome = Reply | NotMine
