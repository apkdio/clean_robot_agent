"""SOP 步骤返回体：问下一槽、给结果、收尾或无活跃流程。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SopStepResult:
    kind: str
    reply: str = ""
    done: bool = False

    def __post_init__(self):
        if self.kind not in ("ask", "reply", "done", "noop"):
            raise ValueError("SOP 返回 kind 不在闭集内：%r" % self.kind)
        if self.kind == "noop" and (self.reply or self.done):
            raise ValueError("SOP noop 不能带 reply/done")
        if self.kind == "ask" and self.done:
            raise ValueError("SOP ask 不能标记 done")
