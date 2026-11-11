"""SOP 步骤返回体：问下一槽、给结果、收尾或无活跃流程。"""
from __future__ import annotations

from dataclasses import dataclass

from utils.dto.handoff import Handoff


@dataclass(frozen=True)
class ExtractResult:
    """ask 步骤 extractor 的纯返回体；槽位写入与 trace 由 base 统一执行。"""

    kind: str
    value: object = None
    extra: dict | None = None
    reason: str = ""

    def __post_init__(self):
        if self.kind not in ("skip", "done", "fail"):
            raise ValueError("SOP extract kind 不在闭集内：%r" % self.kind)
        if self.extra is not None and not isinstance(self.extra, dict):
            raise ValueError("SOP extract extra 必须是 dict")
        if self.reason and not isinstance(self.reason, str):
            raise ValueError("SOP extract reason 必须是字符串")


@dataclass(frozen=True)
class SopStepResult:
    kind: str
    reply: str = ""
    done: bool = False
    handoff: Handoff | None = None

    def __post_init__(self):
        if self.kind not in ("ask", "reply", "done", "noop"):
            raise ValueError("SOP 返回 kind 不在闭集内：%r" % self.kind)
        if self.kind == "noop" and (self.reply or self.done or self.handoff):
            raise ValueError("SOP noop 不能带 reply/done/handoff")
        if self.kind == "ask" and (self.done or self.handoff):
            raise ValueError("SOP ask 不能标记 done/handoff")
        if self.handoff is not None and self.kind not in ("reply", "done"):
            raise ValueError("SOP handoff 只能出现在 reply/done")
