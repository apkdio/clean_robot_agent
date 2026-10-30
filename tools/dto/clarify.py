"""澄清业务的返回体与载荷。"""
from __future__ import annotations

from dataclasses import dataclass

CLARIFY_DONE = "done"
CLARIFY_RESUME = "resume"
CLARIFY_DROP = "drop"
CLARIFY_KINDS = (CLARIFY_DONE, CLARIFY_RESUME, CLARIFY_DROP)


@dataclass(frozen=True)
class ClarifyData:
    """resume 的业务载荷：恢复 query、本轮偏好与拒绝标记。"""

    query: str
    prefer_op: str | None = None
    enter_declined: bool = False   # 本轮跳过进入门；持久拒绝标记由业务另行写入


@dataclass(frozen=True)
class ClarifyResult:
    """澄清处理器的统一返回体：code + message + data（data 按 code 取用）。"""

    code: str
    message: str = ""
    data: ClarifyData | None = None

    def __post_init__(self):
        if self.code not in CLARIFY_KINDS:
            raise ValueError("clarify code 不在闭集内：%r" % self.code)
        if self.code == CLARIFY_DONE and not self.message:
            raise ValueError("clarify done 必须带 message")
        if self.code == CLARIFY_RESUME and (not self.data or not self.data.query):
            raise ValueError("clarify resume 必须带 query")

    @property
    def kind(self) -> str:
        """兼容旧读侧；新代码用 code。"""
        return self.code

    @classmethod
    def done(cls, reply: str) -> "ClarifyResult":
        return cls(CLARIFY_DONE, message=reply)

    @classmethod
    def resume(cls, query: str, prefix: str = "", prefer_op: str | None = None,
               enter_declined: bool = False) -> "ClarifyResult":
        return cls(CLARIFY_RESUME, message=prefix, data=ClarifyData(
            query=query, prefer_op=prefer_op, enter_declined=enter_declined))

    @classmethod
    def drop(cls) -> "ClarifyResult":
        return cls(CLARIFY_DROP)
