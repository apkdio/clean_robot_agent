"""主链阶段的控制流契约：继续或结束。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Continue:
    """本轮继续走下一阶段；prefix 是要先输出给用户的横切话术。"""

    prefix: str = ""


@dataclass(frozen=True)
class Finish:
    """本轮到此结束；text 是成品话术，stream 是待消费的流式生成器。

    text 与 stream 最多给一个；都不给表示静默结束。
    """

    text: str | None = None
    stream: Iterator[str] | None = None
    prefix: str = ""

    def __post_init__(self):
        if self.text is not None and self.stream is not None:
            raise ValueError("Finish 不能同时携带 text 和 stream")

    @classmethod
    def text_reply(cls, text: str, prefix: str = "") -> "Finish":
        return cls(text=text, prefix=prefix)

    @classmethod
    def stream_reply(cls, stream: Iterator[str], prefix: str = "") -> "Finish":
        return cls(stream=stream, prefix=prefix)

    @classmethod
    def silent(cls, prefix: str = "") -> "Finish":
        return cls(prefix=prefix)


StageOutcome = Continue | Finish
