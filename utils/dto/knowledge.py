"""知识库用例结果：Controller 只负责状态码，service 返回结构化结果。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class KnowledgeStats:
    collection_name: str
    chunk_count: int

    def payload(self) -> dict:
        return {"collection_name": self.collection_name, "chunk_count": self.chunk_count}


@dataclass(frozen=True)
class IngestResult:
    status: Literal["ok", "error"]
    file: str = ""
    chunks: int = 0
    sha256: str = ""
    elapsed: float = 0.0
    message: str = ""

    def __post_init__(self):
        if self.status not in ("ok", "error"):
            raise ValueError("IngestResult.status 不在闭集内：%r" % (self.status,))

    def payload(self) -> dict:
        body = {"status": self.status, "file": self.file}
        if self.status == "ok":
            body.update(chunks=self.chunks, sha256=self.sha256, elapsed=self.elapsed)
        elif self.message:
            body["message"] = self.message
        return body


@dataclass(frozen=True)
class ResetResult:
    status: Literal["ok", "busy"]
    deleted: int = 0

    def __post_init__(self):
        if self.status not in ("ok", "busy"):
            raise ValueError("ResetResult.status 不在闭集内：%r" % (self.status,))

    def payload(self) -> dict:
        if self.status == "busy":
            return {"status": "busy"}
        return {"status": "ok", "deleted": self.deleted}
