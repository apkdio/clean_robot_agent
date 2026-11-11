"""知识库用例门面：上传入库、重置、统计与热更新生命周期。"""
from __future__ import annotations

from gateway.redis_store import acquire_lock, release_lock
from tools.config_tool import get_config
from tools.dto.knowledge import IngestResult, KnowledgeStats, ResetResult
from service.hot_ingest import (
    invalidate_runtime_caches,
    invalidate_snapshot,
    start_hot_ingest,
)
from tools.vector_store import ingest_file, list_collections_info, reset_collection

_INGEST_LOCK = "lock:ingest"


def supported_exts() -> tuple:
    return tuple((get_config("rag") or {}).get("supported_exts", [".txt", ".pdf"]))


def collections_info() -> KnowledgeStats:
    info = list_collections_info()
    return KnowledgeStats(
        collection_name=info.get("collection_name", ""),
        chunk_count=int(info.get("chunk_count") or 0),
    )


def acquire_ingest_lock() -> bool:
    return acquire_lock(_INGEST_LOCK)


def release_ingest_lock() -> None:
    release_lock(_INGEST_LOCK)


def ingest_uploaded_file(file_path: str) -> IngestResult:
    """入库单个文件；成功后让快照和运行时缓存失效，下一轮懒重建。"""
    result = ingest_file(file_path)
    if result.get("status") == "ok":
        # 上传按 basename 替换会盖掉同名知识文件的 chunk，磁盘原文件哈希却没变；
        # 快照与索引从此对不上，作废让下轮热更新全量对齐。
        invalidate_snapshot()
        invalidate_runtime_caches()
    return IngestResult(
        status="ok" if result.get("status") == "ok" else "error",
        file=result.get("file", ""),
        chunks=int(result.get("chunks") or 0),
        sha256=result.get("sha256", ""),
        elapsed=float(result.get("elapsed") or 0),
        message=result.get("message", ""),
    )


def reset_knowledge_base() -> ResetResult:
    """清空知识库；拿不到锁返回 busy 结果，由 Controller 映射成 409。"""
    if not acquire_ingest_lock():
        return ResetResult("busy")
    try:
        # 先失效快照再清空：两步之间中断时，下一轮按「全新增」重入，
        # 不会留下「空索引 + 快照称已同步」。
        invalidate_snapshot()
        deleted = reset_collection()
        invalidate_runtime_caches()
        return ResetResult("ok", deleted=deleted)
    finally:
        release_ingest_lock()


def start_background_ingest() -> None:
    start_hot_ingest()
