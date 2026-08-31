"""热更新：周期性扫描知识目录并将变更同步到 Chroma。

每隔 SCAN_INTERVAL 秒（默认 30 分钟），它会：
  1. 递归扫描 data/knowledge/，计算每个文件的 MD5。
  2. 与上一次快照（data/state/ingest_snapshot.json）进行对比。
  3. 增量应用差异：添加新文件、重新摄入变更过的文件、
     删除已移除文件的 chunk。
  4. 重建稀疏（BM25）索引并更新快照。

在守护线程上运行，因此绝不会阻塞 Flask 请求循环。
"""

from __future__ import annotations

import json
import os
import threading
import time

from log_tool import get_logger

logger = get_logger(name="hot_ingest")

# ──────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────

_SCAN_INTERVAL = 30 * 60          # 30 分钟
_KNOWLEDGE_DIR = "data/knowledge"
_SNAPSHOT_FILE = "data/state/ingest_snapshot.json"

_SUPPORTED_EXTS = (".txt", ".md", ".pdf", ".csv", ".docx", ".pptx", ".xlsx")

_lock = threading.Lock()          # 序列化摄入与重置操作


# ──────────────────────────────────────────────────────────────────────────
# 快照
# ──────────────────────────────────────────────────────────────────────────

def _snapshot_path() -> str:
    from path_tool import get_abs_path
    return get_abs_path(_SNAPSHOT_FILE)


def _knowledge_path() -> str:
    from path_tool import get_abs_path
    return get_abs_path(_KNOWLEDGE_DIR)


def load_snapshot() -> dict:
    """读取上一次快照。如果不存在则返回 {}（首次运行）。"""
    path = _snapshot_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[HotIngest] Failed to load snapshot: %s", e)
        return {}


def save_snapshot(snapshot: dict) -> None:
    path = _snapshot_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def invalidate_snapshot() -> None:
    """丢弃快照，使下一个周期执行一次完整的重新摄入。"""
    path = _snapshot_path()
    if os.path.isfile(path):
        os.remove(path)
        logger.info("[HotIngest] Snapshot invalidated")


# ──────────────────────────────────────────────────────────────────────────
# 扫描与差异对比
# ──────────────────────────────────────────────────────────────────────────

def scan_files(data_dir: str) -> dict:
    """递归扫描 data_dir → {相对路径: md5}。"""
    from file_tools import get_file_md5_hex

    result = {}
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if not f.lower().endswith(_SUPPORTED_EXTS):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, data_dir)
            md5 = get_file_md5_hex(full)
            if md5 is not None:
                result[rel] = md5
    return result


def diff_snapshot(old: dict, new: dict):
    """返回旧快照与新快照之间的 (added, changed, removed)。"""
    added = {k: v for k, v in new.items() if k not in old}
    changed = {k: v for k, v in new.items() if k in old and old[k] != v}
    removed = [k for k in old if k not in new]
    return added, changed, removed


# ──────────────────────────────────────────────────────────────────────────
# 应用变更
# ──────────────────────────────────────────────────────────────────────────

def apply_changes(data_dir: str, added: dict, changed: dict, removed: list) -> dict:
    """根据文件差异增量同步 Chroma。"""
    from vector_store import get_vector_store, ingest_file

    store = get_vector_store()
    results = {"added": 0, "changed": 0, "removed": 0}

    for rel in added:
        r = ingest_file(os.path.join(data_dir, rel))
        if r.get("status") == "ok":
            results["added"] += 1
            logger.info("[HotIngest] added: %s (%d chunks)", rel, r.get("chunks", 0))

    for rel in changed:
        file_name = os.path.basename(rel)
        store._collection.delete(where={"file_name": file_name})
        r = ingest_file(os.path.join(data_dir, rel))
        if r.get("status") == "ok":
            results["changed"] += 1
            logger.info("[HotIngest] changed: %s (%d chunks)", rel, r.get("chunks", 0))

    for rel in removed:
        file_name = os.path.basename(rel)
        store._collection.delete(where={"file_name": file_name})
        results["removed"] += 1
        logger.info("[HotIngest] removed: %s", rel)

    return results


def _rebuild_sparse_index() -> None:
    """强制重建 BM25 索引（丢弃缓存，避免复用陈旧的 pickle）。"""
    from path_tool import get_abs_path
    # 同步失效型号元数据缓存（models.pkl）——知识库变更后型号数据已过期
    for cache_name in ("data/pkl/bm25_index.pkl", "data/pkl/models.pkl"):
        cache = get_abs_path(cache_name)
        if os.path.isfile(cache):
            os.remove(cache)
    from vector_store import build_hybrid_index
    build_hybrid_index()

    # 重置 agent 的惰性检索器单例，使下一次查询重新加载
    try:
        import agent
        agent._hybrid_retriever = None
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────
# 周期
# ──────────────────────────────────────────────────────────────────────────

def run_once(data_dir: str = None) -> bool:
    """执行一次扫描 + 增量同步。若有任何变更则返回 True。"""
    data_dir = data_dir or _knowledge_path()
    if not os.path.isdir(data_dir):
        logger.warning("[HotIngest] knowledge dir missing: %s", data_dir)
        return False

    with _lock:
        old = load_snapshot()
        new = scan_files(data_dir)
        added, changed, removed = diff_snapshot(old, new)

        if not (added or changed or removed):
            logger.info("[HotIngest] No changes detected")
            return False

        logger.info(
            "[HotIngest] added=%d changed=%d removed=%d",
            len(added), len(changed), len(removed),
        )
        apply_changes(data_dir, added, changed, removed)
        _rebuild_sparse_index()
        save_snapshot(new)
        return True


def hot_ingest_loop(data_dir: str = None, interval: int = _SCAN_INTERVAL) -> None:
    """后台循环：每隔 `interval` 秒执行一次 run_once。"""
    data_dir = data_dir or _knowledge_path()
    logger.info("[HotIngest] Started, interval=%ds dir=%s", interval, data_dir)
    while True:
        try:
            run_once(data_dir)
        except Exception as e:
            logger.error("[HotIngest] cycle failed: %s", e)
        time.sleep(interval)


def start_hot_ingest(data_dir: str = None, interval: int = _SCAN_INTERVAL) -> threading.Thread:
    """启动热更新守护线程。返回线程对象。"""
    t = threading.Thread(
        target=hot_ingest_loop,
        args=(data_dir or _knowledge_path(), interval),
        daemon=True,
        name="hot-ingest",
    )
    t.start()
    return t
