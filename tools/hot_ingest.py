"""热更新：周期性扫描知识目录，把文件增删改增量同步到 Chroma，并重建 BM25 索引。

比对依据是文件 SHA256 快照（data/state/ingest_snapshot.json）；每次同步成功后，
再把「索引里现在是什么」追加一行到 data/state/kb_metadata.jsonl（逐文件 sha256 + chunk 数，
只增不改，历史即知识库版本记录，回归与环境核对直接读最后一行）。跑在守护线程上，不阻塞 Flask 请求循环。
"""

from __future__ import annotations

import json
import os
import threading
import time

from tools.log_tool import get_logger

logger = get_logger(name="hot_ingest")

# ──────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────

_SCAN_INTERVAL = 30 * 60          # 30 分钟
_SNAPSHOT_FILE = "data/state/ingest_snapshot.json"
_METADATA_FILE = "data/state/kb_metadata.jsonl"
METADATA_VERSION = 1

_SUPPORTED_EXTS = (".txt", ".md", ".pdf", ".csv", ".docx", ".pptx", ".xlsx")


# ──────────────────────────────────────────────────────────────────────────
# 快照
# ──────────────────────────────────────────────────────────────────────────

def _snapshot_path() -> str:
    from tools.path_tool import get_abs_path
    return get_abs_path(_SNAPSHOT_FILE)


def _knowledge_path() -> str:
    """知识库源目录（rag.yaml 的 data_dir，默认 data/knowledge）。"""
    from tools.config_tool import get_data_dir
    return get_data_dir()


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
# 知识库版本记录（data/state/kb_metadata.jsonl，只增不改）
# ──────────────────────────────────────────────────────────────────────────

def metadata_path() -> str:
    from tools.path_tool import get_abs_path
    return get_abs_path(_METADATA_FILE)


def kb_fingerprint(file_list: dict) -> str:
    """短指纹：逐文件 sha256 + chunk 数排序后折成一个摘要，任一项变化都会改变它。"""
    import hashlib
    text = "\n".join("%s:%s:%s" % (k, (v or {}).get("sha256"), (v or {}).get("chunks"))
                     for k, v in sorted(file_list.items()))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def load_kb_metadata() -> dict:
    """读最后一行（= 当前生效的知识库版本）；没有或读不动返回 {}。"""
    path = metadata_path()
    if not os.path.isfile(path):
        return {}
    last = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)
    except Exception as e:  # noqa: BLE001
        logger.warning("[HotIngest] Failed to load kb metadata: %s", e)
        return {}
    return last


def append_kb_metadata(file_list: dict, ts: str = "") -> None:
    """把「索引里现在是什么」追加一行（写失败只告警，不影响对话）。"""
    if not file_list:
        return
    rec = {"v": METADATA_VERSION,
           "ts": ts or time.strftime("%Y-%m-%dT%H:%M:%S"),
           "fingerprint": kb_fingerprint(file_list),
           "file_list": file_list}
    try:
        path = metadata_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("[HotIngest] kb metadata → %s（%d 文件，fp=%s）",
                    os.path.basename(path), len(file_list), rec["fingerprint"])
    except Exception as e:  # noqa: BLE001
        logger.warning("[HotIngest] Failed to append kb metadata: %s", e)


def chunks_by_file(names) -> dict:
    """从 Chroma 数每个文件的 chunk 数（以索引为准，不看目录）。"""
    from tools.vector_store import get_vector_store
    store = get_vector_store()
    out = {}
    for name in names:
        try:
            out[name] = len(store._collection.get(where={"file_name": name},
                                                  include=["metadatas"])["ids"])
        except Exception:  # noqa: BLE001
            out[name] = 0
    return out


def kb_file_list(scan: dict) -> dict:
    """扫描结果 + 索引里的 chunk 数 → metadata 的 file_list。"""
    counts = chunks_by_file([os.path.basename(rel) for rel in scan])
    return {rel: {"sha256": sha256, "chunks": counts.get(os.path.basename(rel), 0),
                  "ingested": counts.get(os.path.basename(rel), 0) > 0}
            for rel, sha256 in scan.items()}


# ──────────────────────────────────────────────────────────────────────────
# 扫描与差异对比
# ──────────────────────────────────────────────────────────────────────────

def scan_files(data_dir: str) -> dict:
    """递归扫描 data_dir → {相对路径: sha256}。"""
    from tools.file_tools import get_file_sha256_hex

    result = {}
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if not f.lower().endswith(_SUPPORTED_EXTS):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, data_dir)
            sha256 = get_file_sha256_hex(full)
            if sha256 is not None:
                result[rel] = sha256
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
    from tools.vector_store import get_vector_store, ingest_file

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
    from tools.path_tool import get_abs_path
    # 同步失效型号元数据缓存（models.pkl）——知识库变更后型号数据已过期
    for cache_name in ("data/pkl/bm25_index.pkl", "data/pkl/models.pkl"):
        cache = get_abs_path(cache_name)
        if os.path.isfile(cache):
            os.remove(cache)
    from tools.vector_store import build_hybrid_index
    build_hybrid_index()

    # 重置 agent 的惰性检索器单例，使下一次查询重新加载
    try:
        from tools import agent
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

    # 摄入锁：Redis 分布式锁（多实例只有一个摄入），不可用回退本地；被占用则跳过本次周期
    from tools.redis_store import acquire_lock, release_lock
    if not acquire_lock("lock:ingest"):
        logger.info("[HotIngest] another ingest in progress, skip this cycle")
        return False
    try:
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
        append_kb_metadata(kb_file_list(new))
        return True
    finally:
        release_lock("lock:ingest")


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
    """启动热更新守护线程。返回线程对象（启动前先做一次域映射自检）。"""
    data_dir = data_dir or _knowledge_path()
    # 域映射自检：文件改名/新文件未被认领时，域过滤会静默失效（只能靠全库兜底），这里报出来
    try:
        from config.word_dict_config import validate_knowledge_domains

        for problem in validate_knowledge_domains(data_dir):
            logger.warning("[HotIngest] 域映射自检：%s", problem)
    except Exception as e:  # noqa: BLE001
        logger.warning("[HotIngest] 域映射自检失败：%s", e)
    t = threading.Thread(
        target=hot_ingest_loop,
        args=(data_dir, interval),
        daemon=True,
        name="hot-ingest",
    )
    t.start()
    return t
