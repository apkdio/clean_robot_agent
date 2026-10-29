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
KB_METADATA_VERSION = 2
# 知识库版本记录（data/state/kb_metadata.jsonl）每行的记录格式版本，落盘字段名 schema_v——
# 不是知识库版本（那是 fingerprint），也不是代码版本（那在评测基线的 env.code）。
# v2：字段名由 v 改为 schema_v；老行仍带 v，读侧按 kb_metadata_version() 兼容（视作 1）。

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


def kb_metadata_version(rec: dict) -> int:
    """取一行记录的格式版本：新行是 schema_v，老行用的键是 v（视作 1）；认不出返回 0。"""
    if not isinstance(rec, dict):
        return 0
    if "schema_v" in rec:
        try:
            return int(rec.get("schema_v") or 0)
        except (TypeError, ValueError):
            return 0
    return 1 if "v" in rec else 0


def append_kb_metadata(file_list: dict, ts: str = "") -> None:
    """把「索引里现在是什么」追加一行（写失败只告警，不影响对话）。"""
    if not file_list:
        return
    rec = {"schema_v": KB_METADATA_VERSION,
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


def ids_of_file(name: str) -> list:
    """某文件当前在索引里的 chunk id（换新内容时删旧用）；取不到返回空表。"""
    from tools.vector_store import ids_of_file as _ids_of_file
    try:
        return _ids_of_file(name)
    except Exception as e:  # noqa: BLE001
        logger.warning("[HotIngest] 取旧 chunk id 失败（%s）：%s", name, e)
        return []


def files_missing_from_index(scan: dict, counts: dict) -> list:
    """扫描到、但索引里 0 chunk 的文件（相对路径）——哈希比对看不见这种状态。"""
    return [rel for rel in scan if counts.get(os.path.basename(rel), 0) == 0]


def kb_file_list(scan: dict, counts: dict = None) -> dict:
    """扫描结果 + 索引里的 chunk 数 → metadata 的 file_list。"""
    if counts is None:
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
    # 文件身份（chunk 的 file_name）用 basename：子目录里出现同名文件会互相覆盖 chunk、
    # 计数错乱。当前知识目录是扁平的，这里只做检测告警，真要支持子目录须改身份键
    seen = {}
    for rel in result:
        base = os.path.basename(rel)
        if base in seen:
            logger.warning("[HotIngest] 子目录重名文件（身份键会冲突）：%s 与 %s", seen[base], rel)
        else:
            seen[base] = rel
    return result


def diff_snapshot(old: dict, new: dict):
    """返回旧快照与新快照之间的 (added, changed, removed)。"""
    added = {k: v for k, v in new.items() if k not in old}
    changed = {k: v for k, v in new.items() if k in old and old[k] != v}
    removed = [k for k in old if k not in new]
    return added, changed, removed


def _project_root() -> str:
    """仓库根（本文件在 tools/ 下）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_state_of(path: str) -> str:
    """该文件的 git 状态（只读）：已提交 / 有未提交改动 / 未跟踪 / 未知，供变更明细判断"有人在动"。"""
    import subprocess
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", os.path.abspath(path)],
                             cwd=_project_root(), capture_output=True, text=True,
                             encoding="utf-8", timeout=5)
    except Exception as e:  # noqa: BLE001
        return "未知（git 不可用：%s）" % e
    if out.returncode != 0:
        return "未知（git 返回 %s）" % out.returncode
    line = (out.stdout or "").strip()
    if not line:
        return "已提交（与 HEAD 一致）"
    code = line.split()[0]
    return "未跟踪（新文件）" if code == "??" else "有未提交改动"


def change_detail(path: str) -> str:
    """变更明细：修改时间 + 大小/行数 + git 状态（乱码误触、半截保存都能看出异常）。"""
    try:
        st = os.stat(path)
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
        size = st.st_size
    except OSError as e:
        return "读取文件信息失败：%s" % e
    try:
        with open(path, encoding="utf-8") as f:
            lines = sum(1 for _ in f)
    except Exception:  # noqa: BLE001
        lines = "?"
    return "mtime=%s | %sB / %s行 | git=%s" % (mtime, size, lines, git_state_of(path))


# ──────────────────────────────────────────────────────────────────────────
# 应用变更
# ──────────────────────────────────────────────────────────────────────────

def apply_changes(data_dir: str, added: dict, changed: dict, removed: list) -> dict:
    """根据文件差异增量同步 Chroma。

    changed 先入库、成功后再删旧 chunk：ingest_file 最后一步才落库，先删后插一失败就留下 0 chunk 的空洞。
    """
    from tools.vector_store import get_vector_store, ingest_file

    store = get_vector_store()
    results = {"added": 0, "changed": 0, "removed": 0, "failed": []}

    for rel in added:
        r = ingest_file(os.path.join(data_dir, rel))
        if r.get("status") == "ok":
            results["added"] += 1
            logger.info("[HotIngest] added: %s (%d chunks)", rel, r.get("chunks", 0))
        else:
            results["failed"].append(rel)
            logger.warning("[HotIngest] added 失败，跳过本轮: %s (%s)", rel, r.get("message"))

    for rel in changed:
        file_name = os.path.basename(rel)
        old_ids = ids_of_file(file_name)          # 先把旧 chunk 记下来，插成功再删
        r = ingest_file(os.path.join(data_dir, rel))
        if r.get("status") == "ok":
            if old_ids:
                try:
                    store._collection.delete(ids=old_ids)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[HotIngest] 删旧 chunk 失败（%s，会留重复）: %s", file_name, e)
            results["changed"] += 1
            logger.info("[HotIngest] changed: %s (%d chunks, 删旧 %d)", rel, r.get("chunks", 0), len(old_ids))
        else:
            results["failed"].append(rel)
            logger.warning("[HotIngest] changed 失败，旧 chunk 保持不动、下轮重试: %s (%s)",
                           rel, r.get("message"))

    for rel in removed:
        file_name = os.path.basename(rel)
        store._collection.delete(where={"file_name": file_name})
        results["removed"] += 1
        logger.info("[HotIngest] removed: %s", rel)

    return results


def invalidate_runtime_caches() -> None:
    """清掉散在各模块的进程内缓存（知识库变更后统一走这里）。

    清单：检索器单例与上下文配置（agent）、域路由配置（domain_router）、
    型号名单（model_tool）、chat 配置（llm_tool）、网点清单（service_point_tool）。
    配置类缓存没有热加载通道——**改 yaml 仍需重启**；这里只是把入口收在一处，
    免得每个调用方各记一份清单、漏掉哪个都不知道。
    """
    from tools import agent as _agent
    from tools import domain_router as _domain_router
    from tools import query_rewrite as _query_rewrite
    _agent._hybrid_retriever = None
    _domain_router._rag_cfg_cache = None
    _query_rewrite._cfg_cache = None
    try:
        from tools import llm_tool as _llm
        _llm._chat_cfg_cache = None
        _llm._embedding_cfg_cache = None
    except Exception:  # noqa: BLE001
        pass
    try:
        from function_tools import model_tool as _mt, service_point_tool as _spt
        _mt._model_names_cache = None
        _spt._points_cache = None
    except Exception:  # noqa: BLE001
        pass


def rebuild_sparse_index() -> None:
    """强制重建 BM25 索引并失效相关缓存（丢弃缓存，避免复用陈旧的 pickle）。

    热更新与手动入库都走这里：知识库变更后 BM25 与 models.pkl 都过期了。
    """
    from tools.path_tool import get_abs_path
    # 同步失效型号元数据缓存（models.pkl）——知识库变更后型号数据已过期
    for cache_name in ("data/pkl/bm25_index.pkl", "data/pkl/models.pkl"):
        cache = get_abs_path(cache_name)
        if os.path.isfile(cache):
            os.remove(cache)
    from tools.vector_store import build_hybrid_index
    build_hybrid_index()
    invalidate_runtime_caches()


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

        # 索引自愈：扫到却是 0 chunk 的文件，按 changed 重入（哈希比对看不见这种状态）
        counts = chunks_by_file([os.path.basename(rel) for rel in new])
        for rel in files_missing_from_index(new, counts):
            if rel not in added and rel not in changed:
                logger.warning("[HotIngest] 索引里没有该文件的 chunk，按 changed 重入：%s", rel)
                changed[rel] = new[rel]

        if not (added or changed or removed):
            # 快照刚被作废而目录为空（清库场景）：索引缓存还留着旧内容，强制重建一次，
            # 否则空目录永远走不到下方的 rebuild，旧知识永久陈旧
            if not new and not os.path.isfile(_snapshot_path()):
                logger.warning("[HotIngest] snapshot invalidated with empty dir, force rebuild indexes")
                rebuild_sparse_index()
                save_snapshot(new)
            logger.info("[HotIngest] No changes detected")
            return False

        logger.info(
            "[HotIngest] added=%d changed=%d removed=%d",
            len(added), len(changed), len(removed),
        )
        # 变更明细：在应用之前打，摄取中途失败也留下痕迹
        for kind, rels in (("added", list(added)), ("changed", list(changed)), ("removed", list(removed))):
            for rel in rels:
                if kind == "removed":
                    logger.info("[HotIngest] 变更明细：removed %s | 已从目录消失，按记录删其 chunk", rel)
                else:
                    logger.info("[HotIngest] 变更明细：%s %s | %s", kind, rel,
                                change_detail(os.path.join(data_dir, rel)))
        results = apply_changes(data_dir, added, changed, removed)
        rebuild_sparse_index()
        # 快照只为成功的文件推进，失败的下轮还会被 diff 抓到
        failed = set(results.get("failed") or [])
        if failed:
            logger.warning("[HotIngest] 本轮有 %d 个文件未同步，快照不推进：%s", len(failed), sorted(failed))
        save_snapshot({k: v for k, v in new.items() if k not in failed})
        # 版本记录必须写"摄入后"的真实 chunk 数：apply 前取的 counts 是旧值，
        # 新增文件会记成 0、修改文件的条数与索引对不上（评测侧据此误报不一致）
        counts = chunks_by_file([os.path.basename(rel) for rel in new
                                 if rel not in failed and rel not in removed])
        append_kb_metadata(kb_file_list(new, counts))
        return True
    finally:
        release_lock("lock:ingest")


def hot_ingest_loop(data_dir: str = None, interval: int = _SCAN_INTERVAL) -> None:
    """后台循环：每隔 interval 秒执行一次 run_once。"""
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
