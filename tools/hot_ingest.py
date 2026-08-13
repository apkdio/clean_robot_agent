"""Hot ingestion: periodically scan the knowledge dir and sync changes into Chroma.

Every SCAN_INTERVAL seconds (default 30 min), it:
  1. Scans data/knowledge/ recursively, computing each file's MD5.
  2. Compares against the last snapshot (data/state/ingest_snapshot.json).
  3. Incrementally applies the diff: add new files, re-ingest changed files,
     delete removed files' chunks.
  4. Rebuilds the sparse (BM25) index and updates the snapshot.

Runs on a daemon thread so it never blocks the Flask request loop.
"""

from __future__ import annotations

import json
import os
import threading
import time

from log_tool import get_logger

logger = get_logger(name="hot_ingest")

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────

_SCAN_INTERVAL = 30 * 60          # 30 minutes
_KNOWLEDGE_DIR = "data/knowledge"
_SNAPSHOT_FILE = "data/state/ingest_snapshot.json"

_SUPPORTED_EXTS = (".txt", ".md", ".pdf", ".csv", ".docx", ".pptx", ".xlsx")

_lock = threading.Lock()          # serialize ingest vs. reset


# ──────────────────────────────────────────────────────────────────────────
# Snapshot
# ──────────────────────────────────────────────────────────────────────────

def _snapshot_path() -> str:
    from path_tool import get_abs_path
    return get_abs_path(_SNAPSHOT_FILE)


def _knowledge_path() -> str:
    from path_tool import get_abs_path
    return get_abs_path(_KNOWLEDGE_DIR)


def load_snapshot() -> dict:
    """Read the last snapshot. Returns {} if absent (first run)."""
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
    """Drop the snapshot so the next cycle does a full re-ingest."""
    path = _snapshot_path()
    if os.path.isfile(path):
        os.remove(path)
        logger.info("[HotIngest] Snapshot invalidated")


# ──────────────────────────────────────────────────────────────────────────
# Scan & diff
# ──────────────────────────────────────────────────────────────────────────

def scan_files(data_dir: str) -> dict:
    """Recursively scan data_dir → {relative_path: md5}."""
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
    """Return (added, changed, removed) between old and new snapshots."""
    added = {k: v for k, v in new.items() if k not in old}
    changed = {k: v for k, v in new.items() if k in old and old[k] != v}
    removed = [k for k in old if k not in new]
    return added, changed, removed


# ──────────────────────────────────────────────────────────────────────────
# Apply changes
# ──────────────────────────────────────────────────────────────────────────

def apply_changes(data_dir: str, added: dict, changed: dict, removed: list) -> dict:
    """Incrementally sync Chroma with the file diff."""
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
    """Force a fresh BM25 index (drop cache so stale pickle isn't reused)."""
    from path_tool import get_abs_path
    cache = get_abs_path("data/pkl/bm25_index.pkl")
    if os.path.isfile(cache):
        os.remove(cache)
    from vector_store import build_hybrid_index
    build_hybrid_index()

    # Reset the agent's lazy retriever singleton so the next query reloads
    try:
        import agent
        agent._hybrid_retriever = None
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────
# Cycle
# ──────────────────────────────────────────────────────────────────────────

def run_once(data_dir: str = None) -> bool:
    """Execute one scan + incremental sync. Returns True if anything changed."""
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
    """Background loop: run_once every `interval` seconds."""
    data_dir = data_dir or _knowledge_path()
    logger.info("[HotIngest] Started, interval=%ds dir=%s", interval, data_dir)
    while True:
        try:
            run_once(data_dir)
        except Exception as e:
            logger.error("[HotIngest] cycle failed: %s", e)
        time.sleep(interval)


def start_hot_ingest(data_dir: str = None, interval: int = _SCAN_INTERVAL) -> threading.Thread:
    """Start the hot-ingest daemon thread. Returns the thread object."""
    t = threading.Thread(
        target=hot_ingest_loop,
        args=(data_dir or _knowledge_path(), interval),
        daemon=True,
        name="hot-ingest",
    )
    t.start()
    return t
