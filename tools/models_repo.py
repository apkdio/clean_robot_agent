"""型号记录仓库：全量型号的唯一内存/磁盘缓存。

入库时已经算好的基础字段、`param_*` 与 `tag_*` 都在这里合并成一条型号记录；
调用方只读 `all_records()`，不要再各自扫 Chroma 或另建缓存。
"""
from __future__ import annotations

import os
import pickle
import re
import threading

from config.word_dict_config import BRAND_NAME
from tools.log_tool import get_logger

logger = get_logger(name="models_repo")

_CACHE_FILE = "data/pkl/models.pkl"
_CACHE_SCHEMA = 1
_LOCK = threading.RLock()
_cache = {"fingerprint": None, "records": None}


def bare_name(name: str) -> str:
    """去掉品牌前缀后的型号名（「不染一尘云顶 X2」→「云顶 X2」）。"""
    return name[len(BRAND_NAME):] if name.startswith(BRAND_NAME) else name


def match_key(name: str) -> str:
    """型号名匹配/去重键：去品牌前缀 + 去空白。"""
    return re.sub(r"\s+", "", bare_name(name or ""))


def _order_key(doc):
    """Chroma 返回顺序不保证；按文件名 + chunk 起点排序，保证重建结果稳定。"""
    meta = doc.metadata or {}
    start = meta.get("start_index")
    try:
        start = int(start)
    except (TypeError, ValueError):
        start = -1
    return (str(meta.get("file_name") or ""), start, str(meta.get("source") or ""))


def _merge_record(base: dict, incoming: dict, name: str) -> None:
    """重复型号只补缺失字段；非空字段冲突保留先到的，并告警。"""
    for key, value in incoming.items():
        if key.startswith("tag_"):
            # 标签是布尔事实：任一 chunk 命中即为 True，不被另一 chunk 的 False 覆盖
            base[key] = bool(base.get(key)) or bool(value)
            continue
        current = base.get(key)
        if current in (None, "") and value not in (None, ""):
            base[key] = value
        elif value not in (None, "") and value != current:
            logger.warning("[ModelsRepo] 型号 %s 字段 %s 冲突：保留 %r，忽略 %r",
                           name, key, current, value)


def _build_records() -> list[dict]:
    """从 Chroma 全量 chunk 重建记录；按 match_key 去重并合并。"""
    from tools.metadata_extractor import extract_model_info
    from tools.vector_store import search_by_filter

    docs = sorted(search_by_filter({"file_name": {"$ne": "__never__"}}), key=_order_key)
    merged = {}
    for doc in docs:
        info = extract_model_info(doc)
        name = info.get("name")
        if not name or info.get("price") is None:
            continue
        key = match_key(name)
        if not key:
            continue
        info.update({k: v for k, v in (doc.metadata or {}).items()
                     if k.startswith(("param_", "tag_"))})
        if key not in merged:
            merged[key] = info
        else:
            _merge_record(merged[key], info, name)
    return list(merged.values())


def _load_cached(fingerprint: str):
    from tools.path_tool import get_abs_path

    path = get_abs_path(_CACHE_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("schema") == _CACHE_SCHEMA and payload.get("fingerprint") == fingerprint:
            return payload.get("records") or []
    except Exception as e:  # noqa: BLE001
        logger.warning("[ModelsRepo] 读取型号缓存失败：%s", e)
    return None


def _save_cached(fingerprint: str, records: list[dict]) -> None:
    from tools.path_tool import get_abs_path

    path = get_abs_path(_CACHE_FILE)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"schema": _CACHE_SCHEMA, "fingerprint": fingerprint,
                         "records": records}, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:  # noqa: BLE001
        logger.warning("[ModelsRepo] 写入型号缓存失败：%s", e)


def all_records() -> list[dict]:
    """全量型号记录（基础字段 + `param_*` + `tag_*`），进程内命中直接返回。

    冷启动先按 `store_fingerprint()` 校验磁盘缓存；知识库变化由 ingest/hot ingest
    调用 `invalidate()`，不会靠调用方各自判断缓存是否过期。
    """
    with _LOCK:
        if _cache["records"] is not None:
            return _cache["records"]
        from tools.vector_store import store_fingerprint

        fingerprint = store_fingerprint()
        records = _load_cached(fingerprint)
        if records is None:
            records = _build_records()
            _save_cached(fingerprint, records)
            logger.info("[ModelsRepo] rebuilt %d records (fingerprint=%s)", len(records), fingerprint)
        _cache["fingerprint"] = fingerprint
        _cache["records"] = records
        return records


def series_catalog() -> list[str]:
    """从型号记录派生系列目录；repo 不可用时回退静态表并 WARNING。"""
    try:
        catalog = sorted({str(r.get("series") or "").strip()
                           for r in all_records()
                           if str(r.get("series") or "").strip()})
        if catalog:
            return catalog
        logger.warning("[ModelsRepo] 型号记录里没有 series，系列目录回退静态表")
    except Exception as e:  # noqa: BLE001
        logger.warning("[ModelsRepo] 系列目录读取失败，回退静态表：%s", e)
    from config.word_dict_config import SERIES_LIST
    return list(SERIES_LIST)


def invalidate() -> None:
    """丢弃内存与磁盘缓存；入库/热更新后调用。"""
    from tools.path_tool import get_abs_path

    with _LOCK:
        _cache["fingerprint"] = None
        _cache["records"] = None
        path = get_abs_path(_CACHE_FILE)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError as e:
                logger.warning("[ModelsRepo] 删除型号缓存失败：%s", e)
