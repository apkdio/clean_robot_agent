"""型号特点标签：受控标签目录 + 型号映射。

标签值是人工/产品主数据维护的事实，不从自由文本实时抽取；`feature` 原文仍保留在
型号记录里，作为证据。
"""
from __future__ import annotations

import json
import os

from config.word_dict_config import BRAND_NAME, TAG_CATALOG, TAG_ORDER
from tools.log_tool import get_logger
from tools.path_tool import get_abs_path

logger = get_logger(name="model_tags")

_DATA_FILE = "data/state/model_tags.json"
_cache = None


def _key(name: str) -> str:
    text = (name or "").strip()
    if text.startswith(BRAND_NAME):
        text = text[len(BRAND_NAME):]
    return text.replace(" ", "")


def load_model_tags() -> dict:
    """读取型号 → tag_id 列表；文件缺失返回空映射并显式告警。"""
    global _cache
    if _cache is not None:
        return _cache
    path = get_abs_path(_DATA_FILE)
    if not os.path.isfile(path):
        logger.warning("[ModelTags] %s 不存在，标签筛选按 unknown 处理", _DATA_FILE)
        _cache = {}
        return _cache
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning("[ModelTags] 读取 %s 失败：%s", _DATA_FILE, e)
        _cache = {}
        return _cache
    out = {}
    for name, item in (raw or {}).items():
        tags = item.get("tags") if isinstance(item, dict) else item
        if not isinstance(tags, list):
            logger.warning("[ModelTags] %s 的 tags 不是数组，忽略", name)
            continue
        unknown = [t for t in tags if t not in TAG_CATALOG]
        if unknown:
            logger.warning("[ModelTags] %s 含未登记标签 %s，已忽略", name, unknown)
        out[_key(name)] = [t for t in tags if t in TAG_CATALOG]
    _cache = out
    return _cache


def tags_of(name: str) -> set:
    return set(load_model_tags().get(_key(name), ()))


def has_tags(name: str, tags) -> bool:
    want = set(tags or ())
    return bool(want) and want.issubset(tags_of(name))


def tag_options():
    """按固定顺序返回 [(tag_id, label)]，供 SOP 展示。"""
    return [(t, TAG_CATALOG[t]["label"]) for t in TAG_ORDER]


def reset_cache() -> None:
    """测试/热更新用：清空映射缓存。"""
    global _cache
    _cache = None


def model_tag_source(name: str) -> str:
    """当前映射里该型号是否有标签记录（unknown 与“明确没有”分开）。"""
    return "known" if _key(name) in load_model_tags() else "unknown"
