"""追问回答：全局最新/价格极值，以及相对上一轮展示边界的筛选。

相对维度的过滤与排序统一走 `service.model_catalog.filter_by_relative`；
本模块只负责把用户话术解析成有序的 rels，并组织展示文案。
"""
from __future__ import annotations

import re

from config.word_dict_config import (
    CHEAPER_WORDS,
    LATEST_WORDS,
    NEWER_WORDS,
    OLDER_WORDS,
    PRICIER_WORDS,
    RECENT_VAGUE_WORDS,
    REL_LABEL,
    REL_SUPER,
)
from service.model_catalog import RELATIVE_SPEC, filter_by_relative
from mapper.context_store import get_last_models, set_last_models
from utils.metadata_rules import format_models
from service.metadata_query import enumerate_models

_REL_WORDS = {
    "cheaper": CHEAPER_WORDS,
    "pricier": PRICIER_WORDS,
    "newer": NEWER_WORDS,
    "older": OLDER_WORDS,
}
# 同一互斥对只能保留一个方向；按 query 中最后出现的位置裁决。
_REL_PAIRS = (("cheaper", "pricier"), ("newer", "older"))


def _last_pos(query: str, words) -> int:
    """词表在 query 中最后一次出现的位置；没有返回 -1。"""
    return max((query.rfind(w) for w in words if w), default=-1)


def _relative_parse(query: str) -> tuple:
    """返回 (按出现顺序的 rels, 发生互斥对冲突时最终选中的 rels)。"""
    query = query or ""
    positions = {rel: _last_pos(query, words) for rel, words in _REL_WORDS.items()}
    # 「更新更便宜」这类并列结构里的裸「更新」不在词表里，单独补一次
    match = re.search(r"更新更(?:便宜|贵|新|旧)", query)
    if match:
        positions["newer"] = max(positions["newer"], match.start())

    conflicts = []
    for left, right in _REL_PAIRS:
        if positions[left] < 0 or positions[right] < 0:
            continue
        chosen = left if positions[left] > positions[right] else right
        conflicts.append(chosen)
        positions[left if chosen == right else right] = -1

    rels = [rel for rel, pos in sorted(positions.items(), key=lambda item: item[1]) if pos >= 0]
    return rels, conflicts


def relative_rels(query: str) -> list:
    """解析相对维度，按 query 出现顺序返回；同一互斥对只保留最后命中的方向。"""
    return _relative_parse(query)[0]


def _rel_header(rels: list) -> str:
    """相对追问的回复头（多维度时并列）。"""
    return "比刚才那几款%s的有：" % "、".join(REL_LABEL[r] for r in rels)


def _rel_none_msg(rels: list) -> str:
    """越过门槛的型号一个都没有时的回复；多维度要点名组合，避免误读成单维度已到顶。"""
    if len(rels) == 1:
        return "刚才推荐的那几款已经是目前%s的了～" % REL_SUPER[rels[0]]
    return "比刚才那几款%s的型号暂时没有～" % "、".join(REL_LABEL[r] for r in rels)


def _relative_slice(prev: list, models: list, rels: list) -> list:
    """按相对维度过滤（同时满足），最接近门槛的在前，最多 3 条。"""
    return filter_by_relative(models, prev, rels)[:3]


def handle_followup(session_id: str, query: str):
    """回答追问。返回回复文本或 None。

    全局类（最近发布 / 最贵 / 最便宜）不依赖上一轮；相对类（更便宜 / 更贵 / 更新 / 更旧）
    以上一轮展示过的边界为门槛，在全库取越过边界、最接近边界的几款。
    """
    # 最近发布类：全局检索（不依赖上一轮上下文）
    latest_hit = any(w in query for w in LATEST_WORDS)
    vague_recent = (
            "最近" in query
            and not re.search(r"最近(?:半|几|[0-9一二两三四五六七八九十]|个?[月年周天])", query)
            and any(w in query for w in RECENT_VAGUE_WORDS)
    )
    if latest_hit or vague_recent:
        models = _search_global("publish_date", True, "publish_date")[:5]
        set_last_models(session_id, models)
        return format_models(models, "最近发布的机器人有这几款：")

    # 全局价格极值：最贵/最便宜 → 全局按价格排序取极值（不依赖上一轮）
    if "最贵" in query or "价格最高" in query:
        models = _search_global("price")
        if models:
            set_last_models(session_id, models[-1:])
            return format_models(models[-1:], "目前最贵的是这一款：")
    if "最便宜" in query or "价格最低" in query:
        models = _search_global("price")
        if models:
            set_last_models(session_id, models[:1])
            return format_models(models[:1], "目前最便宜的是这一款：")

    # 相对上一轮展示范围的追问：更便宜 / 更贵 / 更新 / 更旧
    rels, conflicts = _relative_parse(query)
    if rels:
        prev = get_last_models(session_id)
        if not prev:
            return None
        field = RELATIVE_SPEC[rels[0]][0]
        got = _relative_slice(prev, _search_global(field, False, field), rels)
        if not got:
            return _rel_none_msg(rels)
        set_last_models(session_id, got)
        reply = format_models(got, _rel_header(rels))
        if conflicts:
            hint = "我猜您想要的是「%s」，先按这个看：\n" % "、".join(
                REL_LABEL[r] for r in conflicts)
            reply = hint + reply
        return reply

    return None


def _search_global(sort_key: str = "price", reverse: bool = False, require_field: str = "price"):
    """全局枚举所有型号并按指定字段排序（价格升序 / 发布时间倒序等）。"""
    models = enumerate_models({"file_name": {"$ne": "__never__"}}, require_field=require_field)
    default = "" if sort_key == "publish_date" else 0
    models.sort(key=lambda m: m.get(sort_key) or default, reverse=reverse)
    return models
