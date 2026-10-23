"""型号查询 SOP：预算 / 系列 / 发布时间 / 标签补槽，最后调用 find_models。"""
from __future__ import annotations

from sops.base import SKIP, register
from config.word_dict_config import (
    BARE_BUDGET_RE,
    MODEL_QUERY_EMPTY_REPLIES,
    TAG_ORDER,
)
from function_tools.model_tool import find_models_by_args
from tools.metadata_extractor import extract_price_constraint, extract_series, format_model_line
from tools.model_query import resolve_date_range
from tools.selection import is_skip, parse_multi_select
from tools.model_tags import tag_options


def _skip_text(text: str) -> bool:
    return is_skip(text)


def _extract_budget(text: str, slots: dict):
    if _skip_text(text):
        return SKIP
    parsed = extract_price_constraint(text)
    if parsed:
        return parsed
    m = BARE_BUDGET_RE.match(text or "")
    return (None, int(m.group(1))) if m else None


def _extract_series(text: str, slots: dict):
    if _skip_text(text):
        return SKIP
    return extract_series(text) or None


def _extract_publish(text: str, slots: dict):
    if _skip_text(text):
        return SKIP
    return resolve_date_range(text)


def _extract_tags(text: str, slots: dict):
    kind, nums = parse_multi_select(text, len(TAG_ORDER))
    if kind == "skip":
        return SKIP
    if kind != "ok":
        return None
    return [TAG_ORDER[i - 1] for i in nums]


def _tag_ask(ctx: dict) -> str:
    lines = ["还需要哪些特点？回复序号，多个用逗号；不需要就回复「跳过」。"]
    for i, (tag_id, label) in enumerate(tag_options(), 1):
        lines.append(f"{i} {label}")
    return "\n".join(lines)


def _search(slots: dict):
    args = {"limit": 50}
    budget = slots.get("budget")
    if budget:
        try:
            low, high = budget[0], budget[1]
            args["budget_min"] = int(low or 0)
            args["budget_max"] = int(high or 0)
        except (TypeError, ValueError, IndexError):
            pass
    series = slots.get("series")
    if series:
        args["series"] = series
    publish = slots.get("publish_range")
    if publish:
        try:
            args["publish_start"], args["publish_end"] = publish[0], publish[1]
        except (TypeError, ValueError, IndexError):
            pass
    tags = slots.get("tags") or []
    if tags:
        args["tags"] = list(tags)
    models = find_models_by_args(args)
    lines = [format_model_line(m) for m in models]
    return {"count": len(models), "list": "\n".join(lines), "models": models}


def _reply(ctx: dict) -> str:
    n = int(ctx.get("count") or 0)
    if n <= 0:
        return MODEL_QUERY_EMPTY_REPLIES[0]
    head = "符合您条件的机器人有 %d 款：" % n
    return head + "\n\n" + (ctx.get("list") or "")


MODEL_QUERY_SOP = {
    "id": "model_query",
    "trigger": [],          # 只由 find_models 执行体进入，不参与 match_sop
    "enter_confirm": "never",
    "guards": [],
    "intro": "好的，我来帮您筛选型号～",
    "max_retry": 2,
    "steps": [
        {
            "id": "ask_budget",
            "type": "ask",
            "slot": "budget",
            "ask": "预算大概是多少呢？可以说「3000以内」「2000左右」；不确定就回复「跳过」。",
            "retry": "预算我没太听清，能说个大概数字吗？也可以回复「跳过」。",
            "extract": _extract_budget,
        },
        {
            "id": "ask_series",
            "type": "ask",
            "slot": "series",
            "ask": "有偏好的系列吗？比如「净白 S」「净界 P」；不确定就回复「跳过」。",
            "retry": "系列我没认出来，可以说系列名，或回复「跳过」。",
            "extract": _extract_series,
        },
        {
            "id": "ask_publish",
            "type": "ask",
            "slot": "publish_range",
            "ask": "对发布时间有要求吗？比如「2025年」「最近半年」；不确定就回复「跳过」。",
            "retry": "时间我没太听懂，可以说「2025年」这类范围，或回复「跳过」。",
            "extract": _extract_publish,
        },
        {
            "id": "ask_tags",
            "type": "ask",
            "slot": "tags",
            "ask": _tag_ask,
            "retry": "没认出选项，请回复序号（如「1,3」），或回复「跳过」。",
            "extract": _extract_tags,
        },
        {
            "id": "do_search",
            "type": "action",
            "action": _search,
        },
        {
            "id": "reply",
            "type": "reply",
            "template": _reply,
        },
    ],
}


register(MODEL_QUERY_SOP)
