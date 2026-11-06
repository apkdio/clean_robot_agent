"""选购/型号查询 SOP：预算 / 系列 / 发布时间 / 标签补槽，最后调用 find_models。"""
from __future__ import annotations

from sops.base import SKIP, is_aftersales, is_brand, is_consulting, register
from config.word_dict_config import (
    BARE_BUDGET_RE,
    MODEL_QUERY_CONSTRAINT_RETRY_REPLY,
    MODEL_QUERY_EMPTY_REPLIES,
    MODEL_QUERY_RESULT_PREFIX,
    MODEL_QUERY_TRIGGER,
    MODEL_QUERY_WEAK_TRIGGER,
    TAG_CATALOG,
    TAG_ORDER,
)
from function_tools.model_tool import find_models_result
from tools import trace_store
from tools.log_tool import get_logger
from tools.metadata_extractor import extract_price_constraint, extract_series, format_model_line
from tools.model_query import format_constraints, parse_publish
from tools.prompts_tool import render
from tools.selection import is_skip, parse_multi_select

logger = get_logger(name="model_query_sop")


def _skip_text(text: str) -> bool:
    return is_skip(text)


def _extract_budget(text: str, _slots: dict):
    if _skip_text(text):
        return SKIP
    parsed = extract_price_constraint(text)
    if parsed:
        return parsed
    m = BARE_BUDGET_RE.match(text or "")
    return (None, int(m.group(1))) if m else None


def _extract_series(text: str, _slots: dict):
    if _skip_text(text):
        return SKIP
    return extract_series(text) or None


def _extract_publish(text: str, _slots: dict):
    if _skip_text(text):
        return SKIP
    rng = parse_publish(text)
    return (rng.start, rng.end) if rng else None


def _extract_tags(text: str, _slots: dict):
    kind, nums = parse_multi_select(text, len(TAG_ORDER))
    if kind == "skip":
        return SKIP
    if kind != "ok":
        return None
    return [TAG_ORDER[i - 1] for i in nums]


def _tag_ask(_ctx: dict) -> str:
    lines = ["还需要哪些特点？回复序号，多个用逗号；不需要就回复「跳过」。"]
    for i, tag_id in enumerate(TAG_ORDER, 1):
        lines.append(f"{i} {TAG_CATALOG[tag_id]['label']}")
    return "\n".join(lines)


def _pair(value, cast=None):
    """校验可空二元组；形状/类型不对返回 None，不把字符串切片当约束。"""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    try:
        return tuple(None if v is None else (cast(v) if cast else v) for v in value)
    except (TypeError, ValueError):
        return None


def _search(slots: dict):
    args = {"limit": 50}
    status = slots.get("_slot_status") or {}
    dropped = []

    budget = slots.get("budget")
    if budget is not None:
        pair = _pair(budget, int)
        if pair is None:
            dropped.append("budget")
        else:
            low, high = pair
            if low is not None:
                args["budget_min"] = low
            if high is not None:
                args["budget_max"] = high
    if status.get("budget") == "dropped":
        dropped.append("budget")

    series = slots.get("series")
    if series:
        args["series"] = series
    if status.get("series") == "dropped":
        dropped.append("series")

    publish = slots.get("publish_range")
    if publish is not None:
        pair = _pair(publish)
        if pair is None:
            dropped.append("publish_range")
        else:
            args["publish_start"], args["publish_end"] = pair
    if status.get("publish_range") == "dropped":
        dropped.append("publish_range")

    tags = slots.get("tags") or []
    if tags:
        args["tags"] = list(tags)
    if status.get("tags") == "dropped":
        dropped.append("tags")

    if dropped:
        dropped = list(dict.fromkeys(dropped))
        trace_store.node("find_models", status="degraded", by="slot_invalid",
                         in_={"dropped": ",".join(dropped)}, out={"n": 0})
        logger.warning("[ModelQuery] dropped invalid/abandoned slots: %s", dropped)
        return {"count": 0, "list": "", "models": [],
                "error": "invalid_slots", "dropped": dropped}

    result = find_models_result(args)
    if result.dropped:
        trace_store.node("find_models", status="degraded", by="invalid_args",
                         in_={"dropped": ",".join(result.dropped)},
                         out={"n": 0})
        logger.warning("[ModelQuery] dropped find_models args: %s", result.dropped)
        return {"count": 0, "list": "", "models": [],
                "error": "invalid_slots", "dropped": result.dropped}
    models = result.models
    lines = [format_model_line(m) for m in models]
    return {"count": len(models), "list": "\n".join(lines), "models": models,
            "condition": format_constraints(result.constraints)}


def _make_reply(templates: dict):
    """按结果分支选择 reply 文案；模板由 SOP 的 reply 步骤传入。"""
    def reply(ctx: dict) -> str:
        if ctx.get("error") == "invalid_slots":
            return templates.get("invalid") or MODEL_QUERY_CONSTRAINT_RETRY_REPLY
        n = int(ctx.get("count") or 0)
        if n <= 0:
            return templates.get("empty") or MODEL_QUERY_EMPTY_REPLIES[0]
        return render(
            "sop:model_query:reply",
            template=templates.get("ok") or "",
            condition=ctx.get("condition") or "全部型号",
            number=n, count=n, models=ctx.get("list") or "",
        ) or ""
    return reply


MODEL_QUERY_SOP = {
    "id": "model_query",
    "trigger": MODEL_QUERY_TRIGGER,
    "enter_confirm": "always",
    "weak_trigger": MODEL_QUERY_WEAK_TRIGGER,
    "guards": [
        {"check": is_consulting},
        {"check": is_aftersales},
        {"check": is_brand},
    ],
    "intro": "好的，我来帮您筛选合适的型号～",
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
            "template": _make_reply({
                "invalid": MODEL_QUERY_CONSTRAINT_RETRY_REPLY,
                "empty": MODEL_QUERY_EMPTY_REPLIES[0],
                "ok": MODEL_QUERY_RESULT_PREFIX + "\n\n$models\n\n需要我帮您对比其中两款吗？",
            }),
        },
    ],
}


register(MODEL_QUERY_SOP)
