"""模型查询业务：预算/日期/系列约束 → find_models → 结构化直出。

只负责“筛选并拼出型号列表”，不处理型号详情、追问或 SOP。
"""
from __future__ import annotations

from tools import trace_store
from tools.dto.model_query import ModelQueryResult
from tools.log_tool import get_logger
from tools.metadata_extractor import (
    build_filter,
    extract_series,
    format_model_line,
    resolve_budget_filter,
)
from function_tools.date_tool import DATE_TOOL_SCHEMA, calc_date_range, parse_date
from function_tools.model_tool import find_models_by_args
from config.word_dict_config import (
    BRAND_EVENT_WORDS,
    DATE_SELECT_HINT_RE,
    PURCHASE_DURATION_RE,
    RELEASE_HINT_RE,
    TIME_HINT_RE,
)

logger = get_logger(name="model_query")


def _budget_bounds(query: str, allow_llm: bool = True) -> tuple[int, int]:
    """预算上下限；allow_llm=False 时只走规则（供操作判别/打分）。"""
    f = resolve_budget_filter(query) if allow_llm else build_filter(query)
    if not f:
        return 0, 0
    low = high = 0

    def walk(node):
        nonlocal low, high
        if not isinstance(node, dict):
            return
        if "$and" in node:
            for sub in node["$and"]:
                walk(sub)
            return
        cond = node.get("min_price")
        if isinstance(cond, dict) and "$lte" in cond:
            high = max(high, int(cond["$lte"]))
        cond = node.get("max_price")
        if isinstance(cond, dict) and "$gte" in cond:
            low = max(low, int(cond["$gte"]))

    walk(f)
    return low, high


def resolve_date_range(query: str, allow_llm: bool = True):
    """规则优先、LLM FC 兜底，返回 (start_date, end_date) 或 None。"""
    if PURCHASE_DURATION_RE.search(query) and not RELEASE_HINT_RE.search(query):
        logger.info("[ModelQuery] purchase duration, skip date range")
        return None
    r = parse_date(query)
    if r:
        return r[0], r[1]
    if not TIME_HINT_RE.search(query) or not allow_llm:
        return None
    try:
        from langchain_core.messages import HumanMessage
        from tools.llm_tool import chat_with_tools, get_chat_model_name

        resp = chat_with_tools(
            [HumanMessage(content=query)],
            [DATE_TOOL_SCHEMA],
            model=get_chat_model_name(),
        )
        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            tc = tool_calls[0]
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            result = calc_date_range(args.get("expression", ""))
            if "error" not in result:
                return result["start_date"], result["end_date"]
    except Exception as e:  # noqa: BLE001
        logger.warning("[ModelQuery] date tool calling failed: %s", e)
    return None


def is_model_query_intent(query: str) -> bool:
    """明确在问型号/产品/系列，而不是普通叙述里碰巧出现「有什么」。"""
    words = ("型号", "机型", "产品", "系列", "推荐", "清单", "列表", "款式")
    return any(w in query for w in words)


def resolve_model_constraints(query: str, allow_llm: bool = True):
    """把 query 收敛成 (find_models 参数, kind)；allow_llm=False 只走规则。"""
    low, high = _budget_bounds(query, allow_llm=allow_llm)
    date = None
    if not any(w in query for w in BRAND_EVENT_WORDS):
        date = resolve_date_range(query, allow_llm=allow_llm)
    # 日期常是陪衬：没有预算、也没有发布/选型信号时不独立直出
    if date and not (low or high) and not DATE_SELECT_HINT_RE.search(query):
        logger.info("[ModelQuery] date without selection hint, skip: %s", query[:40])
        date = None
    series = extract_series(query) if is_model_query_intent(query) else ""
    if not (low or high or date or series):
        return None
    if series and not (low or high or date):
        kind = "series"
    elif low or high:
        kind = "budget+date" if date else "budget"
    elif date:
        kind = "date"
    else:
        kind = "series"
    args = {
        "budget_min": low,
        "budget_max": high,
        "publish_start": date[0] if date else "",
        "publish_end": date[1] if date else "",
        "series": series or "",
        "limit": 50,
    }
    return args, kind


def constraints_to_slots(query: str) -> dict:
    """把已解析约束转成 model_query SOP 的预填槽。"""
    resolved = resolve_model_constraints(query)
    if not resolved:
        return {}
    args, _kind = resolved
    slots = {}
    if args.get("budget_min") or args.get("budget_max"):
        slots["budget"] = (args["budget_min"] or None, args["budget_max"] or None)
    if args.get("series"):
        slots["series"] = args["series"]
    if args.get("publish_start"):
        slots["publish_range"] = (args["publish_start"], args["publish_end"])
    return slots


def resolve_model_query(query: str):
    """按预算/日期筛选型号；无约束或无结果返回 None，由 agent 回退 RAG。"""
    resolved = resolve_model_constraints(query)
    if resolved is None:
        return None
    args, kind = resolved
    models = find_models_by_args(args)
    trace_store.node("find_models", status="hit" if models else "ok",
                     in_={"kind": kind, "budget_min": args["budget_min"],
                          "budget_max": args["budget_max"],
                          "publish_start": args["publish_start"] or "-",
                          "publish_end": args["publish_end"] or "-",
                          "series": args.get("series") or "-"},
                     out={"n": len(models)})
    if not models:
        return None
    return ModelQueryResult(models=models, kind=kind, constraints=args)


def format_model_query(result: ModelQueryResult) -> str:
    """按约束类型选前缀，列表统一用 format_model_line。"""
    n = len(result.models)
    series = result.constraints.get("series") or "该"
    prefix = {
        "budget": f"在您预算内的机器人有 {n} 款：",
        "date": f"该时间段内发布的机器人有 {n} 款：",
        "series": f"{series}系列有以下型号 {n} 款：",
        "budget+date": f"符合您预算和时间要求的机器人有 {n} 款：",
    }.get(result.kind, f"符合条件的机器人有 {n} 款：")
    return prefix + "\n\n" + "\n".join(format_model_line(m) for m in result.models)
