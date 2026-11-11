"""模型查询业务：约束解析、直出编排与型号意图判断。

约束 DTO 与预算/日期/系列解析在本模块；型号筛选在 `service.model_catalog`。
S9 撤销后不再保留独立 domain 层，避免只为一个消费者维护半挂目录。
"""
from __future__ import annotations

import json
from dataclasses import asdict

from tools import trace_store
from tools.dto.model_query import (FindModelsResult, ModelQueryConstraints,
                                   PriceRange, PublishRange)
from tools.log_tool import get_logger
from tools.metadata_extractor import extract_series, format_model_line, resolve_budget_range
from tools.prompts_tool import render
from service.model_catalog import find_models
from tools.date_rules import parse_date, publish_int
from config.word_dict_config import (
    BRAND_EVENT_WORDS,
    DATE_SELECT_HINT_RE,
    MODEL_QUERY_RESULT_PREFIX,
    PURCHASE_DURATION_RE,
    RELEASE_HINT_RE,
    TAG_CATALOG,
    TIME_HINT_RE,
)

logger = get_logger(name="model_query")


def parse_budget(query: str, allow_llm: bool = True) -> PriceRange | None:
    """预算表达 → PriceRange；规则优先，必要时 LLM 兜底。"""
    rng = resolve_budget_range(query, allow_llm=allow_llm)
    if rng is None:
        return None
    return PriceRange(rng[0], rng[1])


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
    from tools.llm_extract import extract_date_range

    return extract_date_range(query)


def parse_publish(query: str, allow_llm: bool = True) -> PublishRange | None:
    """发布时间表达 → PublishRange（内部统一 YYYYMMDD）。"""
    r = resolve_date_range(query, allow_llm=allow_llm)
    if not r:
        return None
    start, end = publish_int(r[0]), publish_int(r[1])
    if not start:
        return None
    return PublishRange(start, end or None)


def resolve_model_constraints(query: str, allow_llm: bool = True):
    """把 query 收敛成 (ModelQueryConstraints, kind)；allow_llm=False 只走规则。"""
    budget = parse_budget(query, allow_llm=allow_llm)
    publish = None
    if not any(w in query for w in BRAND_EVENT_WORDS):
        publish = parse_publish(query, allow_llm=allow_llm)
    # 日期常是陪衬：没有预算、也没有发布/选型信号时不独立直出
    if publish and not budget and not DATE_SELECT_HINT_RE.search(query):
        logger.info("[ModelQuery] date without selection hint, skip: %s", query[:40])
        publish = None
    series = extract_series(query) if is_model_catalog_intent(query) else ""
    if not (budget or publish or series):
        return None
    if series and not (budget or publish):
        kind = "series"
    elif budget:
        kind = "budget+date" if publish else "budget"
    elif publish:
        kind = "date"
    else:
        kind = "series"
    constraints = ModelQueryConstraints(
        budget=budget,
        publish=publish,
        series=series or "",
        limit=50,
    )
    return constraints, kind


def constraints_to_slots(query: str) -> dict:
    """把已解析约束转成 model_query SOP 的预填槽。"""
    resolved = resolve_model_constraints(query)
    if not resolved:
        return {}
    constraints, _kind = resolved
    slots = {}
    if constraints.budget is not None:
        slots["budget"] = (constraints.budget.min, constraints.budget.max)
    if constraints.series:
        slots["series"] = constraints.series
    if constraints.publish is not None:
        slots["publish_range"] = (constraints.publish.start, constraints.publish.end)
    return slots


def format_constraints(constraints: ModelQueryConstraints) -> str:
    """把已校验约束渲染成确定性的中文条件串。"""
    parts = []
    b = constraints.budget
    if b is not None:
        if b.min is not None and b.max is not None:
            parts.append("预算 %s–%s 元" % (b.min, b.max))
        elif b.max is not None:
            parts.append("预算 %s 元以内" % b.max)
        elif b.min is not None:
            parts.append("预算 %s 元以上" % b.min)
    p = constraints.publish
    if p is not None and p.start is not None:
        start = _fmt_day(p.start)
        end = _fmt_day(p.end) if p.end else start
        parts.append("%s 至 %s 发布" % (start, end) if start != end else "%s 发布" % start)
    if constraints.model_names:
        parts.append("型号 " + "、".join(constraints.model_names))
    if constraints.series:
        parts.append("%s 系列" % constraints.series)
    for cond in constraints.params:
        parts.append("%s %s %s" % (cond.key, cond.op, cond.value))
    if constraints.scope:
        labels = {"cheaper": "更便宜", "pricier": "更贵", "newer": "更新", "older": "更旧"}
        parts.extend(labels.get(s, s) for s in constraints.scope)
    if constraints.tags:
        parts.extend((TAG_CATALOG.get(t) or {}).get("label", t) for t in constraints.tags)
    return "、".join(parts) or "全部型号"


def _fmt_day(value: int) -> str:
    text = str(value)
    return "%s-%s-%s" % (text[:4], text[4:6], text[6:8]) if len(text) == 8 else text


def has_model_ref(query: str) -> bool:
    """query 是否指向具体型号：指代词或直接出现的型号名。"""
    from config.word_dict_config import MODEL_REF_WORDS

    if any(w in query for w in MODEL_REF_WORDS):
        return True
    try:
        from service.model_catalog import model_name_in_query
        return model_name_in_query(query)
    except Exception as e:  # noqa: BLE001
        logger.warning("[ModelQuery] model name check failed: %s", e)
        return False


def is_model_detail_query(query: str) -> bool:
    """型号详情/对比入口：强触发词，或咨询词 + 具体型号指代。"""
    from config.word_dict_config import MODEL_CONSULT_WORDS, MODEL_QUERY_WORDS

    if any(w in query for w in MODEL_QUERY_WORDS):
        return True
    return any(w in query for w in MODEL_CONSULT_WORDS) and has_model_ref(query)


def is_model_catalog_intent(query: str) -> bool:
    """型号/系列枚举意图，覆盖「哪几款 / 有哪些 / 型号清单」等 catalog 问法。"""
    from config.word_dict_config import MODEL_LIST_WORDS

    words = tuple(MODEL_LIST_WORDS) + ("哪几款", "有哪些")
    return any(w in query for w in words)


def resolve_model_query(query: str):
    """按预算/日期筛选型号；无约束或无结果返回 None，由 agent 回退 RAG。"""
    resolved = resolve_model_constraints(query)
    if resolved is None:
        return None
    constraints, kind = resolved
    result = find_models(constraints)
    models = result.models
    status = "degraded" if result.dropped else ("hit" if models else "miss")
    out = {"n": len(models)}
    if result.dropped:
        out["dropped"] = ",".join(result.dropped)
    if result.notes:
        out["notes"] = json.dumps(result.notes, ensure_ascii=False)
    trace_store.node("find_models", status=status,
                     in_={"kind": kind,
                          "constraint_detail": json.dumps(asdict(constraints),
                                                          ensure_ascii=False, default=str)},
                     out=out)
    if result.dropped or not models:
        return None
    return FindModelsResult(models=models, dropped=result.dropped, notes=result.notes,
                            constraints=constraints, kind=kind)


def format_model_query(result: FindModelsResult) -> str:
    """统一条件前缀 + 型号列表；条件只来自实际生效的 constraints。"""
    n = len(result.models)
    constraint_text = format_constraints(result.constraints or ModelQueryConstraints())
    prefix = render("model_query_result", template=MODEL_QUERY_RESULT_PREFIX,
                    condition=constraint_text, number=n) or ""
    return prefix + "\n\n" + "\n".join(format_model_line(m) for m in result.models)
