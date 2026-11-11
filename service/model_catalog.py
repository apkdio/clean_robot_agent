"""型号搜索工具：让模型自主决策要查哪个型号，再按型号名精准检索型号详情。

用于「型号详情咨询 / 上下文对比」兜底（处理指代「这两个/它」、对比「区别」）。
型号名提取走 `service.llm_extract`（主生成模型）；本模块只负责筛选、
排序、参数校验等确定性能力，不直接发起 LLM 调用。
"""

from __future__ import annotations

import re
from typing import Dict, List

from config.word_dict_config import (
    ABOVE_RE as _ABOVE_RE,
    BELOW_RE as _BELOW_RE,
    ENUM_RE as _ENUM_RE,
    PRICE_KEYS as _PRICE_KEYS,
    PUBLISH_KEYS as _PUBLISH_KEYS,
    TAG_CATALOG as _TAG_CATALOG,
)
from utils.date_rules import publish_int
from utils.dto.model_query import (FindModelsResult, ModelQueryConstraints,
                                   ParamFilter, PriceRange, PublishRange)
from utils.log_tool import get_logger
from mapper.models_repo import all_records, match_key

logger = get_logger(name="model_tool")


def _enumerate_model_infos() -> List[Dict]:
    """枚举全量型号信息（去重），复用 metadata_extractor 的条目校验。"""
    from service.metadata_query import enumerate_models
    return enumerate_models({"file_name": {"$ne": "__never__"}}, require_field="price")


def get_all_model_names() -> List[str]:
    """全量型号名列表，直接从 models_repo 缓存派生（不再另建名单缓存）。"""
    return [m["name"] for m in _enumerate_model_infos() if m.get("name")]


def search_models_by_names(model_names: str) -> List[Dict]:
    """根据型号名（逗号分隔，LLM 提取结果）检索型号详情。

    两段式匹配（见 BC-20260921-06）：① 去品牌前缀后以查询名结尾算精确命中，
    任一 key 精确命中时只返回精确结果（「净界 P2」不再带出 P2 Lite / P2 Pro）；
    ② 精确全未命中才退回子串匹配，兼容 LLM 偶发把系列名当型号（「云顶 X」）。
    """
    names = [n.strip() for n in (model_names or "").split(",") if n.strip()]
    if not names:
        return []

    models = _enumerate_model_infos()
    keys = [match_key(n.replace("系列", "")) for n in names]
    exact = [m for m in models if any(k and match_key(m.get("name", "")).endswith(k) for k in keys)]
    if exact:
        return exact

    return [m for m in models if any(k and k in match_key(m.get("name", "")) for k in keys)]


def model_name_in_query(query: str) -> bool:
    """判断 query 里是否直接出现具体型号名（去品牌前缀、去空白后子串匹配）。

    如型号「不染一尘云顶 X2」→ 匹配键「云顶X2」，query 含「云顶X2」或「云顶 X2」
    即命中。用于「云顶 X2 怎么样」这类直接报型号名的详情咨询。
    """
    flat = re.sub(r"\s+", "", query)
    for name in get_all_model_names():
        short = match_key(name)
        if short and short in flat:
            return True
    return False




def _param_of(record: Dict, key: str) -> str:
    """取某型号的规格原文（param_<键>）；特殊键「参考价」映射到 price。"""
    if key in _PRICE_KEYS:
        return str(record.get("price") or "")
    return str(record.get(f"param_{key}") or "")


def _param_num_of(record: Dict, key: str):
    """取某型号规格的数值（param_<键>_num）；特殊键「参考价」映射到 price。"""
    if key in _PRICE_KEYS:
        return record.get("price")
    return record.get(f"param_{key}_num")


# 相对上一轮展示边界：scope -> (记录字段, 边界函数, 比较方向, 展示名, 降序?)
# field 用真实记录字段，followup / find_models / 格式化共用一套语义。
RELATIVE_SPEC = {
    "cheaper": ("price", min, "lt", "参考价", True),
    "pricier": ("price", max, "gt", "参考价", False),
    "newer": ("publish_date", max, "gt", "发布时间", False),
    "older": ("publish_date", min, "lt", "发布时间", True),
}


def _bound_value(record: Dict, field: str) -> int:
    """取价格或发布时间（数值化），用于相对上一轮边界的比较。"""
    return int(record.get("price") or 0) if field == "price" else _publish_int(record)


def _filter_relative(models: List[Dict], prev_models, scope: str) -> List[Dict]:
    """按 scope 过滤出越过上一轮展示边界的型号（更便宜/更贵/更新/更旧）。"""
    field, pick, op, _label, _desc = RELATIVE_SPEC[scope]
    edges = [_bound_value(m, field) for m in (prev_models or [])]
    edges = [e for e in edges if e]
    if not edges:
        return []
    edge = pick(edges)
    return [m for m in models
            if _bound_value(m, field) and
            (_bound_value(m, field) < edge if op == "lt" else _bound_value(m, field) > edge)]


def filter_by_relative(models: List[Dict], prev_models, rels) -> List[Dict]:
    """按相对维度过滤并排序：先同时满足所有维度，再按第一个维度取“离边界最近”。

    followup 和 find_models 共用这里；调用方只负责把 rels 解析成有序列表。
    """
    rels = [r for r in (rels or []) if r in RELATIVE_SPEC]
    for rel in rels:
        models = _filter_relative(models, prev_models, rel)
    if not rels or not models:
        return []
    field, _pick, _op, _label, desc = RELATIVE_SPEC[rels[0]]
    return sorted(models, key=lambda m: _bound_value(m, field), reverse=desc)


def _scope_list(scope) -> List[str]:
    """把 scope 归一成小写取值列表（支持字符串、逗号串、数组）。"""
    if isinstance(scope, (list, tuple)):
        raw = [str(s) for s in scope]
    else:
        raw = str(scope or "").replace("，", ",").split(",")
    return [s.strip().lower() for s in raw if s.strip()]


def _relative_scopes(scope) -> List[str]:
    """scope 里的相对维度（可多选，同时满足）。"""
    return [s for s in _scope_list(scope) if s in RELATIVE_SPEC]


def _sort_value_of(record: Dict, sort_by: str) -> float:
    """排序取值：参考价 → price；发布时间 → publish_date(YYYYMMDD)；其余 → param_<键>_num。"""
    if sort_by in _PRICE_KEYS:
        return float(record.get("price") or 0)
    if sort_by in _PUBLISH_KEYS:
        return float(_publish_int(record))
    return float(_param_num_of(record, sort_by) or 0)


def _publish_int(record: Dict) -> int:
    """型号记录的发布时间 → int YYYYMMDD（统一走 capability.date.publish_int）。"""
    return publish_int(record.get("publish_date"))


def _drop(dropped: List[str], notes: List[str], field: str, detail: str) -> None:
    """记录一个被丢弃的参数；调用方据此走 degraded / 澄清，不能静默吞掉。"""
    dropped.append(field)
    notes.append("%s: %s" % (field, detail))
    logger.warning("[ModelTool] dropped %s: %s", field, detail)


def _budget_value(raw, field: str, dropped: List[str], notes: List[str]):
    """预算数值 → 非负 int 或 None；非法值丢弃并返回 None。"""
    if raw in (None, "", 0):
        return None
    if isinstance(raw, bool):
        _drop(dropped, notes, field, "boolean is not a budget")
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        _drop(dropped, notes, field, "non-numeric %r" % (raw,))
        return None
    if value < 0:
        _drop(dropped, notes, field, "negative %r" % (raw,))
        return None
    return value or None


def _publish_bounds(args: Dict, dropped: List[str], notes: List[str]):
    """日期参数 → (lo, hi) int 或 None；非法/短日期丢弃。"""
    start = args.get("publish_start")
    end = args.get("publish_end")
    rng = str(args.get("publish_range") or "").strip()
    if start not in (None, ""):
        lo = publish_int(start)
        if lo < 10000101:
            _drop(dropped, notes, "publish_start", "invalid/short date %r" % (start,))
            return None
        hi = None
        if end not in (None, ""):
            hi = publish_int(end)
            if hi < 10000101:
                _drop(dropped, notes, "publish_end", "invalid/short date %r" % (end,))
                hi = None
        if hi is not None and hi < lo:
            _drop(dropped, notes, "publish_range", "end before start")
            return None
        return lo, hi
    if end not in (None, ""):
        _drop(dropped, notes, "publish_end", "end without start")
    if rng:
        from utils.date_rules import calc_date_range

        parsed = calc_date_range(rng)
        if "start_date" not in parsed:
            _drop(dropped, notes, "publish_range", "unparseable %r" % (rng,))
            return None
        lo = publish_int(parsed["start_date"])
        hi = publish_int(parsed.get("end_date")) or None
        if lo < 10000101:
            _drop(dropped, notes, "publish_range", "unparseable %r" % (rng,))
            return None
        return lo, hi
    return None


def build_constraints(args: Dict, query: str = "") -> tuple:
    """把 LLM/内部 dict 校验成 ModelQueryConstraints，返回 (constraints, dropped, notes)。

    这里只做解析与校验，不筛选型号；非法字段只丢自身并记录，供调用方决定降级/澄清。
    """
    args = dict(args or {})
    semantic_notes: List[str] = []
    if query:
        args, semantic_notes = validate_find_models_args(query, args)
    dropped: List[str] = []
    notes: List[str] = list(semantic_notes)

    raw_scopes = _scope_list(args.get("scope"))
    scopes = []
    for scope in raw_scopes:
        if scope in RELATIVE_SPEC:
            scopes.append(scope)
        else:
            _drop(dropped, notes, "scope", "unknown %r" % (scope,))

    budget = None
    phrase = str(args.get("budget") or "").strip()
    bmin = bmax = None
    if phrase:
        from utils.metadata_rules import extract_price_constraint
        parsed = extract_price_constraint(phrase)
        if parsed is not None and (parsed[0] is not None or parsed[1] is not None):
            bmin, bmax = parsed[0] or None, parsed[1] or None
        else:
            _drop(dropped, notes, "budget", "unparseable %r" % (phrase,))
    if bmin is None and bmax is None:
        bmin = _budget_value(args.get("budget_min"), "budget_min", dropped, notes)
        bmax = _budget_value(args.get("budget_max"), "budget_max", dropped, notes)
    if bmin is not None and bmax is not None and bmin > bmax:
        _drop(dropped, notes, "budget", "min > max (%s > %s)" % (bmin, bmax))
        bmin = bmax = None
    if bmin is not None or bmax is not None:
        budget = PriceRange(bmin, bmax)

    bounds = _publish_bounds(args, dropped, notes)
    publish = PublishRange(*bounds) if bounds is not None else None

    model_names = []
    model_arg = args.get("model")
    if model_arg not in (None, ""):
        if not isinstance(model_arg, str):
            _drop(dropped, notes, "model", "not a string %r" % (model_arg,))
        else:
            model_names = [k.strip() for k in model_arg.split(",") if k.strip()]

    series = ""
    series_arg = args.get("series")
    if series_arg not in (None, ""):
        if not isinstance(series_arg, str):
            _drop(dropped, notes, "series", "not a string %r" % (series_arg,))
        else:
            series = series_arg.strip()

    params = []
    raw_params = args.get("params")
    if raw_params not in (None, ""):
        if not isinstance(raw_params, (list, tuple)):
            _drop(dropped, notes, "params", "not a list")
            raw_params = []
        for idx, cond in enumerate(raw_params):
            field = "params[%d]" % idx
            if not isinstance(cond, dict):
                _drop(dropped, notes, field, "not an object")
                continue
            key = str(cond.get("key") or "").strip()
            op = str(cond.get("op") or "含").strip()
            val = str(cond.get("value") or "").strip()
            if not key or not val:
                _drop(dropped, notes, field, "empty key/value")
                continue
            if op not in ("含", "=", ">=", "<="):
                _drop(dropped, notes, field, "unknown op %r" % (op,))
                continue
            if op in (">=", "<=") and not _is_number(val):
                _drop(dropped, notes, field, "non-numeric comparison %r" % (val,))
                continue
            params.append(ParamFilter(key, op, val))

    sort_by = str(args.get("sort_by") or "").strip()
    if sort_by and sort_by not in (_PRICE_KEYS + _PUBLISH_KEYS):
        if not any(_param_num_of(m, sort_by) is not None for m in all_records()):
            _drop(dropped, notes, "sort_by", "unknown %r" % (sort_by,))
            sort_by = ""
    order = str(args.get("order") or "desc").strip().lower()
    if order not in ("asc", "desc"):
        _drop(dropped, notes, "order", "unknown %r" % (order,))
        order = "desc"

    raw_tags = args.get("tags")
    tags = []
    if raw_tags not in (None, ""):
        if not isinstance(raw_tags, (list, tuple)):
            _drop(dropped, notes, "tags", "not a list")
        else:
            for tag in raw_tags:
                tag = str(tag).strip()
                if tag and tag not in _TAG_CATALOG:
                    _drop(dropped, notes, "tags", "unknown %r" % (tag,))
                elif tag:
                    tags.append(tag)

    raw_limit = args.get("limit")
    if raw_limit in (None, ""):
        limit = 5
    else:
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            _drop(dropped, notes, "limit", "non-integer %r" % (raw_limit,))
            limit = 5
        if limit <= 0:
            _drop(dropped, notes, "limit", "must be positive")
            limit = 5
    limit = max(1, min(limit, 50))

    constraints = ModelQueryConstraints(
        budget=budget,
        publish=publish,
        model_names=tuple(model_names),
        series=series,
        params=tuple(params),
        scope=tuple(scopes),
        tags=tuple(tags),
        sort_by=sort_by,
        order=order,
        limit=limit,
    )
    return constraints, list(dict.fromkeys(dropped)), notes


def find_models(constraints: ModelQueryConstraints,
                prev_models: "List[Dict] | None" = None) -> FindModelsResult:
    """按已校验的 ModelQueryConstraints 执行筛选 → 排序 → limit。"""
    models = all_records()

    rels = [s for s in constraints.scope if s in RELATIVE_SPEC]
    if rels:
        # 多维度同时满足：「更新更便宜的」= 比上一轮更新 且 更便宜
        models = filter_by_relative(models, prev_models, rels)

    if constraints.budget is not None:
        if constraints.budget.min is not None:
            models = [m for m in models if int(m.get("price") or 0) >= constraints.budget.min]
        if constraints.budget.max is not None:
            models = [m for m in models if int(m.get("price") or 0) <= constraints.budget.max]

    if constraints.publish is not None:
        lo = constraints.publish.start
        hi = constraints.publish.end or 99991231
        if lo is not None:
            models = [m for m in models if lo <= _publish_int(m) <= hi]

    if constraints.model_names:
        keys = [match_key(k) for k in constraints.model_names]
        exact = [m for m in models if any(match_key(m["name"]).endswith(k) for k in keys)]
        models = exact or [m for m in models if any(k in match_key(m["name"]) for k in keys)]

    if constraints.series:
        want = match_key(constraints.series.replace("系列", ""))
        models = [m for m in models if want in match_key(str(m.get("series") or ""))]

    for cond in constraints.params:
        if cond.op in (">=", "<="):
            threshold = float(cond.value)
            models = [m for m in models
                      if _param_num_of(m, cond.key) is not None
                      and (float(_param_num_of(m, cond.key)) >= threshold if cond.op == ">=" else float(_param_num_of(m, cond.key)) <= threshold)]
        else:
            models = [m for m in models if cond.value in _param_of(m, cond.key)]

    if constraints.sort_by:
        models = sorted(models, key=lambda m: _sort_value_of(m, constraints.sort_by),
                        reverse=(constraints.order != "asc"))

    if constraints.tags:
        filter_tags = [t for t in constraints.tags if (_TAG_CATALOG.get(t) or {}).get("mode") == "filter"]
        boost_tags = [t for t in constraints.tags if (_TAG_CATALOG.get(t) or {}).get("mode") == "boost"]
        if filter_tags:
            models = [m for m in models
                      if all(m.get(f"tag_{tag}") is True for tag in filter_tags)]
        if boost_tags:
            # 稳定分组：有任一 boost 标签的排前面，组内保留原排序
            models = sorted(models, key=lambda m: 0 if any(
                m.get(f"tag_{tag}") is True for tag in boost_tags) else 1)

    return FindModelsResult(models[:constraints.limit], [], [], constraints)


def find_models_result(args: Dict, prev_models: "List[Dict] | None" = None) -> FindModelsResult:
    """LLM/内部 dict → 校验后的 constraints → 查询结果。"""
    constraints, dropped, notes = build_constraints(args)
    result = find_models(constraints, prev_models)
    return FindModelsResult(result.models, dropped, notes, constraints)


def _spec_values_of(key: str) -> List[str]:
    """取某规格键在型号缓存里的全部取值（用于校验 LLM 填的键/值）。"""
    field = "param_%s" % key
    return [str(record.get(field)) for record in all_records()
            if record.get(field) not in (None, "")]


def validate_find_models_args(query: str, args: Dict) -> tuple:
    """按用户原话纠正模型填的 find_models 参数，返回 (纠正后的 args, 纠正说明)。"""
    args = dict(args or {})
    notes: List[str] = []
    query = query or ""

    # ① 方向：原话说「以上」却只给了上限（反之亦然）→ 按原话翻面
    bmin, bmax = args.get("budget_min") or 0, args.get("budget_max") or 0
    if _ABOVE_RE.search(query) and bmax and not bmin:
        args["budget_min"], args["budget_max"] = bmax, 0
        notes.append("方向纠正：原话是「以上」，budget_max=%s → budget_min" % bmax)
    elif _BELOW_RE.search(query) and bmin and not bmax:
        args["budget_max"], args["budget_min"] = bmin, 0
        notes.append("方向纠正：原话是「以内/以下」，budget_min=%s → budget_max" % bmin)

    # budget 原话能解析出边界时，它是唯一口径，数值侧清空避免两套口径打架
    phrase = str(args.get("budget") or "").strip()
    if phrase:
        from utils.metadata_rules import extract_price_constraint
        if extract_price_constraint(phrase):
            if args.get("budget_min") or args.get("budget_max"):
                notes.append("原话「%s」可解析，忽略模型填的数值预算" % phrase)
            args["budget_min"] = args["budget_max"] = 0

    terms, kept = [], []
    for cond in args.get("params") or []:
        if not isinstance(cond, dict):
            notes.append("剔除非对象条件 %r" % (cond,))
            continue
        key = str(cond.get("key") or "").strip()
        op = str(cond.get("op") or "含").strip()
        val = str(cond.get("value") or "").strip()
        cond = dict(cond, key=key, op=op, value=val)
        if not key or not val:
            notes.append("剔除空条件（key/value 为空）")
            continue

        # ② 数值比较却填了非数字（「大吸力」）→ 该条件恒不成立，剔掉；软偏好提升为排序
        if op in (">=", "<=") and not _is_number(val):
            if not (args.get("sort_by") or "").strip():
                args["sort_by"], args["order"] = key, "desc"
                notes.append("「%s」（「%s」）不是数值，从筛选条件提升为排序依据" % (key, val))
            else:
                notes.append("剔除无效数值条件：%s %s %s" % (key, op, val))
            continue

        # 价格类条件与 budget 原话重复 → 只留一份口径
        if phrase and key in _PRICE_KEYS:
            notes.append("剔除与 budget 原话重复的价格条件 %s %s %s" % (key, op, val))
            continue
        # 价格类条件方向与原话相悖（同样源于「预算 X 元」模板）→ 按原话翻面
        if key in _PRICE_KEYS and op in (">=", "<="):
            if _ABOVE_RE.search(query) and op == "<=":
                notes.append("方向纠正：参考价 <= → >=（原话是「以上」）")
                cond["op"] = ">="
            elif _BELOW_RE.search(query) and op == ">=":
                notes.append("方向纠正：参考价 >= → <=（原话是「以内/以下」）")
                cond["op"] = "<="

        # 键/值不存在 → 不纠正（该条件命中 0 条会走兜底），只记一笔便于排查
        values = _spec_values_of(key)
        if values and not any(values):
            notes.append("未知规格键「%s」（知识库无此参数，将命中 0 条）" % key)
        elif values and not any(val in v for v in values):
            notes.append("值「%s」在知识库 %s 里不存在（将命中 0 条）" % (val, key))
        kept.append(cond)
    args["params"] = kept

    sort_by = str(args.get("sort_by") or "").strip()
    if sort_by and sort_by not in (_PRICE_KEYS + _PUBLISH_KEYS):
        values = _spec_values_of(sort_by)
        if values and not any(values):
            notes.append("排序键「%s」在知识库里不存在（将按 0 排序）" % sort_by)

    # 相对维度（更便宜/更贵/更新/更旧）的排序由规则固定：最接近上一轮边界的优先；多维度取第一个
    _rels = _relative_scopes(args.get("scope"))
    if _rels:
        spec = RELATIVE_SPEC[_rels[0]]
        want_key, want_order = spec[3], "desc" if spec[4] else "asc"
        if (args.get("sort_by") or want_key) != want_key or (args.get("order") or want_order) != want_order:
            notes.append("scope=%s：排序已按「%s + %s」处理" % ("+".join(_rels), want_key, want_order))
        args["sort_by"], args["order"] = want_key, want_order

    # ③ 枚举类问法（「有哪些」）不要拿 5 条截断结果
    if _ENUM_RE.search(query):
        try:
            cur = int(args.get("limit") or 5)
        except (TypeError, ValueError):
            cur = 5
        if cur < 20:
            args["limit"] = 20
            notes.append("枚举类问法，limit %s → 20" % cur)
    return args, notes


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
