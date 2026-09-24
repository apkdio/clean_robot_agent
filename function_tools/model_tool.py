"""型号搜索工具：让模型自主决策要查哪个型号，再按型号名精准检索型号详情。

用于「型号详情咨询 / 上下文对比」兜底（处理指代「这两个/它」、对比「区别」）。
与 budget_tool/symptom_tool 不同，这里用主生成模型（默认 7b）——型号名是 string
参数且需理解上下文指代，3b 对 string 参数会原样返回 query、不做提取。
"""

from __future__ import annotations

import re
from typing import Dict, List

from tools.llm_tool import get_chat_model_name

# 型号提取跟随主生成模型（string 参数 + 上下文指代理解，3b 不稳定）
MODEL_TOOL_MODEL = get_chat_model_name()

# 品牌名（型号名前缀，型号上下文判断时去掉，得到「云顶 X2」这类用户口中的型号）
_BRAND_NAME = "不染一尘"


MODEL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_models",
        "description": (
            "从用户的话和对话上下文里，提取用户想查询的扫地机器人具体型号名"
            "（如「云顶 X2」「净白 S1」）。用户可能用指代词（「这两个」「它」）指代上文提到的具体型号，"
            "或直接说型号名。多个型号用逗号分隔。不要提取系列名（如「云顶 X 系列」）。"
            "没有明确型号时返回空字符串。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_names": {
                    "type": "string",
                    "description": "要查询的型号名，多个用逗号分隔（如「云顶 X2」或「云顶 X2,净白 S1」）；无明确型号返回空字符串",
                },
            },
            "required": ["model_names"],
        },
    },
}


def _enumerate_model_infos() -> List[Dict]:
    """枚举全量型号信息（去重），复用 metadata_extractor 的条目校验。"""
    from tools.metadata_extractor import enumerate_models
    return enumerate_models({"file_name": {"$ne": "__never__"}}, require_field="price")


_model_names_cache: List[str] | None = None


def get_all_model_names() -> List[str]:
    """全量型号名列表（模块级缓存），供型号上下文判断用。

    用 require_field="price" 过滤，只保留真正的型号条目（排除品牌介绍 FAQ
    里加粗但无价格的标题，如「不染一尘和云境智能是什么关系？」）。
    """
    global _model_names_cache
    if _model_names_cache is None:
        _model_names_cache = [m["name"] for m in _enumerate_model_infos() if m.get("name")]
    return _model_names_cache


def search_models_by_names(model_names: str) -> List[Dict]:
    """根据型号名（逗号分隔，LLM 提取结果）检索型号详情。

    两段式匹配（见 BC-20260921-06）：① 去品牌前缀后**以查询名结尾**算精确命中，
    任一 key 精确命中时只返回精确结果（「净界 P2」不再带出 P2 Lite / P2 Pro）；
    ② 精确全未命中才退回子串匹配，兼容 LLM 偶发把系列名当型号（「云顶 X」）。
    """
    names = [n.strip() for n in (model_names or "").split(",") if n.strip()]
    if not names:
        return []

    models = _enumerate_model_infos()
    # 两边都去品牌前缀：型号名来自知识库（带「不染一尘」），而 LLM 提取的 key 可能带也可能不带
    def bare(name: str) -> str:
        return name[len(_BRAND_NAME):] if name.startswith(_BRAND_NAME) else name

    keys = [bare(n.replace("系列", "").strip()) for n in names]
    exact = [m for m in models if any(k and bare(m.get("name", "")).endswith(k) for k in keys)]
    if exact:
        return exact

    return [m for m in models if any(k and k in bare(m.get("name", "")) for k in keys)]


def model_name_in_query(query: str) -> bool:
    """判断 query 里是否直接出现具体型号名（去掉品牌前缀后子串匹配）。

    如型号「不染一尘云顶 X2」→ 去掉品牌前缀得「云顶 X2」，query 含「云顶 X2」
    即命中。用于「云顶 X2 怎么样」这类直接报型号名的详情咨询。
    """
    for name in get_all_model_names():
        short = name.replace(_BRAND_NAME, "").strip()
        if short and short in query:
            return True
    return False


# ── find_models：一次调用填完所有结构化约束（P1-5 的工具形态）───────────────
# 与 extract_models（只提取型号名）不同，这个工具承担"型号筛选 + 排序 + 枚举直出"，
# 参数名与知识库结构化元数据对齐：金额 → min_price/max_price，发布区间 → publish_date，
# 参数 → param_<键>（由 extract_spec_metadata 通用解析出来，新增规格键不用改代码）。
FIND_MODELS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_models",
        "description": (
            "按条件筛选/排序扫地机器人型号，直接拿到型号清单（含价格与规格）。"
            "适用：预算推荐、参数筛选（「有没有 LDS 导航的」「有没有红外避障的」）、"
            "排序枚举（「大吸力的几款」「续航最长的」「吸力最强的」）、发布时间区间、系列/型号枚举。"
            "**“大吸力”“最强”“续航长”这类没有具体数字的说法不要自己去定阀值，改写 `sort_by`；"
            "只有在用户说了具体数字时才用 `params` 里的 `>=` / `<=`。"
            "**不要把用户的自然语言直接写进 params 的值里**：值应该是知识库里出现的规格值（如「LDS激光」「红外」）；"
            "不确定有哪些值时，只填 sort_by。"
            "对时间范围，填用户原话（如「最近半年」「2025年」），系统会自己换算。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "budget": {
                    "type": "string",
                    "description": (
                        "预算相关**一律填用户原话**（如「5000元以上」「2000-3000元内」「1500左右」「三千以内」），"
                        "由系统解析方向（以上/以内）、单位与浮动；没有预算填空字符串。"
                        "**不要自己折算成上下限**——「以上」还是「以内」由系统看原话判断。"
                    ),
                },
                "budget_min": {
                    "type": "integer",
                    "description": "预算下限（元）：**仅当用户用中文数字/口语、给不出原话时**（如「预算五千」）才填，否则填 0",
                },
                "budget_max": {
                    "type": "integer",
                    "description": "预算上限（元）：同样仅在给不出原话时才填，否则填 0",
                },
                "publish_range": {
                    "type": "string",
                    "description": "发布时间范围的**原话**（如「最近半年」「2025年三月」「2026年」），由系统换算成区间；没有填空字符串",
                },
                "model": {"type": "string", "description": "型号名（如「云顶 X2」「净界 P2」），多个用逗号分隔；没有填空字符串"},
                "series": {"type": "string", "description": "系列名（如「云顶 X」「净界 P」）；没有填空字符串"},
                "params": {
                    "type": "array",
                    "description": (
                        "规格参数筛选条件，多个条件同时满足。key 用知识库里的规格名（吸力/导航/避障/续航/水箱/基站…）；"
                        "op：含（包含即算命中，如 避障 含 结构光 → 含 3D结构光）、=（完全相等）、>=、<=（数值比较）。"
                        "例：「有没有 LDS 导航的」→ [{\"key\":\"导航\",\"op\":\"含\",\"value\":\"LDS\"}]"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "规格名，如 吸力/导航/避障/续航"},
                            "op": {"type": "string", "enum": ["含", "=", ">=", "<="]},
                            "value": {"type": "string", "description": "规格值或数值，如 LDS / 红外 / 6000"},
                        },
                        "required": ["key", "op", "value"],
                    },
                },
                "sort_by": {
                    "type": "string",
                    "description": "排序依据（规格名，如「吸力」「续航」；或「参考价」）；不排序填空字符串",
                },                "order": {"type": "string", "enum": ["desc", "asc"], "description": "排序方向，默认 desc"},
                "limit": {"type": "integer", "description": "最多返回几条，默认 5"},
            },
            "required": [],
        },
    },
}


def _param_of(record: Dict, key: str) -> str:
    """取某型号的规格原文（param_<键>）；特殊键「参考价」映射到 price。"""
    if key in ("参考价", "价格", "价钱", "价位"):
        return str(record.get("price") or "")
    return str(record.get(f"param_{key}") or "")


def _param_num_of(record: Dict, key: str):
    """取某型号规格的数值（param_<键>_num）；特殊键「参考价」映射到 price。"""
    if key in ("参考价", "价格", "价钱", "价位"):
        return record.get("price")
    return record.get(f"param_{key}_num")


def _publish_int(record: Dict) -> int:
    """型号记录的发布时间 → int YYYYMMDD。

    注意 `extract_model_info` 给的是**文本**（「2026-03-15」），而区间比较要数值，
    所以这里不直接 int()（会抛 ValueError），而是抽数字。
    """
    value = record.get("publish_date")
    if isinstance(value, int):
        return value
    digits = re.sub(r"\D", "", str(value or ""))
    return int(digits) if len(digits) >= 8 else 0


def find_models_by_args(args: Dict) -> List[Dict]:
    """执行 find_models：参数筛选 → 排序 → limit，返回型号记录列表。

    任何单个条件写得不对（如参数键不存在、数值填了非数字）只会让该条件命中 0 条，
    不抛异常——编排层据此决定退回语义检索还是兑底话术。
    """
    from tools.metadata_extractor import get_all_model_records

    args = args or {}
    models = get_all_model_records()

    # 预算优先按**原话**用规则解析：模型只要看到货币单位（「5000元以上」「5000块以上」）
    # 就会切进「预算 X 元」这个惯用模板、把方向词吃掉，稳定填成 budget_max=5000（实测 5/5）；
    # 不填单位反而正确（budget_min）。方向/单位/浮动交给规则，模型只负责把原话搬过来。
    bmin = bmax = 0
    phrase = str(args.get("budget") or "").strip()
    if phrase:
        from tools.metadata_extractor import extract_price_constraint
        parsed = extract_price_constraint(phrase)
        if parsed and (parsed[0] or parsed[1]):
            bmin, bmax = parsed[0] or 0, parsed[1] or 0
    if not bmin and not bmax:
        # 规则解析不出（中文数字「预算五千」、口语「八百多」）→ 退回模型填的数值
        try:
            bmin, bmax = int(args.get("budget_min") or 0), int(args.get("budget_max") or 0)
        except (TypeError, ValueError):
            bmin = bmax = 0
    if bmin:
        models = [m for m in models if int(m.get("price") or 0) >= bmin]
    if bmax:
        models = [m for m in models if int(m.get("price") or 0) <= bmax]

    rng = (args.get("publish_range") or "").strip()
    if rng:
        from function_tools.date_tool import calc_date_range

        parsed = calc_date_range(rng)
        if "start_date" in parsed:
            lo = int(parsed["start_date"].replace("-", ""))
            hi = int((parsed.get("end_date") or "9999-12-31").replace("-", ""))
            models = [m for m in models if lo <= _publish_int(m) <= hi]

    model_arg = (args.get("model") or "").strip()
    if model_arg:
        keys = [k.strip() for k in model_arg.split(",") if k.strip()]
        exact = [m for m in models if any(bare_name(m["name"]).endswith(bare_name(k)) for k in keys)]
        models = exact or [m for m in models if any(bare_name(k) in bare_name(m["name"]) for k in keys)]

    series_arg = (args.get("series") or "").strip()
    if series_arg:
        models = [m for m in models if series_arg.replace("系列", "") in str(m.get("series") or "")]

    for cond in args.get("params") or []:
        if not isinstance(cond, dict):
            continue
        key, op, val = str(cond.get("key") or "").strip(), str(cond.get("op") or "含").strip(), str(cond.get("value") or "").strip()
        if not key or not val:
            continue
        if op in (">=", "<="):
            try:
                threshold = float(val)
            except ValueError:
                models = []
                break
            models = [m for m in models
                      if _param_num_of(m, key) is not None
                      and (float(_param_num_of(m, key)) >= threshold if op == ">=" else float(_param_num_of(m, key)) <= threshold)]
        else:
            models = [m for m in models if val in _param_of(m, key)]

    sort_by = (args.get("sort_by") or "").strip()
    if sort_by:
        order = (args.get("order") or "desc").strip()
        models = sorted(
            models,
            key=lambda m: float(_param_num_of(m, sort_by) or 0),
            reverse=(order != "asc"),
        )

    limit = args.get("limit") or 5
    try:
        limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        limit = 5
    return models[:limit]


def bare_name(name: str) -> str:
    """去掉品牌前缀后的型号名（「不染一尘云顶 X2」→「云顶 X2」）。"""
    return name[len(_BRAND_NAME):] if name.startswith(_BRAND_NAME) else name


# ── 参数校验器：按用户原话纠正模型填的 args（P1-5 基建）───────────────────
# 实测（temp/probe_fc_args.py，多条件下重复 5 次）模型有三类**稳定**错误：
#   ① 看到货币单位就切进「预算 X 元」模板：「5000元以上」稳定填成 budget_max（5/5）；
#   ② 「大吸力」这类模糊说法被塞进 params 做数值比较（吸力 >= "大"）→ 比较恒不成立、结果归零；
#   ③ 问「有哪些」只填 limit=5 → 结果被静默截断。
# 这里只做**确定性**纠正，不去猜用户意图：纠正不了的（未知键、知识库里不存在的值）原样留着，
# 让该条件命中 0 条、由编排层决定退回语义检索还是友好降级。
_ABOVE_RE = re.compile(r"以上|超过|至少|起步|不低于|大于|往上")
_BELOW_RE = re.compile(r"以内|以下|不超过|不到|低于|小于|之内")
_PRICE_KEYS = ("参考价", "价格", "价钱", "价位")
_ENUM_RE = re.compile(r"有哪些|有哪几|都有哪|都有什么|几款|几个|哪些型号")


def _spec_values_of(key: str) -> List[str]:
    """某规格键在知识库里出现过的全部原文（用于判断值是否存在）。"""
    from tools.metadata_extractor import get_all_model_records
    return [str(m.get(f"param_{key}") or "") for m in get_all_model_records()]


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
        from tools.metadata_extractor import extract_price_constraint
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


def run_find_models(query: str, args: Dict) -> tuple:
    """工具层入口：先按原话校验纠正，再执行。返回 (型号列表, 纠正说明)。

    编排层（P1-5）拿 args 后只调这里；`find_models_by_args` 保持"照参数执行"的纯语义，
    供离线测试直接构造参数用。
    """
    fresh, notes = validate_find_models_args(query, args)
    return find_models_by_args(fresh), notes
