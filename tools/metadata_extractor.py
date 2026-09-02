"""结构化元数据提取工具。

从 chunk 文本提取结构化维度（价格、发布时间等），并把用户 query 转成
Chroma 的 `where` 过滤条件。这是让 RAG 流水线在不改动向量检索核心的前提下，
精确处理结构化查询（预算、价格区间、发布时间）的可插拔层。

下面的规则是通用设计，可按领域扩展：
  - extract_*  : chunk 文本 → metadata 字典（入库时用）
  - query_*    : 用户 query → Chroma filter 字典（检索时用）
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from log_tool import get_logger

logger = get_logger(name="metadata_extractor")

# 预算提示特征（决定是否走 LLM 预算兜底）：货币单位/预算词，或"数字 + 范围词"
_BUDGET_HINT_RE = re.compile(
    r"(?:元|块钱?|预算|价位|多少钱)"
    r"|(?:\d+|[一二两三四五六七八九十百千万]+)\s*(?:以内|以下|以上|起|左右|上下|出头|到|至|多)"
)

# 预算提取兜底用的小模型（3b 更快；精度不够可切回 "qwen2.5:7b"）
_BUDGET_TOOL_MODEL = "qwen2.5:3b"

# ──────────────────────────────────────────────────────────────────────────
# chunk → metadata（入库时）
# ──────────────────────────────────────────────────────────────────────────

_PRICE_PATTERN = re.compile(r"参考价[:：]?\s*(\d+(?:\.\d+)?)")
_PRICE_RANGE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*[-–~到]\s*(\d+(?:\.\d+)?)")
_PUBLISH_DATE_PATTERN = re.compile(
    r"发布时间[:：]?\s*(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})"
)


def extract_price_metadata(text: str) -> Dict:
    """从 chunk 提取最低/最高价。

    一个 chunk 可能包含多条价格不同的产品；存 min/max 是为了让预算过滤
    能保留「包含任意一款预算内产品」的 chunk。

    返回：
        {"min_price": int, "max_price": int}，无价格时返回 {}。
    """
    prices = [float(m) for m in _PRICE_PATTERN.findall(text)]
    if not prices:
        return {}
    return {"min_price": int(min(prices)), "max_price": int(max(prices))}


def extract_publish_date(text: str) -> Dict:
    """从 chunk 提取发布时间（归一化为 ISO 格式）。

    识别：发布时间：2024-01-01 / 2024/1/1 / 2024年1月1日
    返回 {"publish_date": "YYYY-MM-DD"}，无发布时间时返回 {}。
    """
    m = _PUBLISH_DATE_PATTERN.search(text)
    if not m:
        return {}
    y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
    # 存成 int YYYYMMDD，便于 Chroma 用 $gte/$lte 做数值比较。
    return {"publish_date": int(f"{y}{mo:02d}{d:02d}")}


def extract_model_info(doc) -> Dict:
    """从单型号 chunk 提取结构化型号信息。

    读取型号名（**加粗**）、价格（来自 metadata，条目级切分后对每个条目都精确）
    以及关键参数。

    返回：
        {"name": str, "series": str, "price": int, "suction": str,
         "navigation": str, "obstacle": str, "publish_date": str}
        —— 缺失字段为空字符串/None。
    """
    text = doc.page_content

    name = ""
    m = re.search(r"\*\*(.+?)\*\*", text)
    if m:
        name = m.group(1).strip()

    price = doc.metadata.get("min_price")
    if price is None:
        price = doc.metadata.get("price")

    def _field(label: str) -> str:
        mm = re.search(rf"{label}[:：]\s*([^\s｜|，,]+)", text)
        return mm.group(1).strip() if mm else ""

    # 系列名含空格（如「净白 S」），不能用 _field（遇空格就截断成「净白」），单独处理
    def _series() -> str:
        mm = re.search(r"系列[:：]\s*([^\n｜|，,]+)", text)
        return mm.group(1).strip() if mm else ""

    return {
        "name": name,
        "series": _series(),
        "price": price,
        "suction": _field("吸力"),
        "navigation": _field("导航"),
        "obstacle": _field("避障"),
        "publish_date": _field("发布时间")
    }


# 型号属性维度映射：aspect 中文取值 → (字段名, 显示模板)
# 用于「型号属性精准查询」（如"云顶 X2 多少钱"只答价格，不啰嗦全字段）
MODEL_ASPECTS = {
    "价格": ("price", "参考价 {v} 元"),
    "吸力": ("suction", "吸力 {v}"),
    "导航": ("navigation", "{v}"),
    "避障": ("obstacle", "{v}"),
    "发布时间": ("publish_date", "{v}"),
}


def format_model_line(info: Dict, aspect: str = None) -> str:
    """把单个型号信息格式化成一行可读文本。

    aspect 指定时只输出对应属性维度（见 MODEL_ASPECTS，如「价格」→ 参考价）；
    否则输出全字段（吸力/导航/避障 + 价格 + 发布时间）。
    """
    name = info.get("name", "")
    if aspect:
        entry = MODEL_ASPECTS.get(aspect)
        if entry:
            field, fmt = entry
            v = info.get(field)
            v_str = fmt.format(v=v if v not in (None, "") else "未知")
            return f"- **{name}**：{v_str}"

    specs = []
    if info.get("suction"):
        specs.append(f"吸力 {info['suction']}")
    if info.get("navigation"):
        nav = info["navigation"]
        if not nav.endswith("导航"):
            nav += "导航"
        specs.append(nav)
    if info.get("obstacle"):
        specs.append(f"{info['obstacle']}避障")
    spec_str = "、".join(specs)
    line = f"- **{name}**"
    if spec_str:
        line += f"：{spec_str}"
    if info.get("price") is not None:
        line += f"，参考价 {info['price']} 元"
    if info.get("publish_date") is not None:
        line += f"（发布日期:{info['publish_date']}）"
    return line


# aspect 关键词 → aspect 中文取值（规则提取，长关键词优先）
_ASPECT_KEYWORDS = [
    ("发布时间", "发布时间"), ("什么时候发布", "发布时间"), ("什么时候上市", "发布时间"),
    ("上市时间", "发布时间"), ("何时发布", "发布时间"), ("何时上市", "发布时间"),
    ("多少钱", "价格"), ("什么价", "价格"), ("价位", "价格"), ("售价", "价格"),
    ("价格", "价格"), ("便宜", "价格"),
    ("吸力", "吸力"), ("导航", "导航"), ("避障", "避障"),
]


def extract_model_aspect(query: str) -> str:
    """从 query 规则提取型号属性维度（确定性，不依赖 LLM）。

    返回 MODEL_ASPECTS 的 key（如「价格」「吸力」），未命中返回空字符串。
    """
    for kw, aspect in _ASPECT_KEYWORDS:
        if kw in query:
            return aspect
    return ""


# 全量型号元数据 pickle 缓存（避免每次型号枚举读 Chroma 全量 + 正则提取）
_MODELS_CACHE_FILE = "data/pkl/models.pkl"


def _match_model_filter(info: Dict, f: dict) -> bool:
    """判断型号 info 是否满足 Chroma 风格的 where 过滤（内存过滤）。

    字段映射：min_price/max_price → price（条目级切分保证 min_price==price），
    publish_date 由字符串转 int 比较；file_name 忽略（型号都在具体型号文件）。
    """
    if not f:
        return True
    if "$and" in f:
        return all(_match_model_filter(info, sub) for sub in f["$and"])
    for key, cond in f.items():
        if key.startswith("$") or key == "file_name":
            continue
        if key in ("min_price", "max_price", "price"):
            val = info.get("price")
        elif key == "publish_date":
            pd = info.get("publish_date") or ""
            val = int(pd.replace("-", "")) if pd else None
        else:
            val = info.get(key)
        if val is None:
            return False
        if isinstance(cond, dict):
            if "$lte" in cond and not (val <= cond["$lte"]):
                return False
            if "$gte" in cond and not (val >= cond["$gte"]):
                return False
            if "$eq" in cond and val != cond["$eq"]:
                return False
        else:
            if val != cond:
                return False
    return True


def _enumerate_models_from_store(filter: dict, require_field: str) -> list[Dict]:
    """底层：从 Chroma 读全量 + 提取 + 去重（不带缓存，重建用）。"""
    from vector_store import search_by_filter
    docs = search_by_filter(filter)
    models, seen = [], set()
    for c in docs:
        info = extract_model_info(c)
        if info.get(require_field) and info.get("name") and info["name"] not in seen:
            seen.add(info["name"])
            models.append(info)
    return models


def get_all_models() -> list[Dict]:
    """全量型号 info 列表（pickle 缓存，chunk_count 做 fingerprint）。

    优先读 data/pkl/models.pkl；失效或不存在则从 Chroma 重建并写缓存。
    型号数据在知识库变更前完全稳定，缓存可跨进程复用，避免每次枚举读 Chroma。
    """
    import os
    import pickle
    from path_tool import get_abs_path
    from vector_store import get_vector_store

    path = get_abs_path(_MODELS_CACHE_FILE)
    fingerprint = get_vector_store()._collection.count()

    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("fingerprint") == fingerprint:
                return payload["models"]
        except Exception:
            pass

    models = _enumerate_models_from_store({"file_name": {"$ne": "__never__"}}, require_field="price")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "models": models}, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return models


def enumerate_models(filter: dict, require_field: str = "price") -> list[Dict]:
    """按 filter 枚举匹配型号（基于 pickle 缓存 + 内存过滤，不读 Chroma）。

    get_all_models 已按 price 过滤，故 require_field="price" 天然满足；
    publish_date 等其它字段再内存过滤一次。
    """
    models = get_all_models()
    if require_field and require_field != "price":
        models = [m for m in models if m.get(require_field)]
    if filter:
        models = [m for m in models if _match_model_filter(m, filter)]
    return models


# 系列列表（与知识库「系列」字段一致，规则子串匹配用）
SERIES_LIST = ["净白 S", "净界 P", "天工 T", "云顶 X"]
SERIES_MAP_DICT = {
    "净白": "净白 S",
    "净界": "净界 P",
    "天工": "天工 T",
    "云顶": "云顶 X"
}


def extract_series(query: str) -> str:
    """从 query 规则提取系列名（子串匹配 SERIES_LIST），未精准命中时尝试走映射。

    返回系列名（如「净白 S」），未命中返回空字符串。
    """
    for s in SERIES_LIST:
        if s in query:
            return s
    for s in SERIES_MAP_DICT:
        if s in query:
            return SERIES_MAP_DICT[s]
    return ""


def enumerate_models_by_series(series: str) -> list[Dict]:
    """枚举指定系列的全部型号（按 series 字段过滤）。"""
    models = enumerate_models({"file_name": {"$ne": "__never__"}}, require_field="price")
    return [m for m in models if m.get("series") == series]


# ──────────────────────────────────────────────────────────────────────────
# query → Chroma filter（检索时）
# ──────────────────────────────────────────────────────────────────────────

_CN_NUM = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _cn_to_int(s: str) -> Optional[int]:
    """把简单中文数字（'一千' / '两千五'）转成 int。"""
    total = 0
    section = 0
    number = 0
    for ch in s:
        if ch in _CN_NUM:
            number = _CN_NUM[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        else:
            return None
    return total + section + number


def extract_price_constraint(query: str) -> Optional[tuple]:
    """从 query 提取价格约束，返回 (min_price, max_price) 元组。

    支持：
      - 区间："1000-2000" / "1000到2000" → (1000, 2000)
      - 上限："1000以内" / "1000以下" → (None, 1000)
      - 下限："1000以上" / "1000起" → (1000, None)
      - 浮动："1000左右" → (500, 1500)（预算 ±500）
    未检测到预算约束时返回 None。
    """
    # 1. 显式区间
    m = _PRICE_RANGE_PATTERN.search(query)
    if m:
        return (int(float(m.group(1))), int(float(m.group(2))))

    # 2. 上限（以内/以下）
    m = re.search(r"(\d+)\s*(?:元|块|块钱)?\s*(?:以内|以下|之内)", query)
    if m:
        return (None, int(m.group(1)))
    m = re.search(r"([零一二两三四五六七八九十百千万]+)\s*(?:以内|以下|之内)", query)
    if m:
        v = _cn_to_int(m.group(1))
        if v is not None:
            return (None, v)

    # 3. 下限（以上/起）
    m = re.search(r"(\d+)\s*(?:元|块|块钱)?\s*(?:以上|起)", query)
    if m:
        return (int(m.group(1)), None)
    m = re.search(r"([零一二两三四五六七八九十百千万]+)\s*(?:以上|起)", query)
    if m:
        v = _cn_to_int(m.group(1))
        if v is not None:
            return (v, None)

    # 4. 浮动（左右 ±500）
    m = re.search(r"(\d+)\s*(?:元|块|块钱)?\s*左右", query)
    if m:
        v = int(m.group(1))
        return (max(0, v - 500), v + 500)
    m = re.search(r"([零一二两三四五六七八九十百千万]+)\s*左右", query)
    if m:
        v = _cn_to_int(m.group(1))
        if v is not None:
            return (max(0, v - 500), v + 500)

    return None


def build_filter(query: str) -> Optional[Dict]:
    """从用户 query 构建 Chroma `where` 过滤条件。

    未检测到结构化约束（普通 RAG 查询）时返回 None。
    """
    c = extract_price_constraint(query)
    if c is None:
        return None
    min_price, max_price = c
    conditions = []
    if max_price is not None:
        conditions.append({"min_price": {"$lte": max_price}})
    if min_price is not None:
        conditions.append({"max_price": {"$gte": min_price}})
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def resolve_budget_filter(query: str) -> Optional[Dict]:
    """把问题里的预算表达解析成 Chroma 过滤条件（规则优先 + LLM 兜底）。

    规则（build_filter）覆盖"1000以内""1000-2000"等常见表达；
    规则 miss 且含预算提示特征时，用 3b function calling 兜底口语/模糊表达
    （"一千来块""1500上下"）；无提示特征则直接返回 None，不白调 LLM。
    """
    # 1. 确定性规则（显式区间 / 单一上限）
    f = build_filter(query)
    if f is not None:
        return f

    # 2. 无预算提示特征 → 直接返回，不调 LLM
    if not _BUDGET_HINT_RE.search(query):
        return None

    # 3. LLM function calling 兜底
    try:
        from function_tools.budget_tool import BUDGET_TOOL_SCHEMA, budget_args_to_filter
        from llm_tool import chat_with_tools
        from langchain_core.messages import HumanMessage
        resp = chat_with_tools(
            [HumanMessage(content=query)],
            [BUDGET_TOOL_SCHEMA],
            model=_BUDGET_TOOL_MODEL,
        )
        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            tc = tool_calls[0]
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            f = budget_args_to_filter(args)
            if f is not None:
                logger.info("[Budget] resolved via LLM tool: %s", args)
                return f
    except Exception as e:
        logger.warning("[Budget] tool calling failed: %s", e)
    return None
