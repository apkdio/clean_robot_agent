"""结构化元数据提取：chunk 文本 → metadata（入库时），用户 query → Chroma filter（检索时）。

让 RAG 流水线在不改动向量检索核心的前提下，精确处理预算、价格区间、发布时间等结构化查询。
命名约定：extract_* 用于入库侧，query_* 用于检索侧。
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from config.word_dict_config import (
    ASPECT_KEYWORDS as _ASPECT_KEYWORDS,
    BUDGET_BARE_PATTERN as _BUDGET_BARE_PATTERN,
    BUDGET_HINT_RE,
    CN_NUM as _CN_NUM,
    CN_UNIT as _CN_UNIT,
    MODEL_ASPECTS,
    PRICE_PATTERN as _PRICE_PATTERN,
    PRICE_RANGE_PATTERN as _PRICE_RANGE_PATTERN,
    PUBLISH_DATE_PATTERN as _PUBLISH_DATE_PATTERN,
    SERIES_ALIASES,
    SPEC_SKIP_KEYS as _SPEC_SKIP_KEYS,
    TAG_CATALOG as _TAG_CATALOG,
)
from function_tools.date_tool import publish_int
from tools.llm_tool import get_small_model_name
from tools.log_tool import get_logger

logger = get_logger(name="metadata_extractor")

# 预算提示特征（决定是否走 LLM 预算兜底）：货币单位/预算词，或"数字 + 范围词"，
# 或"数字 + 语气词收尾"（「两千吧」这类口语，交给 LLM 兜底）
_BUDGET_HINT_RE = BUDGET_HINT_RE



# ──────────────────────────────────────────────────────────────────────────
# chunk → metadata（入库时）
# ──────────────────────────────────────────────────────────────────────────


def extract_price_metadata(text: str) -> Dict:
    """从 chunk 提取最低/最高价，无价格时返回 {}。

    一个 chunk 可能包含多条价格不同的产品，存 min/max 是为了让预算过滤
    能保留「包含任意一款预算内产品」的 chunk。
    """
    prices = [float(m) for m in _PRICE_PATTERN.findall(text)]
    if not prices:
        return {}
    return {"min_price": int(min(prices)), "max_price": int(max(prices))}


def extract_publish_date(text: str) -> Dict:
    """从 chunk 提取发布时间（归一化为 int YYYYMMDD，便于 $gte/$lte 数值比较）。

    识别「发布时间：」后的 2024-01-01 / 2024/1/1 / 2024年1月1日；无则返回 {}。
    """
    m = _PUBLISH_DATE_PATTERN.search(text)
    if not m:
        return {}
    y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
    # 存成 int YYYYMMDD，便于 Chroma 用 $gte/$lte 做数值比较。
    return {"publish_date": int(f"{y}{mo:02d}{d:02d}")}


# 规格行（格式模板强制）：- 吸力：5500Pa｜导航：LDS激光｜避障：结构光
# 只认含分隔符 ｜ 的行，或以 - 开头的行，避免命中正文里的冒号。
_SPEC_PAIR_RE = re.compile(r"([^：:｜|\-\s][^：:｜|]{0,7})[：:]\s*([^｜|\n]{1,40})")
# 数值 + 单位；单位允许中文（分钟 / 毫升），否则「续航：160分钟」抽不出数值。
_SPEC_NUM_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([\u4e00-\u9fffA-Za-z%]{0,4})$")


def _spec_number(value: str):
    """值 → 数值 + 单位中的数字；不是数值则 None。

    """
    m = _SPEC_NUM_RE.match(value.replace(" ", ""))
    if not m:
        return None
    unit = m.group(2)
    has_latin = any(ch.isascii() and ch.isalpha() for ch in unit)
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in unit)
    if has_latin and has_cjk:
        return None
    return float(m.group(1))


def extract_spec_metadata(text: str) -> Dict:
    """从规格行提取通用参数 metadata：param_<键>（原文）+ param_<键>_num（可解析为数值时）。
    """
    out: Dict = {}
    for line in text.split("\n"):
        s = line.strip()
        if not s or ("：" not in s and ":" not in s):
            continue
        if "｜" not in s and "|" not in s and not s.startswith("-"):
            continue
        for key, value in _SPEC_PAIR_RE.findall(s):
            key, value = key.strip(), value.strip()
            if not key or not value or key in _SPEC_SKIP_KEYS:
                continue
            out.setdefault(f"param_{key}", value)
            num = _spec_number(value)
            if num is not None:
                out.setdefault(f"param_{key}_num", num)
    return out


_TAG_LABEL_TO_ID = {meta["label"]: tag_id for tag_id, meta in _TAG_CATALOG.items()}
_TAG_LINE_RE = re.compile(r"^\s*-\s*标签\s*[:：]\s*(.*)$")


def extract_tag_metadata(text: str) -> Dict:
    """从型号条目的「- 标签：…」行提取受控标签，输出全部 `tag_<id>: True/False`。

    标签只认 TAG_CATALOG 的 id 或中文 label；未知标签告警并忽略。
    「无 / 跳过」表示明确不打标签；整条没有标签行时同样全部为 False。
    """
    out = {f"tag_{tag_id}": False for tag_id in _TAG_CATALOG}
    for line in text.splitlines():
        m = _TAG_LINE_RE.match(line)
        if not m:
            continue
        raw = m.group(1).strip()
        if raw in ("", "无", "无标签", "跳过"):
            continue
        for item in re.split(r"[，,、|/]+", raw):
            item = item.strip()
            if not item:
                continue
            tag_id = item if item in _TAG_CATALOG else _TAG_LABEL_TO_ID.get(item)
            if tag_id is None:
                logger.warning("[Tag] 未登记标签 %r，已忽略", item)
                continue
            out[f"tag_{tag_id}"] = True
    return out


def param_inventory() -> Dict:
    """从已入库的 metadata 派生参数清单（键 → 值枚举 / 数值区间 / 覆盖条数）。
    """
    from tools.vector_store import get_vector_store  # 函数内导入：vector_store 依赖本模块，避免循环导入

    metas = get_vector_store()._collection.get(include=["metadatas"])["metadatas"]
    inv: Dict = {}
    for meta in metas:
        for key, value in (meta or {}).items():
            if not key.startswith("param_"):
                continue
            item = inv.setdefault(key, {"values": {}, "coverage": 0})
            text = str(value)
            item["values"][text] = item["values"].get(text, 0) + 1
            item["coverage"] += 1
            if key.endswith("_num"):
                item["num_min"] = min(item.get("num_min", value), value)
                item["num_max"] = max(item.get("num_max", value), value)
    return inv


def extract_model_info(doc) -> Dict:
    """从单型号 chunk 提取结构化型号信息（name/series/price/吸力/导航/避障/发布时间）。
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

    # 特点是整行短语（含空格与加号），同样不能用 _field
    def _feature() -> str:
        mm = re.search(r"特点[:：]\s*([^\n]+)", text)
        return mm.group(1).strip() if mm else ""

    return {
        "name": name,
        "series": _series(),
        "price": price,
        "suction": _field("吸力"),
        "navigation": _field("导航"),
        "obstacle": _field("避障"),
        "publish_date": _field("发布时间"),
        "feature": _feature(),
    }


# 型号属性维度映射（MODEL_ASPECTS）与属性关键词（_ASPECT_KEYWORDS）见 config/word_dict_config.py


def format_model_line(info: Dict, aspect: str = None) -> str:
    """把单个型号信息格式化成一行可读文本。
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


def format_models(models, header: str, aspect: str = None) -> str:
    """型号列表 → 回复文本（标题 + 每型号一行）；aspect 非空时只输出该属性维度。

    型号展示是这一层的事：原先放在 sops/base，逼得 agent（上层）反向 import sops（下层）。
    """
    lines = [format_model_line(m, aspect) for m in (models or [])]
    return header + "\n\n" + "\n".join(lines)


# aspect 关键词 → aspect 中文取值（规则提取，长关键词优先）见 config/word_dict_config.py


def extract_model_aspect(query: str) -> str:
    """从 query 规则提取型号属性维度（确定性，不依赖 LLM）。
    """
    for kw, aspect in _ASPECT_KEYWORDS:
        if kw in query:
            return aspect
    return ""


def _match_model_filter(info: Dict, f: dict) -> bool:
    """判断型号 info 是否满足 Chroma 风格的 where 过滤（内存过滤）。
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
            val = publish_int(info.get("publish_date")) or None
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


def get_all_model_records() -> list[Dict]:
    """全量型号记录（基础字段 + `param_*` + `tag_*`），由 models_repo 统一缓存。"""
    from tools.models_repo import all_records

    return all_records()


def get_all_models() -> list[Dict]:
    """全量型号基础视图（去掉 `param_*` / `tag_*`），供展示和型号名判断。"""
    return [{k: v for k, v in record.items()
             if not k.startswith(("param_", "tag_"))}
            for record in get_all_model_records()]


def enumerate_models(filter: dict, require_field: str = "price") -> list[Dict]:
    """按 filter 枚举匹配型号（全部走 models_repo 缓存，不读 Chroma）。"""
    models = get_all_model_records()
    if require_field:
        if require_field == "price":
            models = [m for m in models if m.get("price") is not None]
        else:
            models = [m for m in models if m.get(require_field)]
    if filter:
        models = [m for m in models if _match_model_filter(m, filter)]
    return models


# 系列目录从 models_repo 派生；别名（SERIES_ALIASES）在 config/word_dict_config.py


def extract_series(query: str) -> str:
    """从 query 规则提取系列名（子串匹配 repo 系列目录），未精准命中时尝试走别名。

    返回系列名（如「净白 S」）。匹配忽略空白，兼容「云顶X」这种不带空格的写法。
    """
    from tools.models_repo import series_catalog

    flat = re.sub(r"\s+", "", query)
    for series in series_catalog():
        if re.sub(r"\s+", "", series) in flat:
            return series
    for alias, series in SERIES_ALIASES.items():
        if alias in flat:
            return series
    return ""


def enumerate_models_by_series(series: str) -> list[Dict]:
    """枚举指定系列的全部型号（按 series 字段过滤）。"""
    models = enumerate_models({"file_name": {"$ne": "__never__"}}, require_field="price")
    return [m for m in models if m.get("series") == series]


# ──────────────────────────────────────────────────────────────────────────
# query → Chroma filter（检索时）
# ──────────────────────────────────────────────────────────────────────────

# 中文数字与单位（_CN_NUM / _CN_UNIT）见 config/word_dict_config.py


def _cn_to_int(s: str) -> Optional[int]:
    """把简单中文数字（'一千' / '两千五'）转成 int。

    尾部裸数字的两种读法：「两千五」＝2500（省略了百）、「一千零一」＝1001（零提示跳位）——
    用「单位后是否出现过零」区分：有零按个位补零读，无零按省略单位读（乘上一单位 ÷ 10）。
    """
    total = section = number = 0
    last_unit = 0
    trailing_zero = False
    for ch in s:
        if ch in _CN_NUM:
            if _CN_NUM[ch] == 0:
                trailing_zero = True
                continue
            number = _CN_NUM[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if number == 0:
                number = 1
            section += number * unit
            last_unit = unit
            number = 0
            trailing_zero = False
        else:
            return None
    if number and last_unit >= 10 and not trailing_zero:
        number = number * (last_unit // 10)
    return total + section + number


def extract_price_constraint(query: str) -> Optional[tuple]:
    """从 query 提取价格约束，返回 (min_price, max_price) 元组，未命中返回 None。
    """
    # 1. 显式区间（两分支：前缀价格词 → 组1/2；尾带货币单位 → 组3/4）
    m = _PRICE_RANGE_PATTERN.search(query)
    if m:
        lo, hi = (m.group(1), m.group(2)) if m.group(1) is not None else (m.group(3), m.group(4))
        return (int(float(lo)), int(float(hi)))

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

    # 5. 口语尾音：「2000吧」「3000呢」→ 同“左右”按 ±500 浮动。
    #    要求语气词紧跟在数字（可带单位）之后、且数字至少三位——避免"保修 2 年吧"这类误判
    m = re.search(r"(\d{3,})\s*(?:元|块|块钱)?\s*[吧呢啊呀](?![0-9])", query)
    if m:
        v = int(m.group(1))
        return (max(0, v - 500), v + 500)

    # 6. 「预算 3000」这类没有方向词的 → 视为上限（说"预算 3000"＝愿意花到 3000）
    m = _BUDGET_BARE_PATTERN.search(query)
    if m:
        return (None, int(m.group(1)))

    return None


def _range_to_filter(low: Optional[int], high: Optional[int]) -> Optional[Dict]:
    """预算上下限 → 兼容 Chroma 的 where；空约束返回 None。"""
    conditions = []
    if high is not None:
        conditions.append({"min_price": {"$lte": high}})
    if low is not None:
        conditions.append({"max_price": {"$gte": low}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def resolve_budget_range(query: str, allow_llm: bool = True) -> Optional[tuple]:
    """预算表达 → (min, max)；规则优先，必要时 LLM 兜底。"""
    c = extract_price_constraint(query)
    if c is not None and (c[0] is not None or c[1] is not None):
        return c
    if not allow_llm or not _BUDGET_HINT_RE.search(query):
        return None

    try:
        from function_tools.budget_tool import BUDGET_TOOL_SCHEMA, budget_args_to_range
        from tools.llm_tool import chat_with_tools
        from langchain_core.messages import HumanMessage
        resp = chat_with_tools(
            [HumanMessage(content=query)],
            [BUDGET_TOOL_SCHEMA],
            model=get_small_model_name(),
        )
        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            tc = tool_calls[0]
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            rng = budget_args_to_range(args)
            if rng is not None:
                logger.info("[Budget] resolved via LLM tool: %s", args)
                return rng
    except Exception as e:
        logger.warning("[Budget] tool calling failed: %s", e)
    return None


def resolve_budget_filter(query: str, allow_llm: bool = True) -> Optional[Dict]:
    """预算表达 → Chroma 过滤条件（旧适配；新代码用 resolve_budget_range）。"""
    rng = resolve_budget_range(query, allow_llm=allow_llm)
    if rng is None:
        return None
    return _range_to_filter(rng[0], rng[1])
