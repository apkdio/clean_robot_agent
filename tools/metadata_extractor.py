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
        {"name": str, "price": int, "suction": str, "navigation": str,
         "obstacle": str} —— 缺失字段为空字符串/None。
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

    return {
        "name": name,
        "price": price,
        "suction": _field("吸力"),
        "navigation": _field("导航"),
        "obstacle": _field("避障"),
        "publish_date":_field("发布时间")
    }


def format_model_line(info: Dict) -> str:
    """把单个型号信息格式化成一行可读文本。"""
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
    line = f"- **{info['name']}**"
    if spec_str:
        line += f"：{spec_str}"
    if info.get("price") is not None:
        line += f"，参考价 {info['price']} 元"
    if info.get("publish_date") is not None:
        line += f"（发布日期:{info['publish_date']}）"
    return line


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


def extract_budget(query: str) -> Optional[int]:
    """从 query 提取价格上限，如 '1000以内' → 1000。

    支持：
      - 数字："1000以内" / "1000元以内" / "1000以下" / "1000块左右"（≈1000）
      - 中文："一千以内" / "两千以下"
    未检测到预算约束时返回 None。
    """
    # 阿拉伯数字
    m = re.search(r"(\d+)\s*(?:元|块|块钱)?\s*(?:以内|以下|之内|之内)", query)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*(?:元|块|块钱)?\s*左右", query)
    if m:
        return int(m.group(1))

    # 中文数字 + 以内/以下
    m = re.search(r"([零一二两三四五六七八九十百千万]+)\s*(?:以内|以下|之内)", query)
    if m:
        return _cn_to_int(m.group(1))
    m = re.search(r"([零一二两三四五六七八九十百千万]+)\s*(?:左右)", query)
    if m:
        return _cn_to_int(m.group(1))
    return None


def extract_price_range(query: str) -> Optional[tuple]:
    """提取显式价格区间 '1000到2000' / '1000-2000' → (1000, 2000)。"""
    m = _PRICE_RANGE_PATTERN.search(query)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def build_filter(query: str) -> Optional[Dict]:
    """从用户 query 构建 Chroma `where` 过滤条件。

    未检测到结构化约束（普通 RAG 查询）时返回 None。
    """
    # 显式区间优先于单一上限
    rng = extract_price_range(query)
    if rng:
        return {
            "$and": [
                {"min_price": {"$lte": rng[1]}},
                {"max_price": {"$gte": rng[0]}},
            ]
        }
    budget = extract_budget(query)
    if budget is not None:
        return {"min_price": {"$lte": budget}}
    return None
