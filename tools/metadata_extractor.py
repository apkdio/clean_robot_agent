"""Metadata extraction utilities.

Extracts structured dimensions (price, etc.) from chunk text and turns user
queries into Chroma `where` filters. This is the pluggable layer that makes
the RAG pipeline handle structured queries (budget, price range) precisely,
without changing the vector-search core.

The rules below are generic and can be extended per domain:
  - extract_*  : chunk text → metadata dict (used at ingest time)
  - query_*    : user query  → Chroma filter dict (used at search time)
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# ──────────────────────────────────────────────────────────────────────────
# Chunk → metadata (ingest time)
# ──────────────────────────────────────────────────────────────────────────

_PRICE_PATTERN = re.compile(r"参考价[:：]?\s*(\d+(?:\.\d+)?)")
_PRICE_RANGE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*[-–~到]\s*(\d+(?:\.\d+)?)")


def extract_price_metadata(text: str) -> Dict:
    """Extract min/max price from a chunk.

    A chunk may contain several product entries with different prices; we store
    min/max so a budget filter can keep chunks that contain ANY in-budget item.

    Returns:
        {"min_price": int, "max_price": int} or {} if no price found.
    """
    prices = [float(m) for m in _PRICE_PATTERN.findall(text)]
    if not prices:
        return {}
    return {"min_price": int(min(prices)), "max_price": int(max(prices))}


def extract_model_info(doc) -> Dict:
    """Extract structured model info from a single-model chunk.

    Reads the model name (in **bold**), its price (from metadata, which is
    precise per-entry after entry-level splitting), and key specs.

    Returns:
        {"name": str, "price": int, "suction": str, "navigation": str,
         "obstacle": str, "extra": str} — missing fields are empty/None.
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
    }


def format_model_line(info: Dict) -> str:
    """Format one model's info into a single readable line."""
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
    return line


# ──────────────────────────────────────────────────────────────────────────
# Query → Chroma filter (search time)
# ──────────────────────────────────────────────────────────────────────────

_CN_NUM = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _cn_to_int(s: str) -> Optional[int]:
    """Convert a simple Chinese numeral like '一千' or '两千五' to int."""
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
    """Extract an upper price limit from a query, e.g. '1000以内' → 1000.

    Handles:
      - 数字: "1000以内" / "1000元以内" / "1000以下" / "1000块左右" (≈1000)
      - 中文: "一千以内" / "两千以下"
    Returns None if no budget constraint is detected.
    """
    # Arabic numerals
    m = re.search(r"(\d+)\s*(?:元|块|块钱)?\s*(?:以内|以下|之内|之内)", query)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*(?:元|块|块钱)?\s*左右", query)
    if m:
        return int(m.group(1))

    # Chinese numerals followed by 以内/以下
    m = re.search(r"([零一二两三四五六七八九十百千万]+)\s*(?:以内|以下|之内)", query)
    if m:
        return _cn_to_int(m.group(1))
    m = re.search(r"([零一二两三四五六七八九十百千万]+)\s*(?:左右)", query)
    if m:
        return _cn_to_int(m.group(1))
    return None


def extract_price_range(query: str) -> Optional[tuple]:
    """Extract an explicit price range '1000到2000' / '1000-2000' → (1000, 2000)."""
    m = _PRICE_RANGE_PATTERN.search(query)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def build_filter(query: str) -> Optional[Dict]:
    """Build a Chroma `where` filter from a user query.

    Returns None when no structured constraint is detected (plain RAG query).
    """
    # Explicit range wins over single upper bound
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
