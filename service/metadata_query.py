"""型号/参数的数据查询：走 mapper 缓存，不直接读写 Chroma。"""
from __future__ import annotations

import re
from typing import Dict, Optional

from config.word_dict_config import SERIES_ALIASES
from utils.metadata_rules import (
    _BUDGET_HINT_RE,
    _match_model_filter,
    _range_to_filter,
    extract_price_constraint,
)
from utils.log_tool import get_logger

logger = get_logger(name="metadata_query")

def get_all_model_records() -> list[Dict]:
    """全量型号记录（基础字段 + `param_*` + `tag_*`），由 models_repo 统一缓存。"""
    from mapper.models_repo import all_records

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


def extract_series(query: str) -> str:
    """从 query 规则提取系列名（子串匹配 repo 系列目录），未精准命中时尝试走别名。

    返回系列名（如「净白 S」）。匹配忽略空白，兼容「云顶X」这种不带空格的写法。
    """
    from mapper.models_repo import series_catalog

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


def resolve_budget_range(query: str, allow_llm: bool = True) -> Optional[tuple]:
    """预算表达 → (min, max)；规则优先，必要时 LLM 兜底。"""
    c = extract_price_constraint(query)
    if c is not None and (c[0] is not None or c[1] is not None):
        return c
    if not allow_llm or not _BUDGET_HINT_RE.search(query):
        return None
    from service.llm_extract import extract_budget_range

    return extract_budget_range(query)


def resolve_budget_filter(query: str, allow_llm: bool = True) -> Optional[Dict]:
    """预算表达 → Chroma 过滤条件（旧适配；新代码用 resolve_budget_range）。"""
    rng = resolve_budget_range(query, allow_llm=allow_llm)
    if rng is None:
        return None
    return _range_to_filter(rng[0], rng[1])
