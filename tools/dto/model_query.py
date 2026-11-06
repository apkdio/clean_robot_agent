"""模型查询结果：预算/日期/系列筛选后的型号集合。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceRange:
    """预算闭区间（元）；None 表示该侧不限制。"""
    min: int | None = None
    max: int | None = None


@dataclass(frozen=True)
class PublishRange:
    """发布时间闭区间，统一 int YYYYMMDD；None 表示该侧不限制。"""
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class ModelQueryResult:
    models: list
    kind: str
    constraints: dict
