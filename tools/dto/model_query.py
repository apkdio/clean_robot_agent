"""模型查询结果：预算/日期/系列筛选后的型号集合。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelQueryResult:
    models: list
    kind: str
    constraints: dict
