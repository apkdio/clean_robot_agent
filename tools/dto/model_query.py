"""模型查询结果：预算/日期/系列筛选后的型号集合。"""
from __future__ import annotations

from dataclasses import dataclass, field


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
class ParamFilter:
    """型号规格筛选条件。 如 吸力/导航_大于/含_6000/LDS """
    key: str
    op: str
    value: str


@dataclass(frozen=True)
class ModelQueryConstraints:
    """型号查询的完整输入：所有约束都先校验成这个对象。"""
    budget: PriceRange | None = None
    publish: PublishRange | None = None
    model_names: tuple[str, ...] = ()
    series: str = ""
    params: tuple[ParamFilter, ...] = ()
    scope: tuple[str, ...] = ()        # 追问逻辑使用，供选择追问范围
    tags: tuple[str, ...] = ()
    sort_by: str = ""
    order: str = "desc"
    limit: int = 5


@dataclass(frozen=True)
class FindModelsResult:
    """型号查询结果：模型列表 + 被丢弃的参数 + 诊断说明 + 实际生效约束。"""
    models: list
    dropped: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    constraints: ModelQueryConstraints | None = None
    kind: str = ""
