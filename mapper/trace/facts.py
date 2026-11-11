"""trace 事实归一化：标量原样保留，复杂值统一序列化成 JSON 字符串。

主 trace 的结构化事实块（retrieval / sop / answer）不走这里；
path[].in/out 与 events.jsonl 的 facts 是叶子事实，统一过 `to_facts()`。
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass


def to_fact(value):
    """把一个事实值转成 trace 可落的标量。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        value = asdict(value)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(value)


def to_facts(facts: dict | None) -> dict:
    """批量归一化；非 dict 视为空事实。"""
    if not isinstance(facts, dict):
        return {}
    return {str(k): to_fact(v) for k, v in facts.items()}
