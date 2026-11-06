"""预算提取工具（规则未命中时的 LLM 兜底）。

规则能识别的常见表达（"1000以内""1000-2000"）不走这里，只兜底口语/模糊表达
（"一千来块""1500上下""两千出头"）。

注意：qwen2.5:3b 的 function calling 只能稳定处理「单一 integer 参数」，多参数
（min/max）会被误填。因此只提取单一上限 budget_max，区间下限仍由规则覆盖，
中文数字区间（"一千五到两千"）退化为只取上限，属可接受的边缘损失。
"""

from __future__ import annotations

from typing import Dict, Optional


BUDGET_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_budget",
        "description": (
            "提取用户话里的购机预算上限（单位：元）。"
            "单一上限（「一千以内」「3000元以下」）填该数值；"
            "模糊表达（「一千来块」「两千左右」「1500上下」「两千出头」「八百多」）填该数值；"
            "区间（「一千到两千」）填区间上限。"
            "完全没有预算时填 0。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "budget_max": {
                    "type": "integer",
                    "description": "预算上限（元），无预算填 0",
                },
            },
            "required": ["budget_max"],
        },
    },
}


def budget_args_to_range(args: Dict) -> Optional[tuple]:
    """把 LLM 返回的 {budget_max} 转成 (None, max)；无效预算返回 None。"""
    if not args:
        return None

    max_val = args.get("budget_max")
    if max_val is None:
        return None

    try:
        max_val = int(max_val)
    except (TypeError, ValueError):
        return None

    if max_val <= 0:
        return None
    return None, max_val


def budget_args_to_filter(args: Dict) -> Optional[Dict]:
    """兼容 Chroma 的旧适配：把 LLM 的预算上限转成 where 条件。"""
    rng = budget_args_to_range(args)
    if rng is None:
        return None
    return {"min_price": {"$lte": rng[1]}}
