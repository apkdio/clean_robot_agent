"""选购推荐 SOP：通过多轮收集预算、宠物情况，然后结构化推荐。"""

import re

from sops.base import register
from config.word_dict_config import PURCHASE_TRIGGER
from tools.metadata_extractor import (
    extract_budget,
    extract_price_range,
    enumerate_models,
    format_model_line,
)


def _extract_budget(text: str, slots: dict):
    """提取预算槽位，支持区间（1000-2000）和单一上限（1000以内）。

    返回 (min_price, max_price) 元组，单一上限时 min_price 为 None。
    """
    rng = extract_price_range(text)
    if rng:
        return rng
    budget = extract_budget(text)
    if budget is not None:
        return (None, budget)
    return None


def _extract_has_pet(text: str, slots: dict):
    """从用户回复提取「是否有宠物」槽位。"""
    if re.search(r"没|无|不养|没有|没养", text):
        return False
    if re.search(r"有|养|猫|狗|宠物", text):
        return True
    return None


def _search(slots: dict):
    """按预算（上限或区间）检索符合条件的产品；预算未知时检索全部。"""
    budget = slots.get("budget")
    if budget:
        min_price, max_price = budget
    else:
        min_price, max_price = None, None
    conditions = []
    if max_price is not None:
        conditions.append({"min_price": {"$lte": max_price}})
    if min_price is not None:
        conditions.append({"max_price": {"$gte": min_price}})
    if len(conditions) == 1:
        filter_dict = conditions[0]
    elif len(conditions) > 1:
        filter_dict = {"$and": conditions}
    else:
        # 无任何过滤条件（预算未知）→ 全匹配检索所有产品
        filter_dict = {"file_name": {"$ne": "__never__"}}
    models = enumerate_models(filter_dict)

    lines = [format_model_line(m) for m in models]
    return {"count": len(models), "list": "\n".join(lines), "models": models}


def _guard_is_consulting(query: str) -> bool:
    """选购咨询（"选购要注意什么"）→ 不触发选购 SOP，走 RAG。"""
    from sops.base import is_consulting
    return is_consulting(query)


def _guard_aftersales(query: str) -> bool:
    """售后咨询（"有没有保修/售后"）→ 不触发选购 SOP，走 RAG。"""
    from sops.base import is_aftersales
    return is_aftersales(query)


def _guard_brand(query: str) -> bool:
    """品牌咨询（"为什么买/优势"）→ 不触发选购 SOP，走 RAG 品牌介绍。"""
    from sops.base import is_brand
    return is_brand(query)


PURCHASE_SOP = {
    "id": "purchase",
    "trigger": PURCHASE_TRIGGER,
    "guards": [
        {"check": _guard_is_consulting},
        {"check": _guard_aftersales},
        {"check": _guard_brand},
    ],
    "intro": "好的，我来帮您推荐一款合适的扫地机器人～",
    "steps": [
        {
            "id": "ask_budget",
            "type": "ask",
            "slot": "budget",
            "ask": "好呀，先了解一下您的预算大概是多少呢？",
            "retry": "预算我没太听清，能说个具体数字吗？比如「1000以内」「2000元左右」。",
            "extract": _extract_budget,
        },
        {
            "id": "ask_pet",
            "type": "ask",
            "slot": "has_pet",
            "ask": "家里有养宠物吗？（有猫狗的话，我会优先推荐防毛发缠绕的机型）",
            "retry": "这个我没听懂哦～家里有养猫狗等宠物吗？回复「有」或「没有」就可以啦。",
            "extract": _extract_has_pet,
        },
        {
            "id": "do_search",
            "type": "action",
            "action": _search,
        },
        {
            "id": "reply",
            "type": "reply",
            "template": (
                "根据您的需求，共找到 {count} 款符合条件的机器人：\n\n"
                "{list}\n\n"
                "需要我帮您对比其中某两款吗？"
            ),
            "fallback": "抱歉，查询出了点小问题，您可以换个方式再问我一次～",
        },
    ],
}


register(PURCHASE_SOP)
