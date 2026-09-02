"""选购推荐 SOP：通过多轮收集预算、宠物情况，然后结构化推荐。"""

import re

from sops.base import register, is_consulting, is_aftersales, is_brand
from config.word_dict_config import PURCHASE_TRIGGER
from tools.metadata_extractor import (
    extract_price_constraint,
    enumerate_models,
    format_model_line,
)


def _extract_budget(text: str, slots: dict):
    """提取预算槽位，返回 (min_price, max_price) 元组（区间/上限/下限/浮动）。"""
    return extract_price_constraint(text)


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


PURCHASE_SOP = {
    "id": "purchase",
    "trigger": PURCHASE_TRIGGER,
    "guards": [
        {"check": is_consulting},
        {"check": is_aftersales},
        {"check": is_brand},
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
