"""售后网点查询 SOP：问城市 → 重名消歧 → 算距离返回最近网点。

触发：网点词（"网点""门店""服务点"等）+ 位置词（"最近""附近""离我"等）。
若前端已授权定位（lng/lat 预填进 slots），跳过问城市直接算最近。
城市名通过 geonamescache 离线解析经纬度，重名城市（如"洛阳"）列出候选让用户选。
"""

import re

from sops.base import register
from config.word_dict_config import SERVICE_POINT_WORDS, LOCATION_WORDS
from tools.log_tool import get_logger

logger = get_logger(name="service_point_sop")


def _guard_policy_consult(query: str) -> bool:
    """「网点怎么查询」这类政策咨询（网点词但不含位置词）→ 不触发网点 SOP。"""
    return not any(w in query for w in LOCATION_WORDS)


def _has_geo(slots: dict) -> bool:
    """前端已授权定位（经纬度预填进 slots）。"""
    return slots.get("lng") is not None and slots.get("lat") is not None


def _extract_location(text: str, slots: dict):
    """从用户话提取城市名，并把经纬度候选存进 slots。

    返回城市名（提取成功）或 None（需反问）。候选唯一 → slots['coord']；
    候选多个 → slots['candidates']（重名，进入消歧）。
    """
    from function_tools.service_point_tool import (
        geocode_city, SERVICE_POINT_TOOL_SCHEMA, SERVICE_POINT_TOOL_MODEL,
    )
    text = (text or "").strip()
    if not text:
        return None

    # 1. 直接 geocode 整句（用户可能只说城市名"开封"）
    name = text
    candidates = geocode_city(name)

    # 2. 失败 → LLM 提取城市名再 geocode
    if not candidates:
        try:
            from tools.llm_tool import chat_with_tools
            from langchain_core.messages import HumanMessage
            resp = chat_with_tools(
                [HumanMessage(content=text)],
                [SERVICE_POINT_TOOL_SCHEMA],
                model=SERVICE_POINT_TOOL_MODEL,
            )
            tool_calls = getattr(resp, "tool_calls", None) or []
            if tool_calls:
                tc = tool_calls[0]
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                name = (args.get("location") or "").strip()
                candidates = geocode_city(name)
        except Exception as e:
            logger.warning("[ServicePoint] city tool calling failed: %s", e)

    if not candidates:
        return None

    if len(candidates) == 1:
        slots["coord"] = candidates[0]
    else:
        slots["candidates"] = candidates
    return name


def _skip_choice(slots: dict) -> bool:
    """候选唯一或已有定位/坐标 → 跳过重名消歧。"""
    if _has_geo(slots) or slots.get("coord"):
        return True
    return len(slots.get("candidates", [])) <= 1


def _choice_prompt(slots: dict) -> str:
    """动态生成消歧话术（列出重名候选，中文名 + 人口区分）。"""
    cands = slots.get("candidates", [])
    lines = []
    for i, c in enumerate(cands):
        label = c.get("cn_name") or c.get("name") or "?"
        pop = c.get("population") or 0
        if pop >= 10000:
            label += f"（人口约 {pop // 10000} 万）"
        elif pop > 0:
            label += f"（人口 {pop}）"
        lines.append(f"{i + 1}. {label}")
    return "查到多个同名地点：\n" + "\n".join(lines) + "\n请回复序号选择～"


def _extract_choice(text: str, slots: dict):
    """解析用户选的序号 → 存 slots['coord']。返回序号（占位）或 None。"""
    m = re.search(r"[1-9]", text or "")
    if not m:
        return None
    idx = int(m.group()) - 1
    cands = slots.get("candidates", [])
    if 0 <= idx < len(cands):
        slots["coord"] = cands[idx]
        return idx
    return None


def _search(slots: dict):
    """按坐标（前端定位或城市）算距离，返回最近网点。"""
    from function_tools.service_point_tool import search_service_points, format_service_points
    lng = slots.get("lng")
    lat = slots.get("lat")
    if lng is not None and lat is not None:
        points, origin = search_service_points(lng=lng, lat=lat)
        return {"answer": format_service_points(points, origin)}
    coord = slots.get("coord")
    if coord:
        points, origin = search_service_points(lng=coord["lng"], lat=coord["lat"])
        return {"answer": format_service_points(points, coord.get("cn_name") or coord.get("name", ""))}
    return {"answer": "抱歉，没能确定您的位置，您可以换个城市名再试～"}


SERVICE_POINT_SOP = {
    "id": "service_point",
    "trigger": SERVICE_POINT_WORDS,
    "guards": [
        {"check": _guard_policy_consult},
    ],
    "intro": "我来帮您查一下附近的售后网点～",
    "steps": [
        {
            "id": "ask_location",
            "type": "ask",
            "slot": "location",
            "skip": _has_geo,
            "ask": "请问您所在的城市是？",
            "retry": "没太听清，能说下您在哪个城市吗？",
            "extract": _extract_location,
        },
        {
            "id": "ask_choice",
            "type": "ask",
            "slot": "choice_idx",
            "skip": _skip_choice,
            "ask": _choice_prompt,
            "retry": "请回复序号（如 1）～",
            "extract": _extract_choice,
        },
        {
            "id": "do_search",
            "type": "action",
            "action": _search,
        },
        {
            "id": "reply",
            "type": "reply",
            "template": "{answer}",
        },
    ],
}


register(SERVICE_POINT_SOP)
