"""售后网点查询：问城市（重名时选序号）→ 按距离列最近网点。

城市识别复用网点工具（geonamescache）。前端给了坐标时主链路直接作答，不进本流程。
"""
import re

from config.word_dict_config import SERVICE_POINT_CONSULT_WORDS, SERVICE_POINT_WORDS
from sops.base import register


def _is_consult(query: str) -> bool:
    return any(w in query for w in SERVICE_POINT_CONSULT_WORDS)


def _choice_prompt(candidates: list) -> str:
    """列出重名候选，让用户回序号选。"""
    lines = []
    for i, c in enumerate(candidates):
        label = c.get("cn_name") or c.get("name") or "?"
        pop = c.get("population") or 0
        if pop >= 10000:
            label += f"（人口约 {pop // 10000} 万）"
        elif pop > 0:
            label += f"（人口 {pop}）"
        lines.append(f"{i + 1}. {label}")
    return "查到多个同名地点：\n" + "\n".join(lines) + "\n请回复序号选择～"


def _extract_city_by_llm(text: str) -> list:
    """整句没识出城市 → 让 LLM 按网点工具的 schema 抽一次（沿用原有兜底）。"""
    from langchain_core.messages import HumanMessage
    from tools.llm_tool import chat_with_tools
    from function_tools.service_point_tool import (
        geocode_city, SERVICE_POINT_TOOL_SCHEMA, SERVICE_POINT_TOOL_MODEL,
    )
    try:
        resp = chat_with_tools(
            [HumanMessage(content=text)],
            [SERVICE_POINT_TOOL_SCHEMA],
            model=SERVICE_POINT_TOOL_MODEL,
        )
        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            tc = tool_calls[0]
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            return geocode_city((args.get("location") or "").strip())
    except Exception:  # noqa: BLE001
        pass
    return []


def _extract_city(text: str, slots: dict):
    """识城市；重名时把候选与选择话术写进 slots，供下一步的 when 与话术占位用。"""
    from function_tools.service_point_tool import geocode_city
    candidates = geocode_city(text) or _extract_city_by_llm(text)
    if not candidates:
        return None
    slots["candidates"] = candidates
    if len(candidates) > 1:
        slots["choice_prompt"] = _choice_prompt(candidates)
    return candidates[0].get("cn_name") or candidates[0].get("name")


def _extract_pick(text: str, slots: dict):
    """答序号 → 候选下标（从 0 起）。"""
    m = re.search(r"[1-9]", text or "")
    if not m:
        return None
    idx = int(m.group()) - 1
    return idx if 0 <= idx < len(slots.get("candidates") or []) else None


def format_nearest(coord: dict) -> str:
    """按城市坐标算最近网点并格式化。"""
    from function_tools.service_point_tool import search_service_points, format_service_points
    points, _ = search_service_points(lng=coord["lng"], lat=coord["lat"])
    return format_service_points(points, coord.get("cn_name") or coord.get("name", ""))


def _search_points(slots: dict):
    """用识别到的城市（或用户选中的同名地点）查最近网点。"""
    candidates = slots.get("candidates") or []
    if not candidates:
        return {"points": "抱歉，没能识别到城市～如果还有查询需求，可以回复「查询离我最近的网点」再次进入哦~"}
    idx = slots.get("pick")
    coord = candidates[idx] if isinstance(idx, int) and 0 <= idx < len(candidates) else candidates[0]
    return {"points": format_nearest(coord)}


# 槽位形如 {"city": xxx, "candidates": [xxx], "choice_prompt": xxx, "pick": xxx}（后两个仅重名时有）；action 返回 {"points": xxx}
SERVICE_POINT_SOP = {
    "id": "service_point",
    "trigger": SERVICE_POINT_WORDS,
    "guards": [{"check": _is_consult}],
    "intro": "好的，帮您查一下最近的售后网点～",
    "max_retry": 5,  # 城市没听清就接着问，别放弃（放弃了就没坐标可查）
    "steps": [
        {
            "id": "ask_city",
            "type": "ask",
            "slot": "city",
            "ask": "请问您所在的城市是？回复城市名就行～",
            "retry": "没太听清您在哪个城市，能再说一下城市名吗？比如「开封」「上海」～",
            "extract": _extract_city,
        },
        {
            "id": "ask_pick",
            "type": "ask",
            "slot": "pick",
            "when": lambda ctx: len(ctx.get("candidates") or []) > 1,
            "ask": "{choice_prompt}",
            "retry": "请回复序号（如 1）选择您要查询的地点～",
            "extract": _extract_pick,
        },
        {"id": "do_search", "type": "action", "action": _search_points},
        {"id": "reply", "type": "reply", "template": "{points}"},
    ],
}

register(SERVICE_POINT_SOP)
