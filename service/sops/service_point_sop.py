"""售后网点查询：问城市（重名时选序号）→ 按距离列最近网点。

城市识别复用网点工具（geonamescache）。前端给了坐标时主链路直接作答，不进本流程。
"""
import re

from config.word_dict_config import SERVICE_POINT_CONSULT_WORDS, SERVICE_POINT_WORDS
from service.sops.base import register
from utils.dto.sop import ExtractResult
from utils.log_tool import get_logger

logger = get_logger(name="sops_service_point")


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
    from service.llm_extract import extract_city_candidates

    # 抽城市失败=槽位收不到，下一步会重问；适配器内部已记 WARNING
    return extract_city_candidates(text)


def _extract_city(text: str, _slots: dict):
    """识城市；重名候选放 extra，由 base 在 done 时原子写入 slots。"""
    from mapper.service_points_repo import geocode_city
    candidates = geocode_city(text) or _extract_city_by_llm(text)
    if not candidates:
        return ExtractResult("fail", reason="no_city")
    extra = {"candidates": candidates}
    if len(candidates) > 1:
        extra["choice_prompt"] = _choice_prompt(candidates)
    city = candidates[0].get("cn_name") or candidates[0].get("name")
    return ExtractResult("done", city, extra=extra)


def _extract_pick(text: str, slots: dict):
    """答序号 → 候选下标（从 0 起）；越界/句子里没有完整序号都算失败。"""
    from utils.selection import parse_index

    candidates = slots.get("candidates") or []
    status, idx = parse_index(text, len(candidates))
    if status != "ok":
        return ExtractResult("fail", reason="no_pick")
    return ExtractResult("done", idx - 1)


def format_nearest(coord: dict) -> str:
    """按城市坐标算最近网点并格式化。"""
    from service.service_point import search_service_points, format_service_points
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
    "name": "售后网点查询",
    "trigger": SERVICE_POINT_WORDS,
    "enter_confirm": "always",
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
            "ask": "$choice_prompt",
            "retry": "请回复序号（如 1）选择您要查询的地点～",
            "extract": _extract_pick,
        },
        {"id": "do_search", "type": "action", "action": _search_points},
        {"id": "reply", "type": "reply", "template": "$points",
         "fallback": "抱歉，没能识别到城市～如果还有查询需求，可以回复「查询离我最近的网点」再次进入哦~"},
    ],
}

register(SERVICE_POINT_SOP)
