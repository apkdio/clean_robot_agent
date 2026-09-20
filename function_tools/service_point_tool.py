"""LLM function calling 的服务网点查询工具。

把用户位置（经纬度）换算成最近的售后网点，按直线距离排序。
城市名 → 经纬度由 geonamescache 离线解析（geocode_city），网点数据存
data/service_point/service_points.json（demo 随机经纬度），不进向量库——
网点查询是「精确匹配 + 距离排序」，经纬度对 embedding 不友好，语义检索
反而是浪费。

对外提供：
  - SERVICE_POINT_TOOL_SCHEMA : 供 LLM 提取城市/地点名（string 参数）
  - SERVICE_POINT_TOOL_MODEL  : 提取地点用的模型名（跟随 agent.yaml llm.model）
  - geocode_city              : 中文城市名 → 经纬度候选列表（geonamescache）
  - haversine                 : 两个经纬度点的球面直线距离（km）
  - search_service_points     : 按经纬度算距离，返回最近网点
  - format_service_points     : 格式化成回复文本
"""

from __future__ import annotations

import json
from math import atan2, cos, radians, sin, sqrt
from typing import Dict, List, Optional, Tuple

from tools.llm_tool import get_chat_model_name

# 提取城市/地点跟随主生成模型（string 参数 + 口语地点，3b 不稳定，同 model_tool）
SERVICE_POINT_TOOL_MODEL = get_chat_model_name()

_SERVICE_POINTS_FILE = "data/service_point/service_points.json"


SERVICE_POINT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_nearest_service_point",
        "description": (
            "从用户话里提取城市或区名（如「上海」「北京」「浦东」），用于查询附近售后网点。"
            "没有明确地点时返回空字符串。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市或区名（如「上海」「浦东」）；无地点返回空字符串",
                },
            },
            "required": ["location"],
        },
    },
}


def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """两个经纬度点之间的球面直线距离（公里，纯本地计算，无地图 API 依赖）。"""
    r = 6371.0  # 地球平均半径（km）
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


_points_cache: Optional[List[Dict]] = None


def _load_points() -> List[Dict]:
    """全量网点清单（模块级缓存）。"""
    global _points_cache
    if _points_cache is None:
        from tools.path_tool import get_abs_path
        with open(get_abs_path(_SERVICE_POINTS_FILE), encoding="utf-8") as f:
            _points_cache = json.load(f)
    return _points_cache


_gc = None  # geonamescache 实例（模块级缓存，内部数据只加载一次）


def _get_gc():
    global _gc
    if _gc is None:
        import geonamescache
        _gc = geonamescache.GeonamesCache(min_city_population=500)
    return _gc


def _pick_cn_name(altnames) -> Optional[str]:
    """从 geonames 别名里选一个中文名，优先含行政区划后缀（省/市/镇/区/县）。"""
    cn = [a for a in (altnames or []) if a and all('\u4e00' <= ch <= '\u9fff' for ch in a)]
    if not cn:
        return None
    for suffix in ("省", "市", "镇", "区", "县"):
        for n in cn:
            if n.endswith(suffix):
                return n
    return max(cn, key=len)


def geocode_city(name: str) -> List[Dict]:
    """城市名 → 候选经纬度列表（geonamescache 离线中文匹配）。

    按国家 CN 过滤 + 人口降序，返回 [{name, cn_name, lng, lat, population}]；
    无匹配返回 []。重名城市（如「洛阳」）会返回多个候选，供上层消歧。
    """
    key = (name or "").strip()
    if not key:
        return []
    try:
        results = _get_gc().search_cities(key, contains_search=False)
    except Exception:
        return []
    candidates = []
    for r in results:
        if r.get("countrycode") != "CN":
            continue
        candidates.append({
            "name": r.get("name"),
            "cn_name": _pick_cn_name(r.get("alternatenames")),
            "lng": r.get("longitude"),
            "lat": r.get("latitude"),
            "population": r.get("population", 0),
        })
    candidates.sort(key=lambda c: c.get("population") or 0, reverse=True)
    return candidates


def search_service_points(
    lng: Optional[float] = None,
    lat: Optional[float] = None,
    limit: int = 5,
) -> Tuple[List[Dict], str]:
    """按经纬度算距离，返回 (最近 limit 个网点, 来源描述)。"""
    points = [dict(p) for p in _load_points()]  # 拷贝，避免污染模块缓存
    if lng is None or lat is None:
        return points[:limit], ""
    for p in points:
        p["distance_km"] = round(haversine(lat, lng, p["lat"], p["lng"]), 1)
    points.sort(key=lambda p: p["distance_km"])
    return points[:limit], "您当前位置"


def format_service_points(points: List[Dict], origin_desc: str = "") -> str:
    """把网点列表格式化成回复文本。"""
    if not points:
        return "抱歉，暂时没有查询到附近的服务网点，您可拨打售后电话进行咨询～"
    lines = []
    for p in points:
        line = f"- **{p['name']}**"
        if p.get("distance_km") is not None:
            line += f"（距您约 {p['distance_km']} 公里）"
        line += f"\n  地址：{p.get('address', '')}"
        if p.get("phone"):
            line += f"｜电话：{p['phone']}"
        if p.get("hours"):
            line += f"｜营业时间：{p['hours']}"
        lines.append(line)
    header = (
        f"离{origin_desc}最近的网点有 {len(points)} 个："
        if origin_desc
        else f"为您找到 {len(points)} 个服务网点："
    )
    return header + "\n\n" + "\n".join(lines)
