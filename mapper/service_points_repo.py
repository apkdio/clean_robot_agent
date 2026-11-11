"""网点数据与城市解析：文件清单 + geonamescache 缓存。"""
from __future__ import annotations

import json
from math import atan2, cos, radians, sin, sqrt
from typing import Dict, List, Optional, Tuple

_SERVICE_POINTS_FILE = "data/service_point/service_points.json"
_points_cache: Optional[List[Dict]] = None
_gc = None


def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """两个经纬度点之间的球面直线距离（公里）。"""
    r = 6371.0  # 地球平均半径（km）
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


def _load_points() -> List[Dict]:
    """全量网点清单（模块级缓存）。"""
    global _points_cache
    if _points_cache is None:
        from utils.path_tool import get_abs_path
        with open(get_abs_path(_SERVICE_POINTS_FILE), encoding="utf-8") as f:
            _points_cache = json.load(f)
    return _points_cache


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
    """城市名 → 候选经纬度列表（按国家 CN 过滤 + 人口降序；无匹配返回 []）。

    重名城市会返回多个候选，供上层消歧。
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
