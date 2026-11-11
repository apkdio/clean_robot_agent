"""网点业务：距离排序与展示；数据读取在 mapper.service_points_repo。"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from mapper.service_points_repo import _load_points, haversine

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
