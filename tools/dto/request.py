"""一轮对话的请求体：跨入口传递用户输入与定位信息。"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatRequest:
    query: str
    session_id: str
    lng: float | None = None
    lat: float | None = None
    retry: bool = False

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query 必须是非空字符串")
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id 必须是非空字符串")
        if not isinstance(self.retry, bool):
            raise ValueError("retry 必须是 bool")
        if (self.lng is None) != (self.lat is None):
            raise ValueError("lng 与 lat 必须同时提供")
        for name, value, low, high in (
            ("lng", self.lng, -180.0, 180.0),
            ("lat", self.lat, -90.0, 90.0),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("%s 必须是数字" % name)
            if not math.isfinite(float(value)) or not low <= float(value) <= high:
                raise ValueError("%s 超出范围" % name)
