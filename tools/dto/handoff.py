"""阶段间前向跳转：SOP 只产出查询意图，由主链后续阶段接管。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Handoff:
    """跳到当前阶段之后的某个阶段；target 必须由 _run_stages 校验为前向。"""

    target: str
    query: str = ""
    domain: str = ""

    def __post_init__(self):
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("Handoff.target 必须是非空字符串")
        if not isinstance(self.query, str):
            raise ValueError("Handoff.query 必须是 str")
        if not isinstance(self.domain, str):
            raise ValueError("Handoff.domain 必须是 str")
