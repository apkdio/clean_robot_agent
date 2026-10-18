"""操作判别器：判断一句话想让系统干哪个操作，供主链路按表序派发。

每个操作一条注册：predicate 判入口（纯函数，只读 query）、allow 管意图合法性、order 定表序。
判据全部复用现成实现（match_sop、领域词表、时间正则等），本文件不重写判据。
"""
from __future__ import annotations

from tools.log_tool import get_logger

logger = get_logger(name="operations")

# 表序 = 现行主链路的先后（追问 → 型号 → 系列 → 网点 → SOP → 日期 → 结构化 → 域路由）
# 派发按这个顺序尝试；多命中且未过阈值门时，它也充当裁决顺序。
ORDER = ("followup", "model", "series", "service_point", "sop", "date", "structured", "domain")

_ROBOTISH = ("robot", "unknown")     # 型号/系列/SOP/日期/结构化这些分支的合法性门


def _p_followup(query: str) -> bool:
    """命中追问触发词（要不要取上一轮推荐，由执行体判断）。"""
    from config.word_dict_config import (LATEST_WORDS, RECENT_VAGUE_WORDS, CHEAPER_WORDS,
                                         PRICIER_WORDS, NEWER_WORDS, OLDER_WORDS)
    words = LATEST_WORDS + RECENT_VAGUE_WORDS + CHEAPER_WORDS + PRICIER_WORDS + NEWER_WORDS + OLDER_WORDS
    return any(w in query for w in words) or any(
        w in query for w in ("最贵", "最便宜", "价格最高", "价格最低"))


def _p_model(query: str) -> bool:
    """命中型号查询入口：触发词，或咨询词加指代特征。"""
    from config.word_dict_config import MODEL_QUERY_WORDS, MODEL_CONSULT_WORDS
    if any(w in query for w in MODEL_QUERY_WORDS):
        return True
    from tools.agent import _has_model_ref
    return any(w in query for w in MODEL_CONSULT_WORDS) and _has_model_ref(query)


def _p_series(query: str) -> bool:
    from config.word_dict_config import SERIES_QUERY_WORDS
    return any(w in query for w in SERIES_QUERY_WORDS)


def _p_service_point(query: str) -> bool:
    from sops import match_sop
    return bool(match_sop(query, "service_point"))


def _p_sop(query: str) -> bool:
    from sops import match_sop
    return bool(match_sop(query))


def _p_date(query: str) -> bool:
    """含时间表达（购买/使用时长不算，与 agent 的日期守卫同口径）。"""
    from config.word_dict_config import TIME_HINT_RE, PURCHASE_DURATION_RE, RELEASE_HINT_RE
    if not TIME_HINT_RE.search(query):
        return False
    return not (PURCHASE_DURATION_RE.search(query) and not RELEASE_HINT_RE.search(query))


def _p_structured(query: str) -> bool:
    from tools.metadata_extractor import build_filter
    return bool(build_filter(query))


def _p_domain(query: str) -> bool:
    from tools.agent import _route_domain
    return bool(_route_domain(query))


# name → (predicate, allow, order)
OPERATIONS = {
    "followup": (_p_followup, _ROBOTISH, ORDER.index("followup")),
    "model": (_p_model, _ROBOTISH, ORDER.index("model")),
    "series": (_p_series, _ROBOTISH, ORDER.index("series")),
    "service_point": (_p_service_point, None, ORDER.index("service_point")),   # 不依赖意图
    "sop": (_p_sop, _ROBOTISH, ORDER.index("sop")),
    "date": (_p_date, _ROBOTISH, ORDER.index("date")),
    "structured": (_p_structured, _ROBOTISH, ORDER.index("structured")),
    "domain": (_p_domain, _ROBOTISH, ORDER.index("domain")),
}


def probe(query: str) -> list:
    """命中哪些操作（按表序返回名字列表）；单个谓词出错只跳过它，不影响其余。"""
    out = []
    for name, (pred, _allow, _order) in sorted(OPERATIONS.items(), key=lambda kv: kv[1][2]):
        try:
            if pred(query):
                out.append(name)
        except Exception as e:  # noqa: BLE001
            logger.warning("[Ops] predicate %s failed: %s", name, e)
    return out


def _allowed(name: str, intent: str) -> bool:
    allow = OPERATIONS[name][1]
    return allow is None or intent is None or intent in allow


def resolve(ops: list, *, intent: str = None, margin: float = None,
            low_conf: float = 0.2) -> tuple:
    """三态出口：按表序可裁决 → win；多命中且意图低置信 → ambiguous；无可用操作 → none。"""
    usable = [o for o in ops if _allowed(o, intent)]
    if not usable:
        return "none", None
    winner = usable[0]                      # ops 已按表序
    if len(usable) > 1 and margin is not None and low_conf > 0 and margin < low_conf:
        return "ambiguous", winner
    return "win", winner


def note_ops(query: str, *, intent: str = None, margin: float = None) -> dict:
    """判定并落 trace 的 route.ops，返回 {ops, usable, state, winner, why}。

    usable 是过 allow 门的候选（按表序），主链路按它派发。
    """
    from tools import trace_store
    raw = probe(query)
    state, winner = resolve(raw, intent=intent, margin=margin)
    usable = [o for o in raw if _allowed(o, intent)]
    note = {"ops": "/".join(usable) or "-", "n": len(usable), "state": state,
            "winner": winner or "-",
            "why": "single" if len(usable) == 1 else ("threshold" if state == "ambiguous" else "order")}
    trace_store.route("ops", **note)
    return {"ops": raw, "usable": usable, **note}
