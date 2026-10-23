"""安全告警窗口的判定与话术。

这里只做纯判定和文本拼装，不推进澄清、不改 SOP 状态：
  - 主链路用 `safety_carry_text()` 走硬承接（独立出口）；
  - 澄清/SOP 重问用 `with_safety_hint()` 只追加软提醒，不消费 pending。
"""
from __future__ import annotations

from tools.config_tool import load_config
from tools.log_tool import get_logger
from tools import trace_store
from config.word_dict_config import (
    SAFETY_CARRY,
    SAFETY_CARRY_PREFIX,
    SAFETY_HINT,
)

logger = get_logger(name="safety")

_cfg_cache = None


def _cfg() -> dict:
    global _cfg_cache
    if _cfg_cache is None:
        try:
            _cfg_cache = load_config("context") or {}
        except Exception as e:  # noqa: BLE001
            logger.warning("[Safety] load context config failed: %s", e)
            _cfg_cache = {}
    return _cfg_cache


def _enabled() -> bool:
    return bool((_cfg().get("context") or {}).get("enabled", True))


def _recent_messages(session_id: str) -> list:
    from tools.context_store import get_recent

    n = int((_cfg().get("context") or {}).get("topic_window", 8) or 8)
    return get_recent(session_id, n=n)


def safety_carry_max() -> int:
    return int((_cfg().get("context") or {}).get("safety_carry_max", 2) or 0)


def window_danger_word(session_id: str) -> str | None:
    """窗口内最近一次安全告警命中的危险词；没有告警返回 None。"""
    from config.word_dict_config import DANGER_WORDS

    for m in reversed(_recent_messages(session_id)):
        if m.get("danger"):
            text = m.get("content") or ""
            return next((w for w in DANGER_WORDS if w in text), "")
    return None


def recent_carry_count(session_id: str) -> int:
    """最近连续几条客服回复是硬承接。"""
    n = 0
    for m in reversed(_recent_messages(session_id)):
        if m.get("role") != "assistant":
            continue
        if (m.get("content") or "").startswith(SAFETY_CARRY_PREFIX):
            n += 1
        else:
            break
    return n


def safety_carry_text(session_id: str, q: str, *, intent: str = None,
                      margin: float = None) -> str | None:
    """主链路硬承接话术；不满足条件返回 None。"""
    if not _enabled():
        return None
    danger_word = window_danger_word(session_id)
    if danger_word is None:
        return None
    max_carry = safety_carry_max()
    if max_carry <= 0 or recent_carry_count(session_id) >= max_carry:
        return None
    if intent is None:
        from tools.intent_router import route_intent_with_margin
        intent, margin = route_intent_with_margin(q)
    high_conf_margin = float((_cfg().get("intent") or {}).get("high_conf_margin", 0) or 0)
    margin = float(margin or 0.0)
    if intent == "robot" and high_conf_margin > 0 and margin >= high_conf_margin:
        return None
    return SAFETY_CARRY.format(danger=danger_word or "安全隐患")


def _hint(session_id: str):
    """返回 (提醒文案, 危险词)；不满足条件返回 None。"""
    if not _enabled() or safety_carry_max() <= 0:
        return None
    danger_word = window_danger_word(session_id)
    if danger_word is None:
        return None
    return SAFETY_HINT.format(danger=danger_word or "安全隐患"), danger_word


def safety_hint_text(session_id: str) -> str | None:
    hit = _hint(session_id)
    return hit[0] if hit else None


def with_safety_hint(reply: str, session_id: str, *, by: str = "",
                     sop: str = "") -> str:
    """在重问话术后追加软提醒；不满足条件或已存在时原样返回。"""
    if not reply:
        return reply
    hit = _hint(session_id)
    if not hit:
        return reply
    hint, danger_word = hit
    if hint in reply:
        return reply
    trace_store.node("guard.safety_hint", status="hit",
                     in_={"by": by or "-", "sop": sop or "-",
                          "danger": danger_word or "-"},
                     out={"chars": len(hint)})
    return reply + "\n\n" + hint
