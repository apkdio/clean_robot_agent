"""安全判定：危险词、告警窗口、硬承接与软提醒共用同一份决策。

这里只做纯判定和文本拼装，不推进澄清、不改 SOP 状态：
  - 主链路 danger 阶段用 `decide_safety(mode="stop")` 拿停机话术；
  - 主链路安全承接用 `decide_safety(mode="carry")`；
  - 澄清/SOP 重问用 `decide_safety(mode="hint")` 只追加提醒，不消费 pending。
"""
from __future__ import annotations

from dataclasses import dataclass

from config.word_dict_config import (
    AFTERSALES_HOTLINE,
    DANGER_WORDS,
    SAFETY_CARRY as SAFETY_CARRY_TEMPLATE,
    SAFETY_CARRY_PREFIX,
    SAFETY_HINT as SAFETY_HINT_TEMPLATE,
    SAFETY_STOP_REPLY,
)
from mapper import trace_store
from utils.config_tool import get_config
from utils.log_tool import get_logger

logger = get_logger(name="safety")

SAFETY_NONE = "none"
SAFETY_STOP = "stop"
SAFETY_CARRY = "carry"
SAFETY_HINT = "hint"
SAFETY_LEVELS = (SAFETY_NONE, SAFETY_STOP, SAFETY_CARRY, SAFETY_HINT)
SAFETY_MODES = (SAFETY_STOP, SAFETY_CARRY, SAFETY_HINT)


@dataclass(frozen=True)
class SafetyDecision:
    """一次安全判定的结果：级别、命中的危险词与要输出的话术。"""

    level: str
    word: str | None = None
    reply: str = ""

    def __post_init__(self):
        if self.level not in SAFETY_LEVELS:
            raise ValueError("SafetyDecision.level 不在闭集内：%r" % self.level)
        if self.word is not None and not isinstance(self.word, str):
            raise ValueError("SafetyDecision.word 必须是 str 或 None")
        if not isinstance(self.reply, str):
            raise ValueError("SafetyDecision.reply 必须是 str")
        if self.level != SAFETY_NONE and not self.reply:
            raise ValueError("命中安全级别必须带话术")


def _cfg() -> dict:
    try:
        return get_config("context") or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("[Safety] load context config failed: %s", e)
        return {}


def _enabled() -> bool:
    return bool((_cfg().get("context") or {}).get("enabled", True))


def _recent_messages(session_id: str) -> list:
    from mapper.context_store import get_recent

    n = int((_cfg().get("context") or {}).get("topic_window", 8) or 8)
    return get_recent(session_id, n=n)


def safety_carry_max() -> int:
    return int((_cfg().get("context") or {}).get("safety_carry_max", 2) or 0)


def detect_danger(query: str) -> str | None:
    """当前句命中的第一个危险词；没有返回 None。"""
    text = query or ""
    return next((w for w in DANGER_WORDS if w in text), None)


def window_danger_word(session_id: str) -> str | None:
    """窗口内最近一次安全告警命中的危险词；没有告警返回 None。"""
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


def decide_safety(session_id: str, query: str = "", *, intent: str = None,
                  margin: float = None, mode: str = SAFETY_CARRY) -> SafetyDecision:
    """统一安全判定。

    - stop：只看当前句是否含危险词；
    - carry：看告警窗口，满足次数/意图条件才硬承接；
    - hint：看告警窗口，只给软提醒，不受连续承接次数和意图限制。
    """
    if mode not in SAFETY_MODES:
        raise ValueError("mode 必须是 %s 之一：%r" % (SAFETY_MODES, mode))

    if mode == SAFETY_STOP:
        word = detect_danger(query)
        if not word:
            return SafetyDecision(SAFETY_NONE)
        return SafetyDecision(
            SAFETY_STOP, word, SAFETY_STOP_REPLY.format(hotline=AFTERSALES_HOTLINE))

    danger_word = window_danger_word(session_id)
    if danger_word is None:
        return SafetyDecision(SAFETY_NONE)
    if not _enabled() or safety_carry_max() <= 0:
        return SafetyDecision(SAFETY_NONE)
    if mode == SAFETY_HINT:
        return SafetyDecision(
            SAFETY_HINT, danger_word or "安全隐患",
            SAFETY_HINT_TEMPLATE.format(danger=danger_word or "安全隐患"))

    if recent_carry_count(session_id) >= safety_carry_max():
        return SafetyDecision(SAFETY_NONE)
    if intent is None:
        from service.intent_router import route_intent_with_margin
        intent, margin = route_intent_with_margin(query)
    high_conf_margin = float((_cfg().get("intent") or {}).get("high_conf_margin", 0) or 0)
    margin = float(margin or 0.0)
    if intent == "robot" and high_conf_margin > 0 and margin >= high_conf_margin:
        return SafetyDecision(SAFETY_NONE)
    return SafetyDecision(
        SAFETY_CARRY, danger_word or "安全隐患",
        SAFETY_CARRY_TEMPLATE.format(danger=danger_word or "安全隐患"))


def safety_carry_text(session_id: str, q: str, *, intent: str = None,
                      margin: float = None) -> str | None:
    """主链路硬承接话术；不满足条件返回 None。"""
    decision = decide_safety(session_id, q, intent=intent, margin=margin, mode=SAFETY_CARRY)
    return decision.reply or None


def safety_hint_text(session_id: str) -> str | None:
    decision = decide_safety(session_id, mode=SAFETY_HINT)
    return decision.reply or None


def with_safety_hint(reply: str, session_id: str, *, by: str = "",
                     sop: str = "") -> str:
    """在重问话术后追加软提醒；不满足条件或已存在时原样返回。"""
    if not reply:
        return reply
    decision = decide_safety(session_id, mode=SAFETY_HINT)
    if decision.level == SAFETY_NONE:
        return reply
    hint = decision.reply
    if hint in reply:
        return reply
    trace_store.node("guard.safety_hint", status="hit",
                     in_={"by": by or "-", "sop": sop or "-",
                          "danger": decision.word or "-"},
                     out={"chars": len(hint)})
    return reply + "\n\n" + hint
