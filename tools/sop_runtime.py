"""SOP 运行时：进出 SOP、重答重开与 SOP 进度记录。"""
from __future__ import annotations

import re

from config.word_dict_config import EXIT_WORDS, SOP_NAMES
from tools import pending_store, trace_store
from tools.log_tool import get_logger
from tools.turn_recorder import log_behavior

logger = get_logger(name="sop_runtime")

_EXIT_RESIDUE_RE = re.compile(r"[\s，,。.、！!？?~～…·「」【】（）()\"'“”‘’:：;；-]+")


def sop_name(sop_id: str) -> str:
    return SOP_NAMES.get(sop_id, "当前")


def is_symbols_only(query: str) -> bool:
    """判断输入是否纯符号（无中文/字母/数字）。"""
    return not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", query)


def has_residual_request(query: str) -> bool:
    """退出/纠正句里是否还带着实际诉求。"""
    rest = query.strip()
    for w in EXIT_WORDS:
        rest = rest.replace(w, "")
    rest = _EXIT_RESIDUE_RE.sub("", rest)
    return len(rest) >= 4 and not rest.isdigit()


def note_sop(session_id: str, sop_id: str = None, **extra) -> None:
    """把 SOP 进度记进 trace：本轮开始时停在哪、槽位填了什么；本轮内收尾的补上结果。"""
    from sops import get_sop_state, pop_finished

    state = get_sop_state(session_id) or pop_finished(session_id)
    fields = dict(extra)
    if state:
        slots = state.get("slots") or {}
        fields.update(id=state.get("sop_id"), step=state.get("step"), n_slots=len(slots),
                      slots=trace_store.brief(slots))
        if state.get("result"):
            fields["result"] = trace_store.brief(state["result"])
    elif sop_id:
        fields.update(id=sop_id, done=True)
    if fields:
        trace_store.step("sop", **fields)


def enter_sop(session_id: str, sop_id: str, query: str, **detail) -> str:
    """进一条 SOP：启动 + 记 trace 进度 + 打行为标签，返回要回给用户的话。"""
    from sops import start_sop
    from tools.context_store import set_last_sop

    slots = detail.pop("slots", None)
    result = start_sop(session_id, sop_id, query, slots=slots)
    reply = result.reply
    note_sop(session_id, sop_id)
    set_last_sop(session_id, sop_id, query)
    if reply:
        log_behavior("start_sop", sop=sop_id, **detail)
    return reply


def reenter_sop_on_retry(session_id: str, query: str) -> tuple:
    """「重新回答」时判断要不要重开 SOP，返回 (是否接管, 要回给用户的话)。"""
    from sops import sop_needs_enter_confirm, enter_confirm_text
    from sops.base import SOPS
    from tools.context_store import get_last_sop
    from tools.clarify_service import open_clarify

    last = trace_store.latest_turn(session_id)
    found = trace_store.find_turn(session_id, query)
    if (not found or not last
            or found.get("date") != last.get("date")
            or found.get("turn_index") != last.get("turn_index")):
        return False, ""
    prev = trace_store.find_turn_by_index(session_id, found.get("date"), found.get("turn_index"))
    sop_id = str((prev.get("sop") or {}).get("id") or "")
    if not sop_id or prev.get("branch") == "exit_sop" or sop_id not in SOPS:
        return False, ""
    entry = str(get_last_sop(session_id).get("entry_query") or query)
    logger.info("[Retry] 重开 SOP: %s（入口 query=%s）", sop_id, entry[:40])
    trace_store.node("sop.reenter", in_={"sop": sop_id, "entry": entry[:40]})
    if sop_needs_enter_confirm(sop_id, entry):
        open_clarify(session_id, pending_store.BIZ_SOP_ENTER,
                     state={"sop": sop_id, "entry": entry},
                     trace_in={"sop": sop_id, "via": "retry", "entry": entry[:40]})
        log_behavior("ask_clarify", kind="confirm_enter", via="retry")
        return True, enter_confirm_text(sop_id)
    return True, enter_sop(session_id, sop_id, entry, via="retry")
