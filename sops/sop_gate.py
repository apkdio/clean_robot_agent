"""SOP 进入/退出确认门：只注册澄清 handler，不参与主链分派。"""
from __future__ import annotations

from config.word_dict_config import (
    ENTER_CONFIRM,
    EXIT_CONFIRM,
    SOP_ENTER_FAILED_REPLY,
)
from tools import pending_store, trace_store
from tools.clarify_service import (
    match_yes_no,
    register_clarify_handler,
)
from tools.dto.clarify import ClarifyResult
from tools.log_tool import get_logger
from tools.safety import with_safety_hint
from sops.runtime import enter_sop_result, note_sop, sop_name
from tools.turn_recorder import log_behavior

logger = get_logger(name="sop_gate")


def _clarify_sop_exit(session_id: str, _query: str, q: str) -> ClarifyResult:
    """SOP 退出确认门：答「是」退出，「不是」继续。"""
    from sops import end_sop, get_active_sop_id, get_sop_state, has_active_sop

    rec = pending_store.take(session_id, pending_store.BIZ_SOP_EXIT)
    if not rec:
        return ClarifyResult.drop()
    if not has_active_sop(session_id):
        pending_store.close(session_id, pending_store.BIZ_SOP_EXIT)
        logger.info("[Clarify] sop.exit 但无活跃 SOP，作废 session=%s", session_id)
        return ClarifyResult.drop()

    st = rec.get("state") or {}
    cur = get_sop_state(session_id)
    if st.get("step") is not None and cur.get("step") != st.get("step"):
        pending_store.close(session_id, pending_store.BIZ_SOP_EXIT)
        logger.warning("[Clarify] sop.exit 状态已变化（step %s → %s），作废",
                       st.get("step"), cur.get("step"))
        return ClarifyResult.drop()

    sop_id = st.get("sop_id") or get_active_sop_id(session_id)
    name = sop_name(sop_id)
    note_sop(session_id, sop_id)

    ans = match_yes_no(q)
    retry = int(rec.get("retry") or 0)
    if ans is False:
        pending_store.close(session_id, pending_store.BIZ_SOP_EXIT)
        log_behavior("sop_step", via="confirm_no")
        return ClarifyResult.done("好的，那我们继续刚才的话题～")
    if ans is True:
        pending_store.close(session_id, pending_store.BIZ_SOP_EXIT)
        end_sop(session_id)
        log_behavior("exit_sop", via="confirm")
        return ClarifyResult.done(f"好的，已退出「{name}」环节～")
    if retry >= 1:
        pending_store.close(session_id, pending_store.BIZ_SOP_EXIT)
        end_sop(session_id)
        log_behavior("exit_sop", via="confirm_timeout")
        return ClarifyResult.done(f"好的，已退出「{name}」环节～有新的问题可以直接问我。")
    pending_store.bump(session_id, pending_store.BIZ_SOP_EXIT)
    log_behavior("ask_clarify", kind="confirm_exit_retry")
    return ClarifyResult.done(with_safety_hint(
        EXIT_CONFIRM.format(sop=name), session_id, by="clarify"))


def _clarify_sop_enter(session_id: str, query: str, _q: str) -> ClarifyResult:
    """SOP 进入确认门：答「是」进引导，「不是」恢复原句直接作答。"""
    from sops import SOPS, enter_decline_hint, has_active_sop, mark_enter_declined

    rec = pending_store.take(session_id, pending_store.BIZ_SOP_ENTER)
    if not rec:
        return ClarifyResult.drop()
    st = rec.get("state") or {}
    sop_id = st.get("sop")
    if not sop_id:
        pending_store.close(session_id, pending_store.BIZ_SOP_ENTER)
        logger.warning("[Clarify] sop.enter 缺 sop，作废 session=%s", session_id)
        trace_store.node("sop.enter_gate", status="skipped",
                         out={"decision": "missing_sop"})
        return ClarifyResult.drop()
    if sop_id not in SOPS:
        pending_store.close(session_id, pending_store.BIZ_SOP_ENTER)
        logger.warning("[Clarify] 待确认里的 SOP 已不存在，作废：%s", sop_id)
        return ClarifyResult.drop()
    if has_active_sop(session_id):
        pending_store.close(session_id, pending_store.BIZ_SOP_ENTER)
        logger.warning("[Clarify] sop.enter 但已有活跃 SOP，作废 session=%s", session_id)
        return ClarifyResult.drop()

    entry = st.get("entry") or query
    ans = match_yes_no(query, gate="enter")
    retry = int(rec.get("retry") or 0)
    if ans is not None:
        decision = "yes" if ans else "no"
    elif retry >= 1:
        decision = "timeout"
    else:
        decision = "unknown"
    trace_store.node("sop.enter_gate",
                     in_={"answer": query[:40], "entry": entry[:40]},
                     out={"decision": decision})
    if ans is True:
        pending_store.close(session_id, pending_store.BIZ_SOP_ENTER)
        result = enter_sop_result(session_id, sop_id, entry, via="confirm_enter")
        if result.handoff is not None:
            return ClarifyResult.done_handoff(result.handoff, result.reply)
        return ClarifyResult.done(result.reply or SOP_ENTER_FAILED_REPLY)
    if ans is False:
        pending_store.close(session_id, pending_store.BIZ_SOP_ENTER)
        mark_enter_declined(session_id, sop_id)
        logger.info("[SOP] enter declined: %s", sop_id)
        return ClarifyResult.resume(
            entry, prefix=enter_decline_hint(sop_id) + "\n\n", enter_declined=True)
    if retry >= 1:
        pending_store.close(session_id, pending_store.BIZ_SOP_ENTER)
        # 用户没有明确拒绝，只是两次都没答清；不写持久拒绝标记，但本轮仍跳过进入门，
        # 否则恢复原句后会在同一轮再次问门。
        logger.info("[SOP] enter gate timed out: %s", sop_id)
        return ClarifyResult.resume(entry, enter_declined=True)
    pending_store.bump(session_id, pending_store.BIZ_SOP_ENTER)
    log_behavior("ask_clarify", kind="confirm_enter_retry")
    return ClarifyResult.done(with_safety_hint(
        ENTER_CONFIRM.format(sop=sop_name(sop_id)), session_id, by="clarify"))


register_clarify_handler(pending_store.BIZ_SOP_EXIT, _clarify_sop_exit)
register_clarify_handler(pending_store.BIZ_SOP_ENTER, _clarify_sop_enter)
