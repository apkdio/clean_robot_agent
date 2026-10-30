"""统一澄清服务：pending 状态只管存储，这里按业务 id 认领并解析回答。"""
from __future__ import annotations

import re

from config.word_dict_config import (
    BLOCK_INJECT_REPLY as _BLOCK_INJECT_REPLY,   # noqa: F401  兼容旧引用
    ENTER_CONFIRM as _ENTER_CONFIRM,
    CLARIFY_GUESS as _CLARIFY_GUESS,
    EXIT_CONFIRM as _EXIT_CONFIRM,
    OPERATION_LABELS as _OPERATION_LABELS,
    OPS_CLARIFY_RETRY as _OPS_CLARIFY_RETRY,
    SOP_ENTER_FAILED_REPLY as _SOP_ENTER_FAILED_REPLY,
)
from tools import pending_store, trace_store
from tools.dto.clarify import ClarifyData, ClarifyResult
from tools.dto import clarify as _clarify_dto
from tools.log_tool import get_logger
from tools.safety import with_safety_hint
from tools.sop_runtime import enter_sop, sop_name, note_sop
from tools.turn_recorder import log_behavior

logger = get_logger(name="clarify_service")

CLARIFY_DONE = _clarify_dto.CLARIFY_DONE
CLARIFY_RESUME = _clarify_dto.CLARIFY_RESUME
CLARIFY_DROP = _clarify_dto.CLARIFY_DROP
_CLARIFY_KINDS = _clarify_dto.CLARIFY_KINDS
_CONFIRM_CLEAN_RE = re.compile(r"[\s，,。.！!？?~～、]+")
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3}


def apply_resume(data: ClarifyData, query: str, q: str, enter_declined_turn: bool,
                 prefer_op: str | None) -> tuple:
    """把 resume 载荷应用到本轮状态，返回 (query, q, declined, prefer)。"""
    if data.enter_declined:
        enter_declined_turn = True
    if data.prefer_op:
        prefer_op = data.prefer_op
    if data.query != query:
        query = data.query
        from tools.query_rewrite import normalize_enabled

        if normalize_enabled():
            from tools.query_normalizer import normalize

            q = normalize(query)
            trace_store.step("rewrite", normalize={"changed": q != query, "text": q[:120]})
        else:
            q = query
    return query, q, enter_declined_turn, prefer_op


def match_op_choice(text: str, options: list):
    """解析操作澄清的选择：1/2（或中文数字、"第一个"）+ 选项别名；认不出返回 None。"""
    t = _CONFIRM_CLEAN_RE.sub("", text or "")
    if not t:
        return None
    m = re.match(r"^[选第]?([1-9])[个项号]?$", t)
    if not m:
        m = re.match(r"^[选第]?([一两二三])[个项号]?$", t)
        if m:
            idx = _CN_NUM.get(m.group(1))
            return options[idx - 1] if idx and idx <= len(options) else None
        for i, op in enumerate(options, 1):
            labels = _OPERATION_LABELS.get(op, ())
            if any(lb in t for lb in labels):
                return op
        return None
    idx = int(m.group(1))
    return options[idx - 1] if 1 <= idx <= len(options) else None


def match_yes_no(query: str, gate: str = "exit") -> bool | None:
    """确认门判据：True=是、False=不是、None=认不出（调用方再问一次，有界两次）。"""
    from config.word_dict_config import (
        CONFIRM_YES_WORDS, CONFIRM_NO_WORDS, CONFIRM_SHORT_YES, CONFIRM_EXIT_ONLY,
    )

    text = _CONFIRM_CLEAN_RE.sub("", query or "")
    if not text:
        return None
    if any(w in text for w in CONFIRM_NO_WORDS):
        return False
    if gate == "exit":
        for w, val in sorted(CONFIRM_EXIT_ONLY.items(), key=lambda kv: len(kv[0]), reverse=True):
            if w in text:
                return val
    if any(w in text for w in CONFIRM_YES_WORDS):
        return True
    if text in CONFIRM_SHORT_YES:
        return True
    return None


def open_clarify(session_id: str, biz: str, state: dict = None,
                 trace_in: dict = None, **data) -> dict:
    """开澄清并落统一节点：提问轮能追到「开了哪个澄清、覆盖了谁的」。"""
    prev = pending_store.peek(session_id)
    rec = pending_store.ask(session_id, biz, state=state, **data)
    opts = data.get("options") or []
    facts = {"biz": biz, "retry": int(rec.get("retry") or 0),
             "options": "/".join(opts) if opts else "-"}
    if trace_in:
        facts.update(trace_in)
    trace_store.node("clarify.open", in_=facts,
                     out={"replaces": (prev or {}).get("biz") or "-"})
    return rec


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
        _EXIT_CONFIRM.format(sop=name), session_id, by="clarify"))


def _clarify_sop_enter(session_id: str, query: str, _q: str) -> ClarifyResult:
    """SOP 进入确认门：答「是」进引导，「不是」恢复原句直接作答。"""
    from sops import SOPS, enter_decline_hint, has_active_sop, mark_enter_declined

    rec = pending_store.take(session_id, pending_store.BIZ_SOP_ENTER)
    if not rec:
        return ClarifyResult.drop()
    st = rec.get("state") or {}
    sop_id = st.get("sop") or "model_query"
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
        reply = enter_sop(session_id, sop_id, entry, via="confirm_enter")
        return ClarifyResult.done(reply or _SOP_ENTER_FAILED_REPLY)
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
        _ENTER_CONFIRM.format(sop=sop_name(sop_id)), session_id, by="clarify"))


def _clarify_ops(session_id: str, query: str, _q: str) -> ClarifyResult:
    """操作澄清：答 1/2（或操作别名）按选择重放；两次认不出取 top1。"""
    rec = pending_store.take(session_id, pending_store.BIZ_OPS_CLARIFY)
    if not rec:
        return ClarifyResult.drop()
    st = rec.get("state") or {}
    opts = list((rec.get("data") or {}).get("options") or [])
    retry = int(rec.get("retry") or 0)

    def _label(op):
        return (_OPERATION_LABELS.get(op) or (op,))[0]

    if not opts:
        pending_store.close(session_id, pending_store.BIZ_OPS_CLARIFY)
        logger.warning("[Clarify] ops.clarify 没有候选，退出澄清 session=%s", session_id)
        return ClarifyResult.drop()
    if len(opts) == 1:
        # 单候选不是真正的二选一；直接按唯一候选继续，避免拼出 "1=x，2=-"。
        pending_store.close(session_id, pending_store.BIZ_OPS_CLARIFY)
        trace_store.node("ops.clarify", status="skipped",
                         in_={"top1": opts[0]},
                         out={"pick": opts[0]})
        return ClarifyResult.resume(st.get("query") or query, prefer_op=opts[0])

    pick = match_op_choice(query, opts)
    if pick:
        pending_store.close(session_id, pending_store.BIZ_OPS_CLARIFY)
        trace_store.node("ops.clarify",
                         in_={"top1": opts[0], "top2": opts[1] if len(opts) > 1 else "-"},
                         out={"pick": pick})
        return ClarifyResult.resume(st.get("query") or query, prefer_op=pick)
    if retry >= 1:
        pending_store.close(session_id, pending_store.BIZ_OPS_CLARIFY)
        trace_store.node("ops.clarify", status="miss",
                         in_={"top1": opts[0] if opts else "-"},
                         out={"pick": opts[0] if opts else "-"})
        return ClarifyResult.resume(
            st.get("query") or query,
            prefix=_CLARIFY_GUESS.format(a=_label(opts[0])) + "\n\n" if opts else "",
            prefer_op=opts[0] if opts else None)
    pending_store.bump(session_id, pending_store.BIZ_OPS_CLARIFY)
    log_behavior("ask_clarify", kind="ops_clarify_retry")
    return ClarifyResult.done(with_safety_hint(_OPS_CLARIFY_RETRY.format(
        a=_label(opts[0]) if opts else "-",
        b=_label(opts[1]) if len(opts) > 1 else "-"), session_id, by="clarify"))


CLARIFY_HANDLERS = {
    pending_store.BIZ_SOP_EXIT: _clarify_sop_exit,
    pending_store.BIZ_SOP_ENTER: _clarify_sop_enter,
    pending_store.BIZ_OPS_CLARIFY: _clarify_ops,
}
