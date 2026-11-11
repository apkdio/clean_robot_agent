"""SOP（标准操作流程）基础设施：会话状态 + 执行器。

一个 SOP 是一组有状态的步骤，通过槽位（slot）记忆上下文，逐步引导用户完成一个业务场景。
步骤类型：ask（提问收集槽位）、action（执行动作，如按预算检索）、reply（输出结果并结束）。
"""
import hashlib
import json
import time

from mapper import trace_store
from utils.dto.sop import ExtractResult, SopStepResult
from utils.log_tool import get_logger
from utils.prompts_tool import render
from service.safety import with_safety_hint
from mapper.state_store import StateStore
from config.word_dict_config import (
    AFTERSALES_WORDS,BUY_WORDS, CONSULT_WORDS, BRAND_WORDS,
    ENTER_CONFIRM,
    SOP_REENTER_HINT, SOP_REENTER_HINT_DEFAULT,
    SOP_ASK_FALLBACK, SOP_EMPTY_REPLY_FALLBACK,
)

logger = get_logger(name="sops_base")
SOPS = {}  # sop_id → sop 定义
# 定义形如 {"id": xxx, "name": xxx, "trigger": [xxx], "guards": [xxx], "intro": xxx,
#           "max_retry": xxx, "steps": [xxx]}；name 缺省回 id。
# register() 在落表前校验必填字段、step 唯一性、回调可调用与枚举值。
# ask 步：{"id": xxx, "type": "ask", "slot": xxx, "ask": str|fn, "retry": str|fn, "extract": fn}
#          fn 返回 ExtractResult(skip/done/fail)；done 的 extra 由 base 合并进 slots，
#          extra["trace"] 由 base 统一写节点。
# action 步：{"id": xxx, "type": "action", "action": fn}；fn 唯一入参是 slots 副本：
#            原件仍是 session["slots"]，fn 对副本的修改不回写；
#            副本带内部 "_slot_status": {slot: filled|skipped|dropped}，仅供 fn 读取，
#            slot_status 本身留在 session；两者都不进入模板上下文。
# reply 步：{"id": xxx, "type": "reply", "template": str|fn, "handoff": fn, "fallback": str}
#          template 只做展示；handoff(ctx) 返回 Handoff 时交给主链后续阶段，返回 None 再回 template/fallback。
# 三种步骤都可带 "when": fn 条件跳过；reply 的 fallback 用于空串或模板异常。
# 执行器每个循环迭代开一条 sop.step span；out 与 sop.flow 的条目同源。


def is_consulting(query: str) -> bool:
    """判断是否是「选购咨询」（选购要注意什么），而非「选购动作」（我要买）。
    需要同时命中两个词表才能判True
    """
    has_buy = any(w in query for w in BUY_WORDS)
    has_consult = any(w in query for w in CONSULT_WORDS)
    return has_buy and has_consult


def is_aftersales(query: str) -> bool:
    """判断是否是「售后咨询」（保修/售后/退换货），而非「选购咨询」。"""
    return any(w in query for w in AFTERSALES_WORDS)


def is_brand(query: str) -> bool:
    """判断是否是「品牌咨询」（为什么买/优势/介绍），而非「选购动作」。"""
    return any(w in query for w in BRAND_WORDS)


def _def_ver(sop: dict) -> str:
    """SOP 定义里「与续跑状态耦合」那部分的短指纹：步骤序列（id / 类型 / 槽位 / 是否带 when）与重试上限。

    只用于判断 Redis 里的半截状态还能不能用：文案（ask / retry / intro / template）与触发词、guards
    变了不影响续跑，不进指纹。
    """
    shape = [[s.get("id"), s.get("type"), s.get("slot"), bool(s.get("when"))] for s in sop.get("steps") or []]
    raw = json.dumps([shape, sop.get("max_retry", 2)], ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


ENTER_CONFIRM_DEFAULT = "always"          # 所有 SOP 默认「进入前先问」
ENTER_CONFIRM_MODES = ("always", "weak", "never")


def _sop_error(sop: dict, message: str, step: dict = None):
    sid = (sop or {}).get("id") or "?"
    if step is None:
        raise ValueError("[SOP] %s: %s" % (sid, message))
    raise ValueError("[SOP] %s step %s: %s" % (sid, step.get("id") or "?", message))


def _validate_sop(sop: dict) -> None:
    """注册前校验定义形状；错误带 SOP / step 定位，避免运行到一半才 KeyError。"""
    if not isinstance(sop, dict):
        raise ValueError("[SOP] 定义必须是 dict")
    sid = str(sop.get("id") or "").strip()
    if not sid:
        _sop_error(sop, "id 不能为空")
    sop["id"] = sid

    raw_name = sop.get("name")
    if raw_name is None:
        name = sid
    elif not isinstance(raw_name, str):
        _sop_error(sop, "name 必须是字符串")
    else:
        name = raw_name.strip() or sid
    sop["name"] = name

    steps = sop.get("steps")
    if not isinstance(steps, (list, tuple)) or not steps:
        _sop_error(sop, "steps 必须是非空列表")
    seen = set()
    for step in steps:
        if not isinstance(step, dict):
            _sop_error(sop, "step 必须是 dict")
        step_id = str(step.get("id") or "").strip()
        if not step_id:
            _sop_error(sop, "step id 不能为空", step)
        if step_id in seen:
            _sop_error(sop, "step id 重复", step)
        seen.add(step_id)
        kind = step.get("type")
        if kind not in ("ask", "action", "reply"):
            _sop_error(sop, "type 必须是 ask/action/reply", step)
        if step.get("when") is not None and not callable(step["when"]):
            _sop_error(sop, "when 必须可调用", step)

        if kind == "ask":
            slot = step.get("slot")
            if not isinstance(slot, str) or not slot.strip():
                _sop_error(sop, "ask 步骤需要非空 slot", step)
            for key in ("ask", "retry"):
                value = step.get(key)
                if key == "ask" and value is None:
                    _sop_error(sop, "ask 步骤需要 ask 文案", step)
                if value is not None and not isinstance(value, str) and not callable(value):
                    _sop_error(sop, "%s 必须是字符串或函数" % key, step)
            if not callable(step.get("extract")):
                _sop_error(sop, "ask 步骤需要可调用的 extract", step)
        elif kind == "action":
            if not callable(step.get("action")):
                _sop_error(sop, "action 步骤需要可调用的 action", step)
        else:
            tpl = step.get("template")
            handoff = step.get("handoff")
            if tpl is None and handoff is None:
                _sop_error(sop, "reply 步骤需要 template 或 handoff", step)
            if tpl is not None and not isinstance(tpl, str) and not callable(tpl):
                _sop_error(sop, "reply template 必须是字符串或函数", step)
            if handoff is not None and not callable(handoff):
                _sop_error(sop, "reply handoff 必须可调用", step)
            fallback = step.get("fallback")
            if fallback is not None and not isinstance(fallback, str):
                _sop_error(sop, "reply fallback 必须是字符串", step)

    guards = sop.get("guards")
    if guards is not None:
        if not isinstance(guards, (list, tuple)):
            _sop_error(sop, "guards 必须是列表")
        for guard in guards:
            if not isinstance(guard, dict) or not callable(guard.get("check")):
                _sop_error(sop, "guard 必须是含可调用 check 的 dict")

    max_retry = sop.get("max_retry", 2)
    if isinstance(max_retry, bool) or not isinstance(max_retry, int) or max_retry < 0:
        _sop_error(sop, "max_retry 必须是非负整数")

    mode = sop.get("enter_confirm")
    if mode is not None and str(mode).strip().lower() not in ENTER_CONFIRM_MODES:
        _sop_error(sop, "enter_confirm 必须是 always/weak/never")

    hint_key = sop.get("reenter_hint_key")
    if hint_key is not None and not isinstance(hint_key, str):
        _sop_error(sop, "reenter_hint_key 必须是字符串")


def register(sop: dict) -> None:
    """注册一个 SOP 定义，并算好它的定义指纹（供 Redis 里的半截状态对账）。"""
    _validate_sop(sop)
    sid = sop["id"]
    if sid in SOPS and SOPS[sid] is not sop:
        logger.warning("[SOP] 重复注册 id=%s，覆盖旧定义", sid)
    sop["def_ver"] = _def_ver(sop)
    SOPS[sid] = sop


def save_recommend(session_id: str, models) -> None:
    """保存指定会话的上一轮推荐结果到 meta（供追问使用，跨重启有效）。"""
    from mapper.context_store import set_last_models
    set_last_models(session_id, models)


# 当前活跃会话，按 session_id 隔离；Redis + 本地双写由 StateStore 统一处理
_sessions = {}
# 状态形如 {"sop_id": xxx, "def_ver": xxx, "step": xxx, "slots": {xxx},
#           "slot_status": {slot: skipped|dropped|filled}, "retry_count": xxx, "result": {xxx}}
# slot_status 只在 ask 步出现；dropped 表示重试耗尽，不能当成用户主动跳过。
# retry_count 只对当前 step 有效；所有推进都走 _advance()，跨 step 自动清零。
_finished = {}  # session_id -> 刚收尾的那份状态（形状同 _sessions，供「重新回答」判断 last_sop，取走即清）
_session_store = StateStore("sop:session:", lambda: _sessions, lambda: _sop_ttl())
_finished_store = StateStore("sop:finished:", lambda: _finished, lambda: _sop_ttl())


def _sop_ttl() -> int:
    from gateway.redis_store import sop_ttl
    return sop_ttl()


def make_session(sop_id: str, step: int = 0, slots: dict = None, retry_count: int = 0) -> dict:
    """造一份带定义指纹的会话状态（_start 与测试注入共用，避免手搓状态漏指纹）。"""
    return {"sop_id": sop_id, "def_ver": (SOPS.get(sop_id) or {}).get("def_ver"),
            "step": int(step), "slots": dict(slots or {}), "retry_count": int(retry_count)}


def _advance(session: dict) -> int:
    """推进到下一步并返回旧下标；retry_count 只属于当前 step，跨 step 必须清零。"""
    old = int(session.get("step", 0))
    session["step"] = old + 1
    session["retry_count"] = 0
    return old


def _same_def(session: dict) -> bool:
    """状态与当前 SOP 定义是否同一版（没带指纹的老状态、SOP 已被删掉，都算对不上）。"""
    sop = SOPS.get(session.get("sop_id") or "")
    return bool(sop) and session.get("def_ver") == sop.get("def_ver")


def _load_session(session_id: str):
    """读会话状态；与当前 SOP 定义对不上就丢掉它（代码改过 SOP、Redis 里的旧状态还在）。

    状态的 step 是步骤下标、slots 是槽位键名，定义一动就会指向别的步骤，续跑必然串场；
    这类状态一律作废，本轮按「没有活跃 SOP」正常作答，并留一条 WARNING 便于对账。
    """
    session = _session_store.get(session_id)
    if session and not _same_def(session):
        cur = (SOPS.get(session.get("sop_id") or "") or {}).get("def_ver") or "未注册"
        logger.warning("[SOP] SOP 定义已变，作废旧状态：sop=%s 状态指纹=%s 现定义=%s（session=%s）",
                       session.get("sop_id"), session.get("def_ver") or "无", cur, session_id)
        # 留下痕（节点与状态都在闭集里；by 的新取值见 notes/TRACE.md）
        trace_store.node("sop.active", status="skipped", by="stale_def",
                         in_={"sop": session.get("sop_id"),
                              "def": "%s→%s" % (session.get("def_ver") or "无", cur)})
        _session_store.delete(session_id)
        return None
    return session


def _save_session(session_id: str, session: dict):
    """写会话状态；StateStore 负责 Redis + 本地双写和过期。"""
    session["ts"] = time.time()
    _session_store.set(session_id, session)


def _delete_session(session_id: str):
    """删除会话状态。"""
    _session_store.delete(session_id)


def pop_finished(session_id: str) -> dict:
    """取走「刚收尾」的 SOP 状态（含槽位与 action 结果），给 trace 用；没取过就是空 dict。"""
    return _finished_store.pop(session_id) or {}


def has_active_sop(session_id: str) -> bool:
    return _load_session(session_id) is not None


def get_active_sop_id(session_id: str):
    """返回指定会话活跃 SOP 的 id；无活跃 SOP 时返回 None。"""
    s = _load_session(session_id)
    return s["sop_id"] if s else None


def get_sop_state(session_id: str) -> dict:
    """返回活跃 SOP 的进度（sop_id / step / slots / retry_count）；无活跃 SOP 时空 dict。"""
    return _load_session(session_id) or {}


def _start(session_id: str, sop_id: str, slots: dict = None):
    session = make_session(sop_id, slots=slots)
    _save_session(session_id, session)
    return session


def _end(session_id: str):
    _delete_session(session_id)


def end_sop(session_id: str, state: dict = None):
    """主动结束指定会话的 SOP（可传入本轮已加载的快照，避免重复读取）。"""
    s = _load_session(session_id) if state is None else state
    if s:
        logger.info(f"[SOP] SOP End:{s['sop_id']}")
    _end(session_id)


def _with_exit_hint(intro: str) -> str:
    """给 SOP 开场语追加退出提示（所有 SOP 通用，用户可回复「0」退出
    """
    return intro + "\n（随时可回复「0」退出本环节）"


def start_sop(session_id: str, sop_id: str, query: str, slots: dict = None) -> SopStepResult:
    """进入一个 SOP，返回 SopStepResult。

    首轮先尝试用触发 query 提取第一个槽位（预填用户已给的信息），
    提取失败再问第一个问题。`slots` 可由调用方预填，已填的 ask 槽直接跳过。
    开场提示（intro）可选，有则先输出。
    """
    from mapper.context_store import set_enter_declined
    set_enter_declined(session_id, False)   # 用户这次真进了流程，之前「别再问」的标记作废
    _start(session_id, sop_id, slots=slots)
    logger.info(f"[SOP] Start SOP: {sop_id}")
    result = _run(session_id, query)
    intro = SOPS[sop_id].get("intro")
    if intro:
        prefix = _with_exit_hint(intro) + "\n\n"
        return SopStepResult(result.kind, prefix + result.reply, result.done,
                             handoff=result.handoff)
    return result


def continue_sop(session_id: str, query: str, state: dict = None) -> SopStepResult:
    """继续指定会话的 SOP；无活跃 SOP 时返回 noop，不再返回 None。"""
    if state is None:
        state = _load_session(session_id)
    if state is None:
        return SopStepResult("noop")
    return _run(session_id, query, session=state)


def match_sop(query: str, sop_id: str = None, log: bool = True):
    """找命中的 SOP：trigger 命中，且它自己的 guards 都没命中（guard 只否掉自己这条）。

    给了 sop_id 就只看这一条（「这条 SOP 该不该走」的判定）；不给则按注册顺序返回第一条，都不命中返回 None。
    `log=False` 给探测/打分调用用：同一轮会探多次，只让真正派发那次打 INFO。
    """
    for sid, sop in SOPS.items():
        if sop_id and sid != sop_id:
            continue
        if not any(kw in query for kw in sop.get("trigger", [])):
            continue
        if any(g["check"](query) for g in sop.get("guards", [])):
            continue
        if log:
            logger.info(f"[SOP] Match SOP:{sid}")
        return sid
    return None


def sop_name(sop_id: str) -> str:
    """SOP 中文名取自定义；未知 id 原样返回，避免把「当前」当成真实流程名。"""
    if not sop_id:
        return "当前"
    sop = SOPS.get(sop_id) or {}
    return str(sop.get("name") or sop_id)


def enter_confirm_mode(sop_id: str) -> str:
    """SOP 自己声明进入门策略；不写默认 always，非法值告警后回默认。"""
    mode = str((SOPS.get(sop_id) or {}).get("enter_confirm") or ENTER_CONFIRM_DEFAULT).strip().lower()
    if mode not in ENTER_CONFIRM_MODES:
        logger.warning("[SOP] %s 的 enter_confirm=%r 非法，按 always 处理", sop_id, mode)
        return ENTER_CONFIRM_DEFAULT
    return mode


def sop_needs_enter_confirm(sop_id: str, query: str = None) -> bool:
    """进入确认门：always 一律问；never 不问；weak 只对弱触发问（旧语义）。"""
    mode = enter_confirm_mode(sop_id)
    if mode == "always":
        return True
    if mode == "never":
        return False
    sop = SOPS.get(sop_id) or {}
    weak = sop.get("weak_trigger") or []
    if not weak or not query:
        return False
    # 强触发词 = 全部命中词 - 弱触发词
    strong = [w for w in sop.get("trigger", []) if w not in weak]
    if any(w in query for w in strong):
        return False
    return any(w in query for w in weak)


def sop_has_strong_trigger(sop_id: str, query: str) -> bool:
    """命中强触发词（不是只用弱词表命中的泛问）；用于打分加分与拒绝后的放行。"""
    sop = SOPS.get(sop_id) or {}
    weak = set(sop.get("weak_trigger") or [])
    return any(w in query for w in sop.get("trigger", []) if w not in weak)


def enter_confirm_text(sop_id: str) -> str:
    return ENTER_CONFIRM.format(sop=sop_name(sop_id))


def enter_declined(session_id: str, sop_id: str) -> bool:
    from mapper.context_store import is_enter_declined
    return is_enter_declined(session_id, sop_id)


def mark_enter_declined(session_id: str, sop_id: str) -> None:
    from mapper.context_store import set_enter_declined
    set_enter_declined(session_id, sop_id=sop_id)


def enter_decline_hint(sop_id: str) -> str:
    """固定话术留在词表；SOP 定义可用 reenter_hint_key 指定，缺省按 sop_id 取。"""
    sop = SOPS.get(sop_id) or {}
    key = sop.get("reenter_hint_key") or sop_id
    return SOP_REENTER_HINT.get(key, SOP_REENTER_HINT_DEFAULT)


def _step_text(value, ctx: dict) -> str:
    """步骤文案可以是字符串或 (ctx) -> str 的函数。"""
    if callable(value):
        value = value(ctx)
    return value or ""


def _ask_text(step: dict, key: str, ctx: dict,
              sop_id: str = "", step_id: str = "") -> str:
    """渲染 ask/retry 文案；ask 缺参时先用 retry 重问，再不行用统一兜底。"""
    text = _step_text(step.get(key, step["ask"]), ctx)
    label = "sop:%s:%s:%s" % (sop_id or "?", step_id or "?", key)
    if not text:
        return SOP_ASK_FALLBACK

    rendered = render(label, template=text, **ctx)
    if rendered is not None:
        return rendered

    logger.warning("[SOP] %s step %s 的 %s 模板不可用，使用兜底话术",
                   sop_id or "?", step_id or "?", key)
    if key == "ask":
        retry = _step_text(step.get("retry"), ctx)
        if retry:
            rendered = render(label + ":retry", template=retry, **ctx)
            if rendered is not None:
                return rendered
    return SOP_ASK_FALLBACK


def _close_step(span_id, status: str, out: dict) -> None:
    """关闭当前 sop.step span；span_id 为空时 no-op。"""
    if span_id is not None:
        trace_store.end_span(span_id, status=status, out=out)


def _write_extract_trace(result: ExtractResult) -> None:
    """extractor 不直接写 trace；需要留痕时把事实放在 extra["trace"]，由这里统一写。"""
    fact = (result.extra or {}).get("trace")
    if not fact:
        return
    if not isinstance(fact, dict) or not fact.get("node"):
        logger.warning("[SOP] extract trace 缺 node，忽略：%r", fact)
        return
    trace_store.node(
        str(fact["node"]),
        status=str(fact.get("status") or "ok"),
        in_=fact.get("in"),
        out=fact.get("out"),
    )


def _extract_slot_extra(result: ExtractResult) -> dict:
    """只有 done 才把 extra 写进 slots；trace 是记录指令，不是槽位。"""
    return {k: v for k, v in (result.extra or {}).items() if k != "trace"}


def _run(session_id: str, user_input: str, session: dict = None) -> SopStepResult:
    """执行/推进指定会话的 SOP；session 可传入本轮已加载的快照。"""
    if session is None:
        session = _load_session(session_id)
    if session is None:
        return SopStepResult("noop")
    sop = SOPS[session["sop_id"]]
    steps = sop["steps"]
    ended = False  # 是否已结束（reply/异常防御里 _end 置 True）
    flow = []      # 本轮走到的动作链（进 trace）
    step_span = None

    try:
        while session["step"] < len(steps):
            step = steps[session["step"]]
            ctx = {**session["slots"], **session.get("result", {})}
            step_in = {"i": session["step"], "type": step["type"],
                       "slot": step.get("slot"), "retry": session.get("retry_count", 0)}
            step_out = {}
            step_span = trace_store.begin_span("sop.step", kind="step", in_=step_in)
            step_status = "ok"

            def _record(entry):
                flow.append(entry)
                step_out.update(entry)

            # 针对网点的结果，如果结果只有一个，跳过反问澄清环节
            if step.get("when") and not step["when"](ctx):
                old_i = _advance(session)
                _record({"i": old_i, "type": step["type"], "outcome": "skipped"})
                _close_step(step_span, "skipped", step_out)
                step_span = None
                continue

            if step["type"] == "ask":
                slot = step.get("slot")
                # 预填槽：不重复问；进入下一槽/动作
                if slot and session["slots"].get(slot) is not None:
                    old_i = _advance(session)
                    _record({"i": old_i, "type": "ask", "slot": slot,
                             "outcome": "skipped", "by": "prefill"})
                    user_input = None
                    _close_step(step_span, "skipped", step_out)
                    step_span = None
                    continue
                # 有用户输入 → 尝试提取槽位；否则 → 问问题
                if user_input is None:
                    logger.info("[SOP] %s ask slot=%s", sop["id"], step.get("slot"))
                    _record({"i": session["step"], "type": "ask",
                             "slot": step.get("slot"), "outcome": "asked"})
                    _close_step(step_span, "ok", step_out)
                    step_span = None
                    return SopStepResult("ask", _ask_text(
                        step, "ask", ctx, sop["id"], step.get("id")))
                result = step["extract"](user_input, session["slots"])
                _write_extract_trace(result)
                if result.kind == "skip":
                    # 显式留空：不重试，直接跳过该槽
                    session["slots"][slot] = None
                    session.setdefault("slot_status", {})[slot] = "skipped"
                    old_i = _advance(session)
                    user_input = None
                    _record({"i": old_i, "type": "ask",
                             "slot": slot, "outcome": "skipped"})
                    _close_step(step_span, "skipped", step_out)
                    step_span = None
                    continue
                if result.kind == "fail":
                    # 提取失败：累加重试次数，超过上限则放弃该槽位
                    session["retry_count"] = session.get("retry_count", 0) + 1
                    if session["retry_count"] > sop.get("max_retry", 2):
                        # 放弃的槽位不放进 slots，避免随 ctx 进入模板；状态单独记 dropped
                        session["slots"].pop(step["slot"], None)
                        session.setdefault("slot_status", {})[step["slot"]] = "dropped"
                        old_i = _advance(session)
                        user_input = None
                        logger.warning("[SOP] %s slot %s abandoned after retries (reason=%s)",
                                       sop["id"], step["slot"], result.reason or "-")
                        _record({"i": old_i, "type": "ask",
                                 "slot": step.get("slot"), "outcome": "abandoned"})
                        _close_step(step_span, "skipped", step_out)
                        step_span = None
                        continue
                    # 首轮提取失败（retry_count==1）→ 用 ask 话术（还没问过用户）
                    # 后续提取失败（retry_count≥2）→ 用 retry 话术（重问）
                    logger.info("[SOP] %s slot %s extract failed (retry=%d, reason=%s)",
                                sop["id"], step["slot"], session["retry_count"], result.reason or "-")
                    if session["retry_count"] == 1:
                        _record({"i": session["step"], "type": "ask",
                                 "slot": step.get("slot"), "outcome": "asked"})
                        _close_step(step_span, "ok", step_out)
                        step_span = None
                        return SopStepResult("ask", with_safety_hint(
                            _ask_text(step, "ask", ctx, sop["id"], step.get("id")),
                            session_id, by="sop", sop=sop["id"]))
                    _record({"i": session["step"], "type": "ask",
                             "slot": step.get("slot"), "outcome": "retry"})
                    _close_step(step_span, "ok", step_out)
                    step_span = None
                    return SopStepResult("ask", with_safety_hint(
                        _ask_text(step, "retry", ctx, sop["id"], step.get("id")),
                        session_id, by="sop", sop=sop["id"]))
                # 提取成功：先合并 extra（如网点候选），再写槽位本身，最后推进
                session["slots"].update(_extract_slot_extra(result))
                session["slots"][step["slot"]] = result.value
                session.setdefault("slot_status", {})[step["slot"]] = "filled"
                old_i = _advance(session)
                user_input = None
                logger.info("[SOP] %s slot %s = %s", sop["id"], step["slot"], result.value)
                _record({"i": old_i, "type": "ask", "slot": step.get("slot"),
                         "outcome": "filled",
                         "value": result.value if len(str(result.value)) <= 40
                         else str(result.value)[:40] + "…"})
                _close_step(step_span, "ok", step_out)
                step_span = None
                continue

            elif step["type"] == "action":
                # 执行 skill，结果存入 result；只有含结构化型号列表时才写 n_models
                action_slots = dict(session["slots"])
                action_slots["_slot_status"] = dict(session.get("slot_status") or {})
                result = step["action"](action_slots)
                if not isinstance(result, dict):
                    _sop_error(sop, "action 必须返回 dict，实际 %s" % type(result).__name__, step)
                session["result"] = result
                entry = {"i": session["step"], "type": "action", "outcome": "ok"}
                if "models" in result:
                    n_models = len(result.get("models") or [])
                    entry["n_models"] = n_models
                    save_recommend(session_id, result["models"])
                    logger.info("[SOP] %s action executed (%d models)", sop["id"], n_models)
                else:
                    logger.info("[SOP] %s action executed", sop["id"])
                _record(entry)
                _advance(session)
                _close_step(step_span, "ok", step_out)
                step_span = None
                continue

            elif step["type"] == "reply":
                # 输出结果并结束；template 只做展示，handoff 负责把后续阶段交给主链。
                ctx = {**session["slots"], **session.get("result", {})}
                _record({"i": session["step"], "type": "reply", "outcome": "done"})
                handoff_factory = step.get("handoff")
                if callable(handoff_factory):
                    handoff = handoff_factory(ctx)
                    if handoff is not None:
                        ended = True
                        _finished_store.set(session_id, session)
                        _end(session_id)
                        _close_step(step_span, "hit", step_out)
                        step_span = None
                        logger.info("[SOP] %s finished (handoff)", sop["id"])
                        return SopStepResult("reply", done=True, handoff=handoff)
                tpl = step.get("template")
                if callable(tpl):
                    reply = tpl(ctx)
                elif tpl is None:
                    reply = step.get("fallback", SOP_EMPTY_REPLY_FALLBACK)
                else:
                    label = "sop:%s:%s:reply" % (sop["id"], step.get("id") or "?")
                    reply = render(label, template=tpl, **ctx)
                    if reply is None:
                        logger.warning("[SOP] %s step %s 的 reply 模板不可用，使用 fallback",
                                       sop["id"], step.get("id") or "?")
                        reply = step.get("fallback", SOP_EMPTY_REPLY_FALLBACK)
                reply = reply if isinstance(reply, str) else ""
                if not reply.strip():
                    logger.warning("[SOP] %s step %s empty reply, fallback",
                                   sop["id"], step.get("id"))
                    reply = step.get("fallback") or SOP_EMPTY_REPLY_FALLBACK
                ended = True
                _finished_store.set(session_id, session)
                _end(session_id)
                _close_step(step_span, "hit", step_out)
                step_span = None
                logger.info("[SOP] %s finished", sop["id"])
                return SopStepResult("reply", reply, done=True)

        # 步骤走完但无 reply（异常防御），安全结束
        ended = True
        _finished_store.set(session_id, session)
        _end(session_id)
        logger.warning("[SOP] %s ended without reply", sop["id"])
        return SopStepResult("done", SOP_EMPTY_REPLY_FALLBACK, done=True)
    except Exception as e:  # noqa: BLE001
        # 步骤函数（when / extract / action）炸了：结束会话，别让下一轮还停在这一步重复失败；
        # 异常照常向上抛——外层 ask_stream 出兜底话术、trace 记 error
        logger.warning("[SOP] %s 第 %s 步执行失败（%s）：%s",
                       sop["id"], session["step"], type(e).__name__, e)
        _close_step(step_span, "error", step_out if "step_out" in locals() else {})
        step_span = None
        cur = steps[session["step"]] if session["step"] < len(steps) else {}
        trace_store.node("sop.step_error", status="error",
                         in_={"step": session["step"], "type": cur.get("type"), "slot": cur.get("slot")})
        ended = True        # 让下面的 finally 不再把状态写回去
        _end(session_id)
        raise
    finally:
        _close_step(step_span, "ok", step_out if "step_out" in locals() else {})
        trace_store.step("sop", flow=flow)
        # 会话未结束（停在 ask 等下一轮）→ 保存最新状态
        if not ended:
            _save_session(session_id, session)
