"""RAG 编排层：扫地机器人知识库的端到端问答流水线。

主链路是固定顺序的阶段序列：硬规则 → 归一化 → 澄清 → 活跃 SOP → 重答
→ 意图 → 安全承接 → 操作判别 → 管线（结构化 / 检索 / 生成）。
顺序即优先级，不允许阶段自行动态跳转。
"""

import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable

from tools import pending_store, trace_store
from tools.config_tool import load_config
from tools.log_tool import clear_log_session, get_logger, set_log_session
from tools.query_rewrite import (
    history_block as _history_block,
    last_user_message as _last_user_message,
    normalize_enabled as _normalize_enabled,
    rewrite_cfg as _rewrite_cfg,
    rewrite_query as _rewrite_query,
    window_domain as _window_domain,
)
from tools.safety import (recent_carry_count, safety_carry_max, safety_carry_text,
                          window_danger_word)
from tools.sop_runtime import (
    enter_sop as _enter_sop,
    has_residual_request as _has_residual_request,
    is_symbols_only as _is_symbols_only,
    note_sop as _note_sop,
    reenter_sop_on_retry as _reenter_sop_on_retry,
    sop_name as _sop_name,
)
from tools.clarify_service import (
    CLARIFY_DONE,
    CLARIFY_RESUME,
    CLARIFY_HANDLERS as _CLARIFY_HANDLERS,
    apply_resume,
    open_clarify as _open_clarify,
)
from tools.domain_router import (
    domain_filter as _domain_filter,
    route_domain as _route_domain,
)
from tools import turn_recorder
from tools.dto.executor import NotMine, Reply
from tools.dto.request import ChatRequest
from tools.dto.stage import Continue, Finish, StageOutcome
from tools.turn_context import TurnContext
from tools.turn_recorder import log_behavior as _log_behavior
from tools.op_executors import OP_EXECUTORS as _OP_EXEC, PIPELINE_OPS as _PIPELINE_OPS
from tools.generation import casual_stream, rag_stream

from config.word_dict_config import (
    EMOTION_MILD,
    EMOTION_STRONG,
    EXIT_WORDS,
    NO_ANSWER_REPLIES,
    REFUSE_OFFTOPIC_REPLY as _REFUSE_OFFTOPIC_REPLY,
    SOP_ENTER_FAILED_REPLY as _SOP_ENTER_FAILED_REPLY,
)
from config.word_dict_config import BLOCK_INJECT_REPLY as _BLOCK_INJECT_REPLY
from config.word_dict_config import EXIT_CONFIRM as _EXIT_CONFIRM
from config.word_dict_config import OPS_CLARIFY as _OPS_CLARIFY
from tools.llm_tool import stream_chat

logger = get_logger(name="agent")

_agent_cfg = load_config("agent")
_llm_cfg = _agent_cfg.get("llm", {})
_behavior = _agent_cfg.get("behavior", {})

_hybrid_retriever = None


def _get_retriever():
    """懒加载双路召回器（稠密 + 稀疏 → RRF）。"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from tools.hybrid_retriever import HybridRetriever

        _hybrid_retriever = HybridRetriever()
        _hybrid_retriever.ensure_sparse_index()
    return _hybrid_retriever


def detect_emotion(query: str) -> str | None:
    """检测用户负面情绪，命中返回安抚话术（未命中返回 None）。只安抚、不拦截。"""
    if any(w in query for w in EMOTION_STRONG):
        logger.info("[Emotion] Strong Negative")
        return "非常抱歉给您带来不好的体验，我马上帮您处理～"
    if any(w in query for w in EMOTION_MILD):
        logger.info("[Emotion] Negative")
        return "别着急，我帮您看看～"
    logger.info("[Emotion] Normal")
    return None


def _ctx_cfg() -> dict:
    from tools.query_rewrite import cfg

    return cfg()


def _resolve_date_filter(query: str) -> dict | None:
    """兼容读侧/评测的日期 filter 适配器；日期解析与守卫在 model_query。"""
    from function_tools.date_tool import build_date_filter
    from tools.model_query import resolve_date_range

    r = resolve_date_range(query)
    return build_date_filter(r[0], r[1]) if r else None


# ── 阶段实现 ────────────────────────────────────────────────────

def _stage_normalize(ctx: TurnContext) -> StageOutcome:
    """入口归一化与重答回滚；硬规则统一用归一化文本判定。"""
    from tools.context_store import rollback_last_user_message

    ctx.q = ctx.query
    if _normalize_enabled():
        from tools.query_normalizer import normalize

        ctx.q = normalize(ctx.query)
        trace_store.step("rewrite", normalize={"changed": ctx.q != ctx.query,
                                               "text": ctx.q[:120]})
    if ctx.retry and _last_user_message(ctx.session_id) == ctx.query:
        rollback_last_user_message(ctx.session_id)
    return Continue()


def _stage_inject_guard(ctx: TurnContext) -> StageOutcome:
    """角色扮演 / 指令注入：直接拒绝，不发给 LLM。"""
    from tools.context_store import append_message
    from config.word_dict_config import INJECT_RE

    if not INJECT_RE.search(ctx.q):
        trace_store.route("inject", hit=False)
        return Continue()
    trace_store.route("inject", hit=True)
    append_message(ctx.session_id, "user", ctx.query, blocked="inject",
                   trace=trace_store.turn_anchor())
    logger.warning("[Guard] injection blocked: %s", ctx.query[:60])
    _log_behavior("block_inject")
    return Finish(text=_BLOCK_INJECT_REPLY)


def _stage_danger_guard(ctx: TurnContext) -> StageOutcome:
    """危险现象：安全优先，立即停机联系售后。"""
    from tools.context_store import append_message
    from config.word_dict_config import DANGER_WORDS

    danger_hit = next((w for w in DANGER_WORDS if w in ctx.q), None)
    if not danger_hit:
        trace_store.route("danger", hit=False)
        return Continue()
    trace_store.route("danger", hit=True, word=danger_hit)
    append_message(ctx.session_id, "user", ctx.query, danger=True,
                   trace=trace_store.turn_anchor())
    logger.warning("[Guard] danger word hit: %s | %s", danger_hit, ctx.query[:60])
    from sops import end_sop

    end_sop(ctx.session_id)
    pending_store.close(ctx.session_id)
    _log_behavior("stop_use_safety", word=danger_hit)
    return Finish(text=(
        "请立即停止使用机器人并断开电源！涉及冒烟/烧焦/进水等安全风险，"
        "不要自行拆机或继续充电，请马上联系官方售后（400-860-1314）处理。"))


def _stage_record_user(ctx: TurnContext) -> StageOutcome:
    """记录用户消息；danger 标记供后续安全承接使用。"""
    from tools.context_store import append_message

    append_message(ctx.session_id, "user", ctx.query, trace=trace_store.turn_anchor())
    return Continue()


def _stage_emotion(ctx: TurnContext) -> StageOutcome:
    """负面情绪：先安抚一句，再继续正常流程。"""
    reply = detect_emotion(ctx.q)
    trace_store.route("emotion", hit=bool(reply))
    return Continue(prefix=reply + "\n\n" if reply else "")


def _stage_pending_clarify(ctx: TurnContext) -> StageOutcome:
    """统一澄清链路：按 biz 分派给业务处理器。"""
    pending = pending_store.peek(ctx.session_id)
    if not pending:
        return Continue()
    biz = pending.get("biz") or "-"
    trace_store.node("clarify", in_={"biz": biz,
                                     "retry": int(pending.get("retry") or 0)})
    handler = _CLARIFY_HANDLERS.get(biz)
    if handler is None:
        logger.warning("[Clarify] 未登记的业务 %r，退出澄清", biz)
        pending_store.close(ctx.session_id)
        return Continue()
    result = handler(ctx.session_id, ctx.query, ctx.q)
    if result.code == CLARIFY_DONE:
        return Finish(text=result.message)
    if result.code == CLARIFY_RESUME:
        ctx.query, ctx.q, ctx.enter_declined_turn, ctx.prefer_op = apply_resume(
            result.data, ctx.query, ctx.q, ctx.enter_declined_turn, ctx.prefer_op)
        return Continue(prefix=result.message)
    return Continue()


def _stage_active_sop(ctx: TurnContext) -> StageOutcome:
    """SOP 会话：有活跃 SOP 时继续该流程，不经过意图路由。"""
    from sops import continue_sop, end_sop, get_active_sop_id, get_sop_state, has_active_sop

    if not has_active_sop(ctx.session_id):
        return Continue()
    sop_id = get_active_sop_id(ctx.session_id)
    name = _sop_name(sop_id)
    trace_store.node("sop.active", status="hit", in_={"sop": sop_id})
    _note_sop(ctx.session_id, sop_id)

    if ctx.q.strip() == "0":
        pending_store.close(ctx.session_id)
        end_sop(ctx.session_id)
        _log_behavior("exit_sop", via="zero")
        return Finish(text=f"好的，已退出「{name}」环节～")

    if any(w in ctx.q for w in EXIT_WORDS) or _is_symbols_only(ctx.q):
        symbols_only = _is_symbols_only(ctx.q)
        if not symbols_only and _has_residual_request(ctx.q):
            pending_store.close(ctx.session_id)
            end_sop(ctx.session_id)
            # 退出后还要继续理解原句，这是中间动作，不占用本轮行为出口。
            trace_store.node("sop.exit", status="hit",
                             in_={"sop": sop_id, "via": "exit_word_with_request"})
            return Continue(prefix=f"好的，已退出「{name}」环节～")
        st = get_sop_state(ctx.session_id)
        _open_clarify(ctx.session_id, pending_store.BIZ_SOP_EXIT,
                      state={"sop_id": sop_id, "step": st.get("step"),
                             "slots": st.get("slots") or {}},
                      trace_in={"sop": sop_id, "step": st.get("step")})
        _log_behavior("ask_clarify",
                      kind="confirm_exit_symbols" if symbols_only else "confirm_exit")
        return Finish(text=_EXIT_CONFIRM.format(sop=name))
    result = continue_sop(ctx.session_id, ctx.query)
    if result is None or result.kind == "noop":
        return Continue()
    _note_sop(ctx.session_id, sop_id, done=bool(result.done))
    if result.reply:
        _log_behavior("sop_step")
    return Finish(text=result.reply or "")


def _stage_retry_reopen_sop(ctx: TurnContext) -> StageOutcome:
    """「重新回答」：上一轮是已结束的 SOP 轮 → 重开同一 SOP。"""
    from sops import has_active_sop

    if not ctx.retry or has_active_sop(ctx.session_id):
        return Continue()
    reentered, reply = _reenter_sop_on_retry(ctx.session_id, ctx.query)
    if not reentered:
        return Continue()
    return Finish(text=reply or _SOP_ENTER_FAILED_REPLY)


def _stage_intent(ctx: TurnContext) -> StageOutcome:
    """本地分类头：robot / casual / other / unknown；归一化可疑时用原文再判一次。"""
    from tools.intent_router import get_guess_hint, route_intent_with_margin

    low_conf_margin = float((_ctx_cfg().get("intent") or {}).get("low_conf_margin", 0) or 0)
    ctx.intent, ctx.margin = route_intent_with_margin(ctx.q)
    if ctx.intent == "unknown" and ctx.margin == 0.0:
        trace_store.route("intent", degraded=True)
        trace_store.node("route.intent", status="degraded")
    ctx.intent_retry = None
    if ctx.q != ctx.query and (
            ctx.intent == "other"
            or (ctx.intent == "casual" and 0 < ctx.margin < low_conf_margin)):
        intent2, margin2 = route_intent_with_margin(ctx.query)
        adopt = intent2 != "other"
        ctx.intent_retry = {"label": intent2, "margin": round(margin2, 6),
                            "adopted": adopt}
        logger.info("[Normalize] 两遍分类：归一化后 %s(%.3f) → 原文 %s(%.3f)，%s",
                    ctx.intent, ctx.margin, intent2, margin2,
                    "采纳原文判定" if adopt else "两遍都判 other")
        if adopt:
            ctx.intent, ctx.margin = intent2, margin2
    trace_store.step("intent", intent=ctx.intent, margin=ctx.margin,
                     intent_retry=ctx.intent_retry)
    ctx.low_conf = low_conf_margin > 0 and ctx.margin < low_conf_margin
    if ctx.intent == "unknown":
        return Continue(prefix=get_guess_hint() + "\n\n")
    return Continue()


def _stage_safety_carry(ctx: TurnContext) -> StageOutcome:
    """近期发生过安全告警 → 主链路直接承接（零检索、零 LLM）。"""
    danger_word = window_danger_word(ctx.session_id)
    reply = safety_carry_text(ctx.session_id, ctx.q, intent=ctx.intent, margin=ctx.margin)
    if not reply:
        return Continue()
    high_conf_margin = float((_ctx_cfg().get("intent") or {}).get("high_conf_margin", 0) or 0)
    _log_behavior("carry_safety", danger=danger_word or "-")
    trace_store.node("guard.safety_carry", status="hit",
                     in_={"count": recent_carry_count(ctx.session_id),
                          "max": safety_carry_max(),
                          "need_margin": high_conf_margin,
                          "margin": round(ctx.margin, 3)},
                     out={"chars": len(reply)})
    return Finish(text=reply)


def _stage_other_downgrade(ctx: TurnContext) -> StageOutcome:
    """低置信 other 不硬拒答：上文有可承接话题则降级 unknown。"""
    if ctx.intent != "other" or not ctx.low_conf:
        return Continue()
    if _window_domain(ctx.session_id):
        logger.info("[Intent] low-confidence other (margin=%.3f) + 可承接话题 → 降级为 unknown",
                    ctx.margin)
        ctx.intent = "unknown"
        trace_store.route("intent", downgrade="unknown")
    else:
        logger.info("[Intent] low-confidence other (margin=%.3f) 无可承接话题 → 维持拒答",
                    ctx.margin)
    return Continue()


def _stage_ops_decide(ctx: TurnContext) -> StageOutcome:
    """操作判别器：按分数与词表产出候选，失败按原链路继续。"""
    try:
        from tools.operations import note_ops

        ctx.op_note = note_ops(ctx.q, intent=ctx.intent, margin=ctx.margin,
                               prefer=ctx.prefer_op)
    except Exception as e:  # noqa: BLE001
        logger.warning("[Ops] 判别失败，回退原链路：%s", e)
        ctx.op_note = {"usable": []}
    return Continue()


def _stage_ops_clarify(ctx: TurnContext) -> StageOutcome:
    """判别器 ambiguous 且候选动作不同 → 先问用户选哪个。"""
    from sops import match_sop

    usable = ctx.op_note.get("usable") or []
    if ctx.prefer_op or ctx.op_note.get("state") != "ambiguous" or len(usable) < 2:
        return Continue()
    try:
        sop_id = match_sop(ctx.q, log=False)
    except Exception:  # noqa: BLE001
        sop_id = None

    def _effect(op):
        if op == "sop":
            return "service_point" if sop_id == "service_point" else "sop:%s" % sop_id
        return op

    if _effect(usable[0]) == _effect(usable[1]):
        return Continue()
    scores = ctx.op_note.get("scores") or {}
    diff = float(scores.get(usable[0], 0.0)) - float(scores.get(usable[1], 0.0))
    _open_clarify(ctx.session_id, pending_store.BIZ_OPS_CLARIFY,
                  state={"query": ctx.q},
                  options=[usable[0], usable[1]],
                  trace_in={"top1": usable[0], "top2": usable[1], "gap": round(diff, 3)})
    _log_behavior("ask_clarify", kind="ops_clarify")
    from config.word_dict_config import OPERATION_LABELS

    a = (OPERATION_LABELS.get(usable[0]) or (usable[0],))[0]
    b = (OPERATION_LABELS.get(usable[1]) or (usable[1],))[0]
    return Finish(text=_OPS_CLARIFY.format(a=a, b=b))


def _stage_ops_dispatch(ctx: TurnContext) -> StageOutcome:
    """按派发顺序执行操作体：接管即结束，不是该管的试下一个。"""
    for name in ctx.op_note.get("usable") or []:
        if name in _PIPELINE_OPS:
            break
        exec_ = _OP_EXEC.get(name)
        if exec_ is None:
            continue
        reply = exec_(ctx)
        if isinstance(reply, NotMine):
            trace_store.route("ops", veto=name)
            continue
        if not isinstance(reply, Reply):
            raise TypeError("操作执行体 %s 返回了未知 DTO：%r" % (name, reply))
        return Finish(text=reply.text)
    return Continue()


def _stage_intent_exit(ctx: TurnContext) -> StageOutcome:
    """other 拒答 / casual 闲聊出口。"""
    if ctx.intent == "other":
        from config.word_dict_config import REPAIR_STRONG_WORDS
        from sops import match_sop

        if any(w in ctx.q for w in REPAIR_STRONG_WORDS):
            sop_id = match_sop(ctx.q)
            if sop_id == "repair":
                reply = _enter_sop(ctx.session_id, sop_id, ctx.q, via="strong_words")
                return Finish(text=reply or _SOP_ENTER_FAILED_REPLY)
        if _route_domain(ctx.q):
            logger.info("[Intent] other 但命中知识域 → 不拒答，交给 RAG：%s", ctx.q[:40])
            trace_store.node("guard.refuse", status="skipped", by="domain_hit")
            return Continue()
        _log_behavior("refuse_offtopic")
        return Finish(text=_REFUSE_OFFTOPIC_REPLY)
    if ctx.intent == "casual":
        _log_behavior("chitchat")
        return Finish(stream=casual_stream(
            ctx.query, temperature=_llm_cfg.get("temperature", 0.3),
            stop_check=ctx.stop_check, stream_chat_func=stream_chat))
    return Continue()


def _stage_retrieve(ctx: TurnContext) -> StageOutcome:
    """检索：改写 → 域定向 → 双路召回 → 注入过滤。"""
    from config.word_dict_config import INJECT_RE

    if ctx.stop_check is not None and ctx.stop_check():
        logger.info("[Agent] stopped before retrieval")
        return Finish(text="")
    hr = _get_retriever()
    turn_recorder.shadow_probe(ctx.q, ctx.session_id, "kb_search", "RAG 检索")

    t0 = time.perf_counter()
    if ctx.low_conf or ctx.intent == "unknown":
        effective_query, rewrite_via = _rewrite_query(ctx.session_id, ctx.q)
    else:
        effective_query, rewrite_via = ctx.q, "none"
    trace_store.add_ms("rewrite", t0)
    trace_store.step("rewrite", via=rewrite_via,
                     effective_query=effective_query[:200])

    t0 = time.perf_counter()
    domain_value, domain_via = _domain_filter(effective_query)
    retrieve_facts = {}
    ranked = hr.search_with_scores(
        effective_query,
        filter={"file_name": domain_value} if domain_value else None,
        trace_out=retrieve_facts)
    chunks = [doc for doc, _score, _meta in ranked]
    trace_store.add_ms("retrieve", t0)
    trace_store.retrieval(**retrieve_facts)
    trace_store.step("retrieve", domain=domain_value, via=domain_via,
                     n_chunks=len(chunks), chunks=trace_store.chunk_brief(ranked))

    if (not chunks and effective_query != ctx.q
            and _rewrite_cfg().get("retry_with_origin", True)):
        logger.info("[Rewrite] no hit via %s, retry with origin over full KB: %s",
                    rewrite_via, ctx.q[:40])
        ranked = hr.search_with_scores(ctx.q, filter=None)
        chunks = [doc for doc, _score, _meta in ranked]
        trace_store.retrieval(source="retry",
                              retry={"rewrite_via": rewrite_via,
                                     "n_chunks": len(chunks)})
        trace_store.step("retrieve", retried=True, n_chunks=len(chunks),
                         chunks=trace_store.chunk_brief(ranked))

    trace_store.chunks(ranked)
    if chunks:
        kept = [c for c in chunks if not INJECT_RE.search(c.page_content or "")]
        if len(kept) != len(chunks):
            logger.warning("[Guard] 检索片段命中注入特征，丢弃 %d/%d 片",
                           len(chunks) - len(kept), len(kept))
            trace_store.step("retrieve", dropped_inject=len(chunks) - len(kept))
            chunks = kept

    if not chunks:
        logger.info("[RAG] 0 chunk retrieved, fall back to canned reply (query=%s)",
                    ctx.query[:40])
        _log_behavior("no_answer_fallback", hits=0)
        return Finish(text=random.choice(NO_ANSWER_REPLIES))
    if _behavior.get("retrieval_only", False):
        _log_behavior("retrieve_answer", mode="retrieval_only")
        src = chunks[0].metadata.get("file_name", "")
        return Finish(text=f"来源：{src}\n\n{chunks[0].page_content}")
    ctx.chunks = chunks
    return Continue()


def _stage_generate(ctx: TurnContext) -> StageOutcome:
    """RAG 流式生成：上下文 + 召回片段 + 答案硬事实校验。"""
    _log_behavior("retrieve_answer", hits=len(ctx.chunks))
    return Finish(stream=rag_stream(
        ctx.query,
        history_block=_history_block(ctx.session_id),
        chunk_texts=[c.page_content for c in ctx.chunks],
        temperature=_llm_cfg.get("temperature", 0.3),
        stop_check=ctx.stop_check,
        stream_chat_func=stream_chat,
    ))


@dataclass(frozen=True)
class StageSpec:
    """一个阶段的名字、trace 节点和实现；顺序由 _STAGES 决定。"""

    id: str
    node: str
    fn: Callable[[TurnContext], StageOutcome]


_STAGES = (
    StageSpec("normalize", "stage.normalize", _stage_normalize),
    StageSpec("inject", "stage.inject", _stage_inject_guard),
    StageSpec("danger", "stage.danger", _stage_danger_guard),
    StageSpec("record_user", "stage.record_user", _stage_record_user),
    StageSpec("emotion", "stage.emotion", _stage_emotion),
    StageSpec("clarify", "stage.clarify", _stage_pending_clarify),
    StageSpec("sop", "stage.sop", _stage_active_sop),
    StageSpec("sop_retry", "stage.sop_retry", _stage_retry_reopen_sop),
    StageSpec("intent", "stage.intent", _stage_intent),
    StageSpec("safety_carry", "stage.safety_carry", _stage_safety_carry),
    StageSpec("other_downgrade", "stage.other_downgrade", _stage_other_downgrade),
    StageSpec("ops_decide", "stage.ops_decide", _stage_ops_decide),
    StageSpec("ops_clarify", "stage.ops_clarify", _stage_ops_clarify),
    StageSpec("ops_dispatch", "stage.ops_dispatch", _stage_ops_dispatch),
    StageSpec("intent_exit", "stage.intent_exit", _stage_intent_exit),
    StageSpec("retrieve", "stage.retrieve", _stage_retrieve),
    StageSpec("generate", "stage.generate", _stage_generate),
)


def _ctx_mark(ctx: TurnContext) -> tuple:
    """阶段前后对比用的轻量指纹，不深拷贝 chunks。"""
    op = ctx.op_note if isinstance(ctx.op_note, dict) else {}
    return (ctx.query, ctx.q, ctx.intent, ctx.margin, ctx.low_conf,
            repr(ctx.intent_retry), ctx.prefer_op, ctx.enter_declined_turn,
            op.get("state"), op.get("winner"), tuple(op.get("usable") or []),
            id(ctx.chunks), len(ctx.chunks or ()))


def _stage_acted(outcome: StageOutcome, changed: bool) -> bool:
    return isinstance(outcome, Finish) or bool(outcome.prefix) or changed


def _stage_out(outcome: StageOutcome | None, acted: bool, stream_completed=None,
               error: BaseException = None) -> dict:
    if outcome is None:
        out = {"result": "error", "acted": bool(acted)}
    elif isinstance(outcome, Continue):
        out = {"result": "continue", "prefix_chars": len(outcome.prefix),
               "acted": bool(acted)}
    else:
        out = {"result": "finish", "prefix_chars": len(outcome.prefix),
               "text_chars": len(outcome.text or ""),
               "stream": bool(outcome.stream), "acted": bool(acted)}
        if stream_completed is not None:
            out["stream_completed"] = bool(stream_completed)
    if error is not None:
        out["error"] = type(error).__name__
    return out


def _record_stage(spec: StageSpec, outcome: StageOutcome | None, started: float,
                  acted: bool, stream_completed=None, error: BaseException = None) -> None:
    status = "error" if error is not None else (
        "hit" if isinstance(outcome, Finish) else "ok")
    trace_store.stage(spec.node, status=status,
                      ms=int((time.perf_counter() - started) * 1000),
                      out=_stage_out(outcome, acted, stream_completed, error))


def _run_stages(ctx: TurnContext) -> Iterator[str]:
    """按固定顺序跑阶段：前缀先输出，Finish 即本轮出口。

    阶段 span 由这里统一落；流式出口要等流消费完才关，避免父节点先于 answer.verify。
    """
    for spec in _STAGES:
        started = time.perf_counter()
        before = _ctx_mark(ctx)
        try:
            outcome = spec.fn(ctx)
        except Exception as exc:  # noqa: BLE001
            _record_stage(spec, None, started, True, error=exc)
            raise
        if isinstance(outcome, Continue):
            _record_stage(spec, outcome, started,
                          _stage_acted(outcome, _ctx_mark(ctx) != before))
            if outcome.prefix:
                yield outcome.prefix
            continue
        if isinstance(outcome, Finish):
            acted = _stage_acted(outcome, _ctx_mark(ctx) != before)
            if outcome.stream is None:
                _record_stage(spec, outcome, started, acted)
                if outcome.prefix:
                    yield outcome.prefix
                if outcome.text:
                    yield outcome.text
                return
            if outcome.prefix:
                yield outcome.prefix
            if outcome.text:
                yield outcome.text
            completed = False
            failed = False
            try:
                yield from outcome.stream
                completed = True
            except Exception as exc:  # noqa: BLE001
                failed = True
                _record_stage(spec, outcome, started, acted, error=exc)
                raise
            finally:
                if not failed:
                    _record_stage(spec, outcome, started, acted,
                                  stream_completed=completed)
            return
        error = TypeError("阶段返回了未知 outcome：%r" % (outcome,))
        _record_stage(spec, None, started, True, error=error)
        raise error


def _ask_stream(query: str, session_id: str = "default", lng=None, lat=None,
                retry: bool = False, stop_check=None):
    """旧主链入口：保留给测试与低层调用，行为与 ask_stream 相同。"""
    request = ChatRequest(query=query, session_id=session_id,
                          lng=lng, lat=lat, retry=retry)
    yield from _run_stages(TurnContext.from_request(request, stop_check))


def ask_stream(request: ChatRequest | str, session_id: str = "default",
               lng=None, lat=None, retry: bool = False, stop_check=None):
    """流式问答入口：开一轮 trace，交给阶段序列，收尾时落盘。

    request 传 ChatRequest 时其余参数忽略；传字符串时按旧签名构造请求。
    """
    if isinstance(request, ChatRequest):
        req = request
    else:
        req = ChatRequest(query=request, session_id=session_id,
                          lng=lng, lat=lat, retry=retry)
    trace_store.begin_turn(req.session_id, req.query,
                           retry=True if req.retry else None)
    set_log_session(req.session_id)
    completed = False
    reply_parts = []
    try:
        for piece in _run_stages(TurnContext.from_request(req, stop_check)):
            reply_parts.append(piece)
            yield piece
        completed = not (stop_check is not None and stop_check())
    except Exception as exc:  # noqa: BLE001
        trace_store.note_error(exc)
        raise
    finally:
        trace_store.end_turn(aborted=not completed, reply="".join(reply_parts))
        clear_log_session()
