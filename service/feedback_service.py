"""反馈用例门面：校验标注、定位轮次并写入 trace 旁路。"""
from __future__ import annotations

from config.word_dict_config import FEEDBACK_REASON_KEYS, FEEDBACK_REASONS
from mapper import trace_store
from utils.dto.feedback import FeedbackRequest, FeedbackResult
from utils.log_tool import get_logger

logger = get_logger(name="feedback_service")


def reasons() -> dict:
    return FEEDBACK_REASONS


def submit(request: FeedbackRequest) -> FeedbackResult:
    """校验并写入反馈；Controller 只负责把 status 映射成 HTTP 码。"""
    session_id = request.session_id
    query = request.query.strip()
    rating = request.rating
    reason = request.reason.strip()
    note = request.note.strip()
    if rating not in ("useful", "useless"):
        return FeedbackResult("error", message="rating 必须是 useful / useless")
    if rating == "useless":
        # 无用必须给一个枚举原因（含 other）：否则标完没法聚类；有用不带原因
        if reason not in FEEDBACK_REASON_KEYS:
            return FeedbackResult(
                "error",
                message="无用原因必须是 %s 之一" % "/".join(FEEDBACK_REASON_KEYS),
            )
    else:
        reason = ""

    ctx = {}
    if request.date is not None and request.turn_index is not None:
        # 上下文消息里带了该轮锚点：直接按（日期, 轮次）取，不再用 query 猜轮次
        ctx = trace_store.find_turn_by_index(
            session_id, request.date, request.turn_index)
        if ctx and ctx.get("query") != query:
            ctx = {}          # 锚点与提交的问句对不上，不当锚点用
    if not ctx:
        ctx = trace_store.find_turn(session_id, query)
    if not ctx:
        # 没对上轮次：标注没地方落，如实告诉前端
        logger.warning("[Feedback] no matching turn: session=%s query=%s",
                       session_id, query[:40])
        return FeedbackResult("no_turn", message="没找到对应的轮次（可能已超出回看天数）")

    marked = trace_store.feedback_of(session_id, ctx.get("date"), ctx.get("turn_index"))
    # 同一轮可以再来一次（改主意）：追加一条、不改历史，读侧按轮次取最后一条为准
    trace_store.feedback_turn(
        ctx, rating=rating,
        reason=reason[:40],
        note=note[:200],
    )
    logger.info("[Feedback] turn=%s/%s rating=%s reason=%s session=%s%s",
                ctx.get("date"), ctx.get("turn_index"), rating,
                reason or "-", session_id, " (revise)" if marked else "")
    return FeedbackResult("ok", date=ctx.get("date"),
                          turn_index=ctx.get("turn_index"),
                          updated=bool(marked))
