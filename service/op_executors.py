"""操作执行体：按操作名执行具体业务，返回统一的 ExecOutcome。

执行体只做业务判断与确定性输出，不直接接触 SSE / webapp。
"""
from __future__ import annotations

from service import turn_recorder
from tools import trace_store
from tools.dto.executor import ExecOutcome, NotMine, Reply
from tools.log_tool import get_logger
from service.model_query import (format_model_query, is_model_detail_query,
                               resolve_model_query)
from service.query_rewrite import history_block
from service.sops.runtime import enter_sop_result
from service.turn_context import TurnContext

logger = get_logger(name="op_executors")

PIPELINE_OPS = {"domain"}


def _resolve_model_query(session_id: str, query: str):
    """型号查询兜底：命中触发词 → LLM 提取型号名 → 按名精准检索型号详情。"""
    if not is_model_detail_query(query):
        return None
    from tools.llm_extract import extract_model_names
    from service.model_catalog import search_models_by_names
    from tools.metadata_extractor import extract_model_aspect, format_models

    try:
        model_names = extract_model_names(query, history=history_block(session_id))
        if model_names:
            models = search_models_by_names(model_names)
            if models:
                aspect = extract_model_aspect(query)
                return format_models(models, "您问的型号信息如下：", aspect)
    except Exception as e:  # noqa: BLE001
        logger.warning("[Agent] Model tool calling failed: %s", e)
    return None


def exec_followup(ctx: TurnContext) -> ExecOutcome:
    from service.followup import handle_followup

    reply = handle_followup(ctx.session_id, ctx.q)
    if reply:
        trace_store.node("followup", status="hit")
        turn_recorder.log_behavior("structured_answer", kind="followup")
        return Reply(reply)
    return NotMine()


def exec_model(ctx: TurnContext) -> ExecOutcome:
    reply = _resolve_model_query(ctx.session_id, ctx.q)
    if reply:
        trace_store.node("model_query", status="hit")
        turn_recorder.log_behavior("structured_answer", kind="model_detail")
        return Reply(reply)
    return NotMine()


def exec_find_models(ctx: TurnContext) -> ExecOutcome:
    from service.model_query import constraints_to_slots, is_model_catalog_intent

    result = resolve_model_query(ctx.q)
    if result:
        turn_recorder.log_behavior("structured_answer",
                                   kind="series" if result.kind == "series" else "filter",
                                   n=len(result.models))
        return Reply(format_model_query(result))
    if is_model_catalog_intent(ctx.q):
        result = enter_sop_result(ctx.session_id, "model_query", ctx.q,
                                  slots=constraints_to_slots(ctx.q))
        return Reply(result.reply, handoff=result.handoff)
    return NotMine()


def exec_service_point(ctx: TurnContext) -> ExecOutcome:
    from service.sops import enter_confirm_text, enter_declined, sop_needs_enter_confirm
    from tools import pending_store
    from service.clarify_service import open_clarify

    has_coords = ctx.lng is not None and ctx.lat is not None
    trace_store.node("service_point", status="hit",
                     out={"mode": "coords" if has_coords else "sop"})
    if has_coords:
        from tools.service_point import format_service_points, search_service_points

        points, origin = search_service_points(lng=ctx.lng, lat=ctx.lat)
        turn_recorder.log_behavior("structured_answer", kind="service_point")
        return Reply(format_service_points(points, origin))
    if (sop_needs_enter_confirm("service_point", ctx.q)
            and not enter_declined(ctx.session_id, "service_point")):
        open_clarify(ctx.session_id, pending_store.BIZ_SOP_ENTER,
                     state={"sop": "service_point", "entry": ctx.q},
                     trace_in={"sop": "service_point", "via": "point"})
        turn_recorder.log_behavior("ask_clarify", kind="confirm_enter", via="point")
        return Reply(enter_confirm_text("service_point"))
    result = enter_sop_result(ctx.session_id, "service_point", ctx.q)
    return Reply(result.reply, handoff=result.handoff)


def exec_sop(ctx: TurnContext) -> ExecOutcome:
    from service.sops import (enter_confirm_text, enter_declined, match_sop,
                      sop_has_strong_trigger, sop_needs_enter_confirm)
    from tools import pending_store
    from service.clarify_service import open_clarify

    sop_id = match_sop(ctx.q)
    if not sop_id:
        return NotMine()
    gate = sop_needs_enter_confirm(sop_id, ctx.q)
    strong = sop_has_strong_trigger(sop_id, ctx.q)
    trace_store.node("sop.match", status="hit",
                     out={"sop": sop_id, "gate": gate, "strong": strong})
    if gate:
        if ctx.enter_declined_turn or enter_declined(ctx.session_id, sop_id):
            if not strong:
                logger.info("[SOP] enter declined, skip gate: %s", sop_id)
                trace_store.step("sop_gate", skipped="declined", sop=sop_id)
                return NotMine()
            result = enter_sop_result(ctx.session_id, sop_id, ctx.q, via="strong")
            return Reply(result.reply, handoff=result.handoff)
        open_clarify(ctx.session_id, pending_store.BIZ_SOP_ENTER,
                     state={"sop": sop_id, "entry": ctx.q},
                     trace_in={"sop": sop_id, "via": "strong" if strong else "weak"})
        turn_recorder.log_behavior("ask_clarify", kind="confirm_enter")
        return Reply(enter_confirm_text(sop_id))
    result = enter_sop_result(ctx.session_id, sop_id, ctx.q,
                              via="strong" if strong else "weak")
    return Reply(result.reply, handoff=result.handoff)


OP_EXECUTORS = {
    "followup": exec_followup,
    "model": exec_model,
    "find_models": exec_find_models,
    "service_point": exec_service_point,
    "sop": exec_sop,
}
