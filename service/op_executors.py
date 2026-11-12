"""操作执行体：按操作名执行具体业务，返回统一的 ExecOutcome。

执行体只做业务判断与确定性输出，不直接接触 SSE / webapp。
"""
from __future__ import annotations

import re

from service import turn_recorder
from mapper import trace_store
from utils.dto.executor import ExecOutcome, NotMine, Reply
from utils.log_tool import get_logger
from service.model_query import (format_model_query, is_model_detail_query,
                               resolve_model_query)
from service.query_rewrite import history_block
from service.sops.runtime import enter_sop_result
from service.turn_context import TurnContext

logger = get_logger(name="op_executors")

PIPELINE_OPS = {"domain"}

_CITY_HINT_RE = re.compile(r"(?:在|离|从|到|去|位于)([\u4e00-\u9fff]{2,8})")


def _has_city_hint(query: str) -> bool:
    """句子里有明确的地点线索才调 LLM，避免「离我最近」被抽成幻觉城市。"""
    for match in _CITY_HINT_RE.finditer(query or ""):
        token = match.group(1)
        if token[0] in "我你他她它这那" or token.startswith(("最近", "附近", "哪")):
            continue
        return True
    return False


def _models_from_previous_turn(session_id: str):
    """meta 没有型号时，从上一轮 user + assistant 文本里做确定性型号匹配。"""
    from mapper.context_store import get_recent
    from service.model_catalog import models_in_text

    msgs = get_recent(session_id)
    if msgs and msgs[-1].get("role") == "user":
        msgs = msgs[:-1]
    idx = next((i for i in range(len(msgs) - 1, -1, -1)
                if msgs[i].get("role") == "assistant"), None)
    if idx is None:
        return []
    turn = [msgs[idx].get("content") or ""]
    for msg in reversed(msgs[:idx]):
        if msg.get("role") == "user":
            turn.append(msg.get("content") or "")
            break
    found = models_in_text("\n".join(turn))
    return found if 0 < len(found) <= 3 else []


def _resolve_model_query(session_id: str, query: str):
    """型号查询兜底：先绑定上一轮推荐，绑定不到再让 LLM 从上下文提取型号名。"""
    if not is_model_detail_query(query):
        return None
    from mapper.context_store import get_last_models, set_last_models
    from service.llm_extract import extract_model_names
    from service.model_catalog import model_name_in_query, models_in_text, search_models_by_names
    from utils.metadata_rules import extract_model_aspect, format_models

    try:
        models = models_in_text(query) if model_name_in_query(query) else []
        if not models:
            prev = get_last_models(session_id) or []
            # 「这两款」只在上一轮确实只展示少量型号时绑定，避免把整页列表当成指代对象。
            if 0 < len(prev) <= 3:
                model_names = ",".join(str(m.get("name") or "") for m in prev if m.get("name"))
                models = search_models_by_names(model_names)
        if not models:
            models = _models_from_previous_turn(session_id)
        if not models:
            model_names = extract_model_names(query, history=history_block(session_id))
            models = search_models_by_names(model_names) if model_names else []
        if models:
            set_last_models(session_id, models)
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
    from mapper.context_store import set_last_models
    from service.model_query import (constraints_to_slots, has_unsupported_param,
                                     is_model_catalog_intent)

    result = resolve_model_query(ctx.q)
    if result and not has_unsupported_param(ctx.q):
        set_last_models(ctx.session_id, result.models)
        turn_recorder.log_behavior("structured_answer",
                                   kind="series" if result.kind == "series" else "filter",
                                   n=len(result.models))
        return Reply(format_model_query(result))
    if is_model_catalog_intent(ctx.q):
        if ctx.enter_declined_turn:
            trace_store.step("sop_gate", skipped="declined_turn", sop="model_query")
            return NotMine()
        result = enter_sop_result(ctx.session_id, "model_query", ctx.q,
                                  slots=constraints_to_slots(ctx.q))
        return Reply(result.reply, handoff=result.handoff)
    return NotMine()


def exec_service_point(ctx: TurnContext) -> ExecOutcome:
    from service.sops import enter_confirm_text, enter_declined, sop_needs_enter_confirm
    from mapper import pending_store
    from mapper.service_points_repo import city_in_query, geocode_city
    from service.clarify_service import open_clarify
    from service.service_point import format_service_points, search_service_points

    cities = geocode_city(ctx.q) or city_in_query(ctx.q)
    city_via = "query"
    if not cities and _has_city_hint(ctx.q):
        try:
            from service.llm_extract import extract_city_candidates

            cities = extract_city_candidates(ctx.q) or []
            city_via = "llm"
        except Exception as e:  # noqa: BLE001
            logger.warning("[ServicePoint] LLM city extract failed: %s", e)
    query_city = cities[0] if len(cities) == 1 else None
    trace_store.node("service_point", status="hit",
                     out={"mode": "llm_city" if query_city and city_via == "llm"
                          else ("query_city" if query_city else "sop")})
    if query_city:
        points, _ = search_service_points(lng=query_city["lng"], lat=query_city["lat"])
        origin = query_city.get("cn_name") or query_city.get("name") or ""
        turn_recorder.log_behavior("structured_answer", kind="service_point")
        return Reply(format_service_points(points, origin))
    if ctx.enter_declined_turn:
        trace_store.step("sop_gate", skipped="declined_turn", sop="service_point")
        return NotMine()
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
    from mapper import pending_store
    from service.clarify_service import open_clarify

    sop_id = match_sop(ctx.q)
    if not sop_id:
        return NotMine()
    gate = sop_needs_enter_confirm(sop_id, ctx.q)
    strong = sop_has_strong_trigger(sop_id, ctx.q)
    trace_store.node("sop.match", status="hit",
                     out={"sop": sop_id, "gate": gate, "strong": strong})
    if gate:
        if ctx.enter_declined_turn:
            # 同一轮刚拒绝进入：先答原问题，强触发也不能反向把用户拉回 SOP。
            logger.info("[SOP] enter declined this turn, skip gate: %s", sop_id)
            trace_store.step("sop_gate", skipped="declined_turn", sop=sop_id)
            return NotMine()
        if enter_declined(ctx.session_id, sop_id):
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
