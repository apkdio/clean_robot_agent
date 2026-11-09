"""故障排查 SOP：问现象 → 检索 → LLM 生成排查步骤。"""

from sops.base import register, is_aftersales
from config.word_dict_config import (
    REPAIR_NO_SYMPTOM_REPLY,
    REPAIR_TRIGGER,
    REPAIR_WEAK_TRIGGER,
    SYMPTOM_MAP,
)

from tools.dto.handoff import Handoff
from tools.dto.sop import ExtractResult
from tools.llm_tool import get_small_model_name
from tools.log_tool import get_logger

# 故障现象兜底用的小模型（默认 3b 更快；精度不够可在 agent.yaml 调 llm.small_model）
logger = get_logger(name="repair_sop")

def _extract_symptom(text: str, _slots: dict):
    """从用户描述提取故障现象，映射成标准检索 query。

    规则（关键词）优先；规则 miss 时用 3b function calling 兜底归类口语故障。
    trace 只以 extra["trace"] 描述，由 base 统一写入。
    """
    # 1. 关键词规则
    for kw, query in SYMPTOM_MAP.items():
        if kw in text:
            logger.info(f"[Repair] symptom {kw}: {query}")
            return ExtractResult("done", query, extra={"trace": {
                "node": "sop.symptom",
                "in": {"text": text[:40]},
                "out": {"query": query, "via": "rule", "kw": kw},
            }})
    logger.warning("[Repair] Keyword match failed! Using LLM to match!")

    # 2. LLM function calling 兜底
    try:
        from function_tools.symptom_tool import SYMPTOM_TOOL_SCHEMA, symptom_id_to_query
        from tools.llm_tool import chat_with_tools
        from langchain_core.messages import HumanMessage
        resp = chat_with_tools(
            [HumanMessage(content=text)],
            [SYMPTOM_TOOL_SCHEMA],
            model=get_small_model_name(),
        )
        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            tc = tool_calls[0]
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            query = symptom_id_to_query(args)
            if query:
                logger.info(f"[Repair] LLM modify query: {query}")
                return ExtractResult("done", query, extra={"trace": {
                    "node": "sop.symptom",
                    "in": {"text": text[:40]},
                    "out": {"query": query, "via": "llm",
                            "symptom_id": (args or {}).get("symptom_id")},
                }})
            logger.warning("[Repair] LLM modify query failed!")
    except Exception as e:
        logger.warning("[Repair] symptom tool calling failed: %s", e)
    # 两条路都没命中：也留一条，便于区分「词表没覆盖」与「模型归类错」
    return ExtractResult("fail", reason="no_symptom", extra={"trace": {
        "node": "sop.symptom", "status": "miss", "in": {"text": text[:40]},
    }})


def _search_repair(slots: dict):
    """把症状归一成检索 query；不在这里检索，也不在这里生成。"""
    query = str(slots.get("symptom") or "").strip()
    return {"query": query, "domain": "repair"}


def _repair_handoff(ctx: dict):
    """症状已确认 → 交给主链 retrieve（domain=repair）→ generate。"""
    query = str(ctx.get("query") or "").strip()
    domain = str(ctx.get("domain") or "repair").strip() or "repair"
    if not query:
        return None
    return Handoff(target="retrieve", query=query, domain=domain)


# 槽位形如 {"symptom": xxx}（检索用的标准说法）；action 只返回 query/domain，生成交给主链。
REPAIR_SOP = {
    "id": "repair",
    "name": "故障排查",
    "trigger": REPAIR_TRIGGER,
    "enter_confirm": "always",
    # 只说「维修/故障/坏了」这类泛问时，先进「进入确认门」问一句（与选购 SOP 同一机制）
    "weak_trigger": REPAIR_WEAK_TRIGGER,
    "guards": [
        {"check": is_aftersales},
    ],
    "intro": "我来帮您排查一下故障～",
    "steps": [
        {
            "id": "ask_symptom",
            "type": "ask",
            "slot": "symptom",
            "ask": "能具体描述一下故障现象吗？比如「不走了」「漏水了」「有异响」这样～",
            "retry": "没太听清故障现象，能再说具体点吗？比如「机器人不动了」「拖地漏水」～",
            "extract": _extract_symptom,
        },
        {
            "id": "do_search",
            "type": "action",
            "action": _search_repair,
        },
        {
            "id": "reply",
            "type": "reply",
            "handoff": _repair_handoff,
            "fallback": REPAIR_NO_SYMPTOM_REPLY,
        },
    ],
}


register(REPAIR_SOP)
