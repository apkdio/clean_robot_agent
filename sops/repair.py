"""故障排查 SOP：问现象 → 检索 → LLM 生成排查步骤。"""

from sops.base import register, is_aftersales
from config.word_dict_config import SYMPTOM_MAP, REPAIR_TRIGGER

from tools.llm_tool import get_small_model_name
from tools.log_tool import get_logger

# 故障现象兜底用的小模型（默认 3b 更快；精度不够可在 agent.yaml 调 llm.small_model）
_SYMPTOM_TOOL_MODEL = get_small_model_name()

logger = get_logger(name="repair_sop")

def _extract_symptom(text: str, slots: dict):
    """从用户描述提取故障现象，映射成标准检索 query。

    规则（关键词）优先；规则 miss 时用 3b function calling 兜底归类口语故障。
    """
    # 1. 关键词规则
    for kw, query in SYMPTOM_MAP.items():
        if kw in text:
            logger.info(f"[Repair] symptom {kw}: {query}")
            return query
    logger.warning("[Repair] Keyword match failed! Using LLM to match!")

    # 2. LLM function calling 兜底
    try:
        from function_tools.symptom_tool import SYMPTOM_TOOL_SCHEMA, symptom_id_to_query
        from tools.llm_tool import chat_with_tools
        from langchain_core.messages import HumanMessage
        resp = chat_with_tools(
            [HumanMessage(content=text)],
            [SYMPTOM_TOOL_SCHEMA],
            model=_SYMPTOM_TOOL_MODEL,
        )
        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            tc = tool_calls[0]
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            query = symptom_id_to_query(args)
            if query:
                logger.info(f"[Repair] LLM modify query: {query}")
                return query
            logger.warning("[Repair] LLM modify query failed!")
    except Exception as e:
        logger.warning("[Repair] symptom tool calling failed: %s", e)
    return None


def _search_and_generate(slots: dict):
    """按现象检索故障条目，交给 LLM 生成排查步骤。"""
    query = str(slots.get("symptom"))
    if not query:
        return {"answer": "抱歉，没识别出具体故障现象，你可以换个说法再描述一下～"}

    from tools.hybrid_retriever import HybridRetriever
    from config.word_dict_config import DOMAIN_MAP
    hr = HybridRetriever()
    hr.ensure_sparse_index()
    # 定向故障排除域，避免"吸力下降"等被选购域的"吸力参数"带偏
    chunks = hr.search(query, filter={"file_name": DOMAIN_MAP["repair"]})
    if not chunks:
        return {"answer": "知识库暂时没有收录这个故障的排查方案，建议联系官方售后进一步咨询～"}

    context = "\n\n".join(c.page_content for c in chunks[:3])

    from langchain_core.messages import HumanMessage, SystemMessage
    from tools.llm_tool import get_chat_model
    from tools.prompts_tool import load_main_prompts

    user_msg = (
        f"参考资料：\n{context}\n\n"
        f"用户故障现象：{query}\n\n"
        f"请基于参考资料，用简洁的分步格式给出故障排查步骤（先检测后修复，给出具体操作）。"
    )
    llm = get_chat_model(temperature=0.3)
    resp = llm.invoke([SystemMessage(content=load_main_prompts()), HumanMessage(content=user_msg)])
    return {"answer": (resp.content or "").strip()}


REPAIR_SOP = {
    "id": "repair",
    "trigger": REPAIR_TRIGGER,
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
            "action": _search_and_generate,
        },
        {
            "id": "reply",
            "type": "reply",
            "template": "{answer}",
            "fallback": "抱歉，出了一点小问题，请重新提问～",
        },
    ],
}


register(REPAIR_SOP)
