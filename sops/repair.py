"""故障排查 SOP：问现象 → 检索 → LLM 生成排查步骤。"""

from sops.base import register

# 现象关键词 → 标准检索 query（查询改写，与知识库条目标题对齐）
_SYMPTOM_MAP = {
    "不动": "机器人不移动怎么办",
    "不走了": "机器人不移动怎么办",
    "趴窝": "机器人不移动怎么办",
    "卡住": "机器人不移动怎么办",
    "漏水": "水箱漏水怎么办",
    "漏水了": "水箱漏水怎么办",
    "异响": "扫地机器人异响怎么办",
    "噪音": "扫地机器人异响怎么办",
    "不充电": "机器人充不进电怎么办",
    "充不进电": "机器人充不进电怎么办",
    "找不到充电座": "机器人找不到充电座怎么办",
    "回不了充": "机器人找不到充电座怎么办",
    "连不上": "APP无法连接机器人怎么办",
    "联网失败": "APP无法连接机器人怎么办",
    "吸力": "吸力下降怎么办",
    "建图": "建图不完整怎么办",
    "地图": "建图不完整怎么办",
    "不干净": "清扫不干净怎么办",
    "拖不干净": "清扫不干净怎么办",
    "异味": "拖布有异味怎么办",
    "发臭": "拖布有异味怎么办",
    "水痕": "拖地后地面有明显水痕",
    "水印": "拖地后地面有明显水痕",
}


# 故障现象兜底用的小模型（3b 更快；精度不够可切回 "qwen2.5:7b"）
_SYMPTOM_TOOL_MODEL = "qwen2.5:3b"


def _extract_symptom(text: str, slots: dict):
    """从用户描述提取故障现象，映射成标准检索 query。

    规则（关键词）优先；规则 miss 时用 3b function calling 兜底归类口语故障。
    """
    # 1. 关键词规则
    for kw, query in _SYMPTOM_MAP.items():
        if kw in text:
            return query

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
                return query
    except Exception as e:
        from tools.log_tool import get_logger
        get_logger(name="repair").warning("[Repair] symptom tool calling failed: %s", e)
    return None


def _search_and_generate(slots: dict):
    """按现象检索故障条目，交给 LLM 生成排查步骤。"""
    query = slots.get("symptom")
    if not query:
        return {"answer": "抱歉，没识别出具体故障现象，你可以换个说法再描述一下～"}

    from tools.hybrid_retriever import HybridRetriever
    from sops.base import DOMAIN_MAP
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
    "trigger": ["故障", "坏了", "不动", "漏水", "异响", "不充电", "异常", "失灵",
                "不好使", "出问题", "趴窝", "卡住", "噪音", "水痕", "水印"],
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
