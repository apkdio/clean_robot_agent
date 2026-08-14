"""RAG agent: end-to-end Q&A pipeline for the cleaning-robot knowledge base.

Flow:
  user query
    → intent_router.route_intent()              # local classifier head: robot/casual/other/unknown
    → hybrid_retriever.search()                 # dense + sparse → RRF fusion
    → structured budget output OR RAG            # generate
"""

from langchain_core.messages import HumanMessage, SystemMessage

from config_tool import load_agent_config
from llm_tool import stream_chat
from log_tool import get_logger
from prompts_tool import load_main_prompts

logger = get_logger(name="agent")

_agent_cfg = load_agent_config()
_llm_cfg = _agent_cfg.get("llm", {})
_behavior = _agent_cfg.get("behavior", {})

_hybrid_retriever = None  # lazy singleton


def _get_retriever():
    """Lazy-init the hybrid retriever (dense + sparse → RRF)."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from hybrid_retriever import HybridRetriever
        _hybrid_retriever = HybridRetriever()
        _hybrid_retriever.ensure_sparse_index()
    return _hybrid_retriever


def ask_stream(query: str):
    """Streaming version of ask() — intent-routed via local classifier head.

    other → polite decline; casual → small talk;
    unknown → soft hint + RAG; robot → structured budget output or RAG.
    """
    from intent_router import route_intent, get_guess_hint
    intent = route_intent(query)

    # Out-of-domain: polite decline
    if intent == "other":
        yield "抱歉，我是扫地机器人专属助手，对这方面不太了解哦～你可以问我扫地机器人的选购、故障排查、使用维护等问题。"
        return

    # Casual greetings / small talk: answer naturally, skip retrieval
    if intent == "casual":
        for chunk in stream_chat(
            [
                SystemMessage(content=load_main_prompts()),
                HumanMessage(content=query),
            ],
            model=_llm_cfg.get("model", "qwen2.5:7b"),
            temperature=_llm_cfg.get("temperature", 0.3),
        ):
            yield chunk
        return

    # Unknown intent: prefix a soft hint, then answer via RAG
    if intent == "unknown":
        yield get_guess_hint() + "\n\n"

    hr = _get_retriever()
    from metadata_extractor import build_filter
    metadata_filter = build_filter(query)

    # Structured query (budget): enumerate ALL matching models via metadata,
    # bypassing top-k so we don't drop any in-budget item.
    if metadata_filter is not None:
        from vector_store import search_by_filter
        from metadata_extractor import extract_model_info, format_model_line
        filtered = search_by_filter(metadata_filter)
        models, seen = [], set()
        for c in filtered:
            info = extract_model_info(c)
            if info.get("price") is not None and info.get("name") and info["name"] not in seen:
                seen.add(info["name"])
                models.append(info)
        if models:
            lines = [format_model_line(m) for m in models]
            yield f"在您预算内的机器人有 {len(models)} 款：\n\n" + "\n".join(lines)
            return
        # No in-budget models found → fall through to normal RAG
        logger.info("[Agent] Budget filter matched no models, falling back to RAG")

    # Normal RAG: dual-route retrieval
    chunks = hr.search(query)

    if not chunks:
        for chunk in stream_chat(
            [
                SystemMessage(content=load_main_prompts()),
                HumanMessage(content=f"知识库中暂无相关内容，请简短回答：{query}"),
            ],
            model=_llm_cfg.get("model", "qwen2.5:7b"),
            temperature=_llm_cfg.get("temperature", 0.3),
        ):
            yield chunk
        return

    if _behavior.get("retrieval_only", False):
        top = chunks[0]
        src = top.metadata.get("file_name", "")
        yield f"📄 来源：{src}\n\n{top.page_content}"
        return

    chunk_texts = [c.page_content for c in chunks]
    context_block = "\n\n".join(chunk_texts)
    user_message = (
        "参考资料：\n" + context_block + "\n\n"
        "问题：" + query + "\n\n"
        "回答规则：\n"
        "1. 若用户是推荐/选购类问题（如「推荐几款」「有什么机器人」「预算XX」「买哪个」），"
        "必须逐条列出参考资料中所有符合条件（价格在预算内）的型号，至少 3 条，能列 4-5 条更好，"
        "每个型号单独一行，格式为「型号名：吸力、导航、避障等关键参数，参考价 XX 元」。"
        "严禁只介绍一个型号。\n"
        "2. 若是一般事实性问题，简要使用中文回答，注意分行。\n"
        "3. 如果参考资料与问题无关：先判断是否打招呼/闲聊，是则按系统提示词「闲聊与问候」自然回应；"
        "若是扫地机器人相关事实性问题，从系统提示词「暂无信息回复」列表随机选一句。"
    )

    for chunk in stream_chat(
        [
            SystemMessage(content=load_main_prompts()),
            HumanMessage(content=user_message),
        ],
        model=_llm_cfg.get("model", "qwen2.5:7b"),
        temperature=_llm_cfg.get("temperature", 0.3),
    ):
        yield chunk
