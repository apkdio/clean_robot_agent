"""RAG agent: end-to-end Q&A pipeline for the cleaning-robot knowledge base.

Flow:
  user query
    → hybrid_retriever.search(query)          # dense + sparse → RRF fusion
    → rag_summarize prompt + chunks            # fill template
    → main_prompt (system) + user message      # build messages
    → LLM generate                              # Ollama qwen2.5:3b
    → answer
"""

from langchain_core.messages import HumanMessage, SystemMessage

from config_tool import load_agent_config
from llm_tool import get_chat_model, stream_chat
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


# 常见打招呼/闲聊模式，命中则跳过检索，直接自然回应
_GREETING_PATTERNS = [
    "你好", "您好","你好啊", "嗨", "哈喽", "hello", "hi", "在吗", "在不在",
    "谢谢", "感谢", "辛苦了", "再见", "拜拜", "晚安", "早上好", "中午好", "晚上好",
    "你是谁", "你叫什么", "你能做什么", "你会什么", "介绍一下你自己",
]


def _is_casual_talk(query: str) -> bool:
    """Detect greetings / thanks / self-intro queries that need no retrieval."""
    q = query.strip().lower()
    return any(p in q for p in _GREETING_PATTERNS)


def ask_stream(query: str):
    """Streaming version of ask() — yields answer chunks as LLM generates them.

    In retrieval_only mode, yields the full chunk text at once.
    In RAG mode, yields partial answer tokens from the LLM.
    """
    # Greetings / small talk: skip retrieval, answer naturally
    if _is_casual_talk(query):
        for chunk in stream_chat(
            [
                SystemMessage(content=load_main_prompts()),
                HumanMessage(content=query),
            ],
            model=_llm_cfg.get("model", "qwen2.5:3b"),
            temperature=_llm_cfg.get("temperature", 0.3),
        ):
            yield chunk
        return

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
            model=_llm_cfg.get("model", "qwen2.5:3b"),
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
        model=_llm_cfg.get("model", "qwen2.5:3b"),
        temperature=_llm_cfg.get("temperature", 0.3),
    ):
        yield chunk


def _direct_answer(query: str) -> str:
    """Fallback: answer without retrieval (when vector store is empty or no hits)."""
    system_prompt = load_main_prompts()
    llm = get_chat_model(
        model=_llm_cfg.get("model", "qwen3:1.7b"),
        base_url=_llm_cfg.get("base_url", "http://localhost:11434/v1"),
        api_key=_llm_cfg.get("api_key", "ollama"),
        temperature=_llm_cfg.get("temperature", 0.3),
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"知识库中暂无相关内容，请根据你的常识简短回答以下问题：{query}"
            if _behavior.get("rag_enabled", True)
            else query
        ),
    ]
    try:
        return llm.invoke(messages).content.strip()
    except Exception as e:
        logger.error(f"[Agent] Direct answer failed: {e}")
        return "抱歉，模型暂时不可用，请稍后再试。"
