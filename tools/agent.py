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
from llm_tool import get_chat_model
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


def ask(query: str) -> str:
    """Answer a user question using hybrid RAG (dense+sparse → RRF → generate).

    Args:
        query: The user's question about cleaning robots.

    Returns:
        The LLM-generated answer string.
    """
    # --- 1. Retrieve (hybrid: dense + sparse → RRF fusion) ---
    hr = _get_retriever()
    chunks = hr.search(query)

    if not chunks:
        logger.warning("[Agent] No chunks retrieved; falling back to direct answer.")
        return _direct_answer(query)

    # --- 2b. Retrieval-only mode: return top chunk directly (no LLM summarization) ---
    # Useful when LLM is too small to follow instructions (e.g. ≤3B models).
    if _behavior.get("retrieval_only", False):
        top = chunks[0]
        src = top.metadata.get("file_name", "")
        return f"📄 来源：{src}\n\n{top.page_content}"

    # --- 3. Build user message with retrieved context ---
    # Concise prompt optimized for small models (≤3B):
    # keep instructions minimal so the model reads the context instead of ignoring it.
    chunk_texts = []
    for c in chunks:
        chunk_texts.append(c.page_content)
    context_block = "\n\n".join(chunk_texts)

    user_message = (
        f"参考资料：\n{context_block}\n\n"
        f"问题：{query}\n\n"
        f"请根据参考资料回答。只输出答案，不要展开其他话题。"
    )

    # --- 4. Call LLM ---
    # Minimal approach for small models: single user message with context inline.
    llm = get_chat_model(
        model=_llm_cfg.get("model", "qwen3:1.7b"),
        base_url=_llm_cfg.get("base_url", "http://localhost:11434/v1"),
        api_key=_llm_cfg.get("api_key", "ollama"),
        temperature=_llm_cfg.get("temperature", 0.3),
    )

    # For small models, the system prompt can overwhelm attention.
    # Use a single concise user message that contains both context and instruction.
    messages = [
        HumanMessage(content=user_message),
    ]

    try:
        resp = llm.invoke(messages)
        answer = resp.content.strip()
        logger.info(f"[Agent] Answer generated, {len(answer)} chars.")
        return answer
    except Exception as e:
        logger.error(f"[Agent] LLM invoke failed: {e}")
        return "抱歉，模型暂时不可用，请稍后再试。"


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
