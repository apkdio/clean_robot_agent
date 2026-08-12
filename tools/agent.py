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

def ask_stream(query: str):
    """Streaming version of ask() — yields answer chunks as LLM generates them.

    In retrieval_only mode, yields the full chunk text at once.
    In RAG mode, yields partial answer tokens from the LLM.
    """
    hr = _get_retriever()
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
        "简要使用中文回答，注意分行，可以参考多个资料进行总结。如果参考资料全部与问题无关，不要展开，只回复「知识库暂无相关信息，请联系官方售后支持~」。"
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
