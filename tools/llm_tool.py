"""LLM and embedding helpers backed by a local Ollama instance.

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1,
so we reuse the already-installed `langchain_openai` package (ChatOpenAI /
OpenAIEmbeddings) and point it at Ollama. This keeps the door open to swap
in any other OpenAI-compatible provider (cloud or self-hosted) by simply
changing base_url / api_key / model in the config.
"""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from log_tool import get_logger

logger = get_logger(name="llm_tool")

# Ollama default connection settings (local, no real API key needed)
_DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
_DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "ollama")
_DEFAULT_CHAT_MODEL = os.environ.get("LLM_CHAT_MODEL", "qwen3:1.7b")
_DEFAULT_EMBED_MODEL = os.environ.get("LLM_EMBED_MODEL", "qwen3-embedding:0.6b")


def get_chat_model(
    model: str = _DEFAULT_CHAT_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
    api_key: str = _DEFAULT_API_KEY,
    temperature: float = 0.3,
    **kwargs,
) -> ChatOpenAI:
    """Return a ChatOpenAI instance wired to the local Ollama server.

    Parameters can be overridden via constructor args or environment variables
    (LLM_BASE_URL / LLM_API_KEY / LLM_CHAT_MODEL).
    """
    logger.info(f"[Chat] init model={model} base_url={base_url}")
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


def get_embedding_model(
    model: str = _DEFAULT_EMBED_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
    api_key: str = _DEFAULT_API_KEY,
    **kwargs,
) -> OpenAIEmbeddings:
    """Return an OpenAIEmbeddings instance wired to the local Ollama server.

    Parameters can be overridden via constructor args or environment variables
    (LLM_BASE_URL / LLM_API_KEY / LLM_EMBED_MODEL).
    """
    logger.info(f"[Embed] init model={model} base_url={base_url}")
    return OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        # Ollama's embedding endpoint expects raw text, not token IDs.
        # Disabling length-checked tokenization sends the text as-is.
        check_embedding_ctx_length=False,
        **kwargs,
    )


def simple_chat(query: str, system: str = "") -> str:
    """One-shot helper: send a single user query and return the text reply.

    Useful as a quick smoke-test against the running Ollama instance.
    """
    llm = get_chat_model()
    messages = []
    if system:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages.append(SystemMessage(content=system))
    else:
        from langchain_core.messages import HumanMessage
    messages.append(HumanMessage(content=query))
    try:
        resp = llm.invoke(messages)
        logger.info(f"[Chat] replied, len={len(resp.content)}")
        return resp.content
    except Exception as e:
        logger.error(f"[Chat] invoke failed: {e}")
        return ""


if __name__ == "__main__":
    # Smoke test: ask a cleaning-robot question
    print("=== Chat smoke test ===")
    answer = simple_chat("扫地机器人吸力下降怎么办？用一句话回答。")
    print(answer)

    print("\n=== Embedding smoke test ===")
    emb = get_embedding_model()
    vec = emb.embed_query("扫地机器人不回充")
    print(f"embedding dim={len(vec)}, first5={vec[:5]}")
