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
_DEFAULT_CHAT_MODEL = os.environ.get("LLM_CHAT_MODEL", "qwen2.5:7b")
_DEFAULT_EMBED_MODEL = os.environ.get("LLM_EMBED_MODEL", "bge-m3")


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
    logger.info(f"[Chat] init model={model}")
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
    logger.info(f"[Embed] init model={model}")
    return OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        # Ollama's embedding endpoint expects raw text, not token IDs.
        # Disabling length-checked tokenization sends the text as-is.
        check_embedding_ctx_length=False,
        **kwargs,
    )


def stream_chat(messages: list, model: str = "", temperature: float = 0.3) -> any:
    """Streaming chat: generator that yields text chunks as they arrive.

    Each yield is a plain string chunk (partial answer text).
    For thinking-capable models (qwen3, deepseek-r1), set model_kwargs
    with `extra_body={"reasoning": True}` to include reasoning tokens.
    """
    m = model or _DEFAULT_CHAT_MODEL
    llm = get_chat_model(model=m, temperature=temperature, streaming=True)
    logger.info(f"[Stream] start model={m}")
    try:
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error(f"[Stream] failed: {e}")
        yield ""
