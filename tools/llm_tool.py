"""基于本地 Ollama 的 LLM / embedding 工厂。

Ollama 暴露了 OpenAI 兼容端点 http://localhost:11434/v1，因此直接复用已安装的
`langchain_openai` 包（ChatOpenAI / OpenAIEmbeddings），把 base_url 指向 Ollama。
这样只需改配置里的 base_url / api_key / model，就能无缝切换到任意其他
OpenAI 兼容服务商（云端或自建）。
"""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from log_tool import get_logger

logger = get_logger(name="llm_tool")

# Ollama 默认连接配置（本地运行，无需真实 API key）
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
    """返回一个连接到本地 Ollama 的 ChatOpenAI 实例。

    参数可通过构造函数入参或环境变量（LLM_BASE_URL / LLM_API_KEY / LLM_CHAT_MODEL）覆盖。
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
    """返回一个连接到本地 Ollama 的 OpenAIEmbeddings 实例。

    参数可通过构造函数入参或环境变量（LLM_BASE_URL / LLM_API_KEY / LLM_EMBED_MODEL）覆盖。
    """
    logger.info(f"[Embed] init model={model}")
    return OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        # Ollama 的 embedding 端点期望原始文本而非 token id，
        # 关闭长度检查的 tokenize，把文本原样发送。
        check_embedding_ctx_length=False,
        **kwargs,
    )


def stream_chat(messages: list, model: str = "", temperature: float = 0.3) -> any:
    """流式对话：生成器，随生成过程逐段产出文本块。

    每次 yield 一个纯字符串块（部分回答文本）。
    对支持思考过程的模型（qwen3、deepseek-r1），可设 model_kwargs
    为 `extra_body={"reasoning": True}` 来包含推理 token。
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


def chat_with_tools(messages: list, tools: list, model: str = "", temperature: float = 0.1):
    """带工具定义的 LLM 调用（function calling）。

    返回 AIMessage；通过 `.tool_calls` 查看模型决定调用的工具。
    `tools` 是一组 OpenAI 风格的 function schema（{"type": "function", "function": {...}}）。
    """

    m = model or _DEFAULT_CHAT_MODEL
    llm = get_chat_model(model=m, temperature=temperature)
    llm_tools = llm.bind_tools(tools)
    logger.info("[ToolChat] start model=%s tools=%s", m, [t["function"]["name"] for t in tools])
    try:
        return llm_tools.invoke(messages)
    except Exception as e:
        logger.error(f"[ToolChat] failed: {e}")
        # 返回一个不带 tool_calls 的空 AIMessage，让调用方走兜底逻辑。
        from langchain_core.messages import AIMessage
        return AIMessage(content="")
