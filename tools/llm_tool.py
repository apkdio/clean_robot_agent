"""基于本地 Ollama 的 LLM / embedding 工厂（复用 langchain_openai，base_url 指向 Ollama）。

chat 模型与端点的单一真源是 config/agent.yaml 的 llm 段（model / small_model /
base_url / api_key）；入参留空时按此解析，配置缺键则回退环境变量与内置默认。
embedding 端点由 config/chroma.yaml 的 embedding 段提供（调用方显式传入）。
"""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config_tool import load_config
from log_tool import get_logger

logger = get_logger(name="llm_tool")

# 兜底默认值：配置项缺失时使用（可先由环境变量覆盖）
_DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
_DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "ollama")
_DEFAULT_CHAT_MODEL = os.environ.get("LLM_CHAT_MODEL", "qwen2.5:7b")
_DEFAULT_SMALL_MODEL = os.environ.get("LLM_SMALL_MODEL", "qwen2.5:3b")
_DEFAULT_EMBED_MODEL = os.environ.get("LLM_EMBED_MODEL", "bge-m3")

_chat_cfg_cache = None


def _chat_cfg() -> dict:
    """延迟读取 agent.yaml 的 llm 段（读盘失败则回退到环境变量/内置默认）。"""
    global _chat_cfg_cache
    if _chat_cfg_cache is None:
        try:
            _chat_cfg_cache = load_config("agent").get("llm", {}) or {}
        except Exception as e:
            logger.warning(f"[Config] load agent llm config failed: {e}")
            _chat_cfg_cache = {}
    return _chat_cfg_cache


def get_chat_model_name() -> str:
    """主生成模型名（agent.yaml llm.model → 环境变量 → 内置默认）。"""
    return _chat_cfg().get("model") or _DEFAULT_CHAT_MODEL


def get_small_model_name() -> str:
    """轻量提取/兜底模型名（agent.yaml llm.small_model → 环境变量 → 内置默认）。

    用于预算/故障分类的 function calling 兜底与会话标题生成：重速度、轻精度。
    """
    return _chat_cfg().get("small_model") or _DEFAULT_SMALL_MODEL


def get_chat_model(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    **kwargs,
) -> ChatOpenAI:
    """返回连接到 OpenAI 兼容服务（默认本地 Ollama）的 ChatOpenAI 实例。

    model / base_url / api_key / max_tokens 留空时按 agent.yaml 的 llm 段解析。
    """
    cfg = _chat_cfg()
    model = model or cfg.get("model") or _DEFAULT_CHAT_MODEL
    base_url = base_url or cfg.get("base_url") or _DEFAULT_BASE_URL
    api_key = api_key or cfg.get("api_key") or _DEFAULT_API_KEY
    if max_tokens is None:
        max_tokens = cfg.get("max_tokens")
    if max_tokens:
        kwargs.setdefault("max_tokens", int(max_tokens))
    logger.info(f"[Chat] init model={model} base_url={base_url} max_tokens={max_tokens or '-'}")
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
    """返回连接到 Ollama 的 OpenAIEmbeddings 实例。

    生产路径由调用方（vector_store）显式传入 chroma.yaml 的 embedding 段；
    入参可用环境变量（LLM_BASE_URL / LLM_API_KEY / LLM_EMBED_MODEL）覆盖。
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
    """流式对话生成器，每次 yield 一个纯字符串文本块。

    对支持思考过程的模型（qwen3、deepseek-r1），可设 model_kwargs 为
    `extra_body={"reasoning": True}` 来包含推理 token。
    """
    m = model or get_chat_model_name()
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
    """带工具定义的 LLM 调用（function calling），返回 AIMessage（`.tool_calls` 查看决定）。"""

    m = model or get_chat_model_name()
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
