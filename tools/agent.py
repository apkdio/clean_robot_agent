"""RAG 编排层：扫地机器人知识库的端到端问答流水线。

流程：
  用户提问
    → intent_router.route_intent()              # 本地分类头：robot/casual/other/unknown
    → hybrid_retriever.search()                 # 稠密 + 稀疏 → RRF 融合
    → 结构化预算直出 或 RAG 生成                  # 输出答案
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from config_tool import load_agent_config
from llm_tool import stream_chat
from log_tool import get_logger
from prompts_tool import load_main_prompts

logger = get_logger(name="agent")

_agent_cfg = load_agent_config()
_llm_cfg = _agent_cfg.get("llm", {})
_behavior = _agent_cfg.get("behavior", {})

_hybrid_retriever = None  # 懒加载单例


def _get_retriever():
    """懒加载双路召回器（稠密 + 稀疏 → RRF）。"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from hybrid_retriever import HybridRetriever
        _hybrid_retriever = HybridRetriever()
        _hybrid_retriever.ensure_sparse_index()
    return _hybrid_retriever


_TIME_HINT_RE = re.compile(
    r"最近|近[一二两三四五六七八九十0-9]|今年|去年|前年|半年|个月内|月内|周内|天内|年内"
    r"|发布|上市|新品|\d{4}\s*年|[一二三四五六七八九十0-9]+\s*月"
)


def _resolve_date_filter(query: str) -> dict | None:
    """把问题里的日期表达（绝对或相对）解析成 Chroma 过滤条件。

    先走确定性规则解析（快且可靠），规则未命中时再回退到 LLM function calling。
    """
    from function_tools.date_tool import (
        parse_date,
        calc_date_range,
        build_date_filter,
        DATE_TOOL_SCHEMA,
    )

    # 1. 确定性规则（先绝对后相对）
    r = parse_date(query)
    if r:
        logger.info("[Agent] Date resolved via rule: %s", r)
        return build_date_filter(r[0], r[1])

    # 2. LLM function calling 兜底
    if not _TIME_HINT_RE.search(query):
        return None
    try:
        from llm_tool import chat_with_tools
        resp = chat_with_tools(
            [HumanMessage(content=query)],
            [DATE_TOOL_SCHEMA],
            model=_llm_cfg.get("model", "qwen2.5:7b"),
        )
        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            tc = tool_calls[0]
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            result = calc_date_range(args.get("expression", ""))
            if "error" not in result:
                logger.info("[Agent] Date resolved via LLM tool: %s", result)
                return build_date_filter(result["start_date"], result["end_date"])
    except Exception as e:
        logger.warning("[Agent] Date tool calling failed: %s", e)
    return None


def ask_stream(query: str):
    """流式问答入口 —— 经本地分类头做意图路由。

    other → 礼貌拒答；casual → 闲聊；
    unknown → 软引导 + RAG；robot → 结构化预算直出 或 RAG。
    """
    from intent_router import route_intent, get_guess_hint
    intent = route_intent(query)

    # 领域外：礼貌拒答
    if intent == "other":
        yield "抱歉，我是扫地机器人专属助手，对这方面不太了解哦～你可以问我扫地机器人的选购、故障排查、使用维护等问题。"
        return

    # 闲聊问候：自然回应，跳过检索
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

    # 意图模糊：先给软引导语，再走 RAG
    if intent == "unknown":
        yield get_guess_hint() + "\n\n"

    hr = _get_retriever()
    from metadata_extractor import build_filter
    metadata_filter = build_filter(query)

    # 解析日期表达（"最近半年"/"2025年三月"）→ 日期过滤
    date_filter = _resolve_date_filter(query)
    filter_kind = "budget" if metadata_filter is not None else ""
    if metadata_filter is not None and date_filter is not None:
        metadata_filter = {"$and": [metadata_filter, date_filter]}
        filter_kind = "budget+date"
    elif date_filter is not None:
        metadata_filter = date_filter
        filter_kind = "date"

    # 结构化查询：按 metadata 枚举所有匹配型号（绕过 top-k，避免漏掉条目）
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
            prefix = {
                "budget": f"在您预算内的机器人有 {len(models)} 款：",
                "date": f"该时间段内发布的机器人有 {len(models)} 款：",
                "budget+date": f"符合您预算和时间要求的机器人有 {len(models)} 款：",
            }.get(filter_kind, f"符合条件的机器人有 {len(models)} 款：")
            yield prefix + "\n\n" + "\n".join(lines)
            return
        # 没有匹配型号 → 回退到普通 RAG
        logger.info("[Agent] Structured filter matched no models, falling back to RAG")

    # 普通 RAG：双路召回
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
