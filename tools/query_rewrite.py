"""上下文承接与 query 改写：历史挑选、规则拼接、LLM 改写与校验。

只影响检索 query，会话记录仍保留用户原话。
"""
from __future__ import annotations

import re
from typing import Literal

from tools.config_tool import load_config
from tools.log_tool import get_logger

logger = get_logger(name="query_rewrite")

_cfg_cache = None


def cfg() -> dict:
    """context.yaml（读失败返回空 dict，走内置默认）。"""
    global _cfg_cache
    if _cfg_cache is None:
        try:
            _cfg_cache = load_config("context") or {}
        except Exception as e:  # noqa: BLE001
            logger.warning("[Context] load config failed: %s", e)
            _cfg_cache = {}
    return _cfg_cache


def enabled() -> bool:
    return bool((cfg().get("context") or {}).get("enabled", True))


def rewrite_cfg() -> dict:
    return (cfg().get("context") or {}).get("rewrite") or {}


def normalize_enabled() -> bool:
    """query 入口归一化总开关（agent.yaml behavior.query_normalize）。"""
    from tools.config_tool import load_config

    behavior = (load_config("agent").get("behavior") or {})
    return bool(behavior.get("query_normalize", True))


def history_block(session_id: str) -> str:
    """拼最近对话历史，供 LLM 自主消解指代。"""
    from tools.context_store import get_recent

    msgs = get_recent(session_id)
    if msgs and msgs[-1].get("role") == "user":
        msgs = msgs[:-1]
    return "\n".join(
        f"{'用户' if m.get('role') == 'user' else '客服'}：{(m.get('content') or '')[:200]}"
        for m in msgs
        if not m.get("blocked")
    )


def last_user_message(session_id: str) -> str:
    """最近一条 user 消息的原文（没有则空串）。"""
    from tools.context_store import get_recent

    for m in reversed(get_recent(session_id)):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def window_messages(session_id: str) -> list:
    """最近 topic_window 条会话记录。"""
    from tools.context_store import get_recent

    n = int((cfg().get("context") or {}).get("topic_window", 8) or 8)
    return get_recent(session_id, n=n)


def window_domain(session_id: str) -> str | None:
    """窗口内最近一条能路由到知识域的 user 消息所属域。"""
    from tools.domain_router import route_domain

    for m in reversed(window_messages(session_id)):
        if m.get("role") == "user" and not m.get("blocked"):
            domain = route_domain(m.get("content") or "")
            if domain:
                return domain
    return None


def _cosine(a, b) -> float:
    import numpy as np

    va, vb = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / denom) if denom else 0.0


def _select_history_turns(session_id: str, query: str) -> list:
    """挑出与当前 query 最相关的历史用户话术（过了相似度下限的）。"""
    conf = rewrite_cfg()
    top_k = int(conf.get("max_history_turns", 2) or 2)
    min_sim = float(conf.get("min_similarity", 0.55) or 0)

    msgs = window_messages(session_id)
    if msgs and msgs[-1].get("role") == "user":
        msgs = msgs[:-1]
    users = [
        m.get("content") or ""
        for m in msgs
        if m.get("role") == "user" and not m.get("blocked") and (m.get("content") or "").strip()
    ]
    if not users:
        return []

    try:
        from tools.llm_tool import get_embedding_model

        vecs = get_embedding_model().embed_documents([query] + users)
    except Exception as e:  # noqa: BLE001
        logger.warning("[Rewrite] embedding failed, skip turn selection: %s", e)
        return []

    scored = sorted(((_cosine(vecs[0], v), t) for t, v in zip(users, vecs[1:])), reverse=True)
    picked = [t for sim, t in scored[:top_k] if sim >= min_sim]
    logger.info(
        "[Rewrite] history turns: picked=%d, top_sim=%.3f, threshold=%.2f",
        len(picked), scored[0][0] if scored else 0.0, min_sim,
    )
    return picked


def _validate_rewrite(text: str, query: str, grounding: str, max_chars: int) -> str | None:
    """校验改写结果；不合格返回 None（宁可不改，也不要把 query 改坏）。"""
    text = (text or "").strip().strip("\"'“”「」《》 \n")
    if not text or len(text) > max_chars:
        return None
    pattern = r"[\u4e00-\u9fffA-Za-z0-9]"
    in_query = set(re.findall(pattern, query)) & set(re.findall(pattern, text))
    in_grounding = set(re.findall(pattern, grounding or "")) & set(re.findall(pattern, text))
    return text if (in_query or in_grounding) else None


def _llm_rewrite(query: str, history: str) -> str | None:
    """用小模型把追问改写成自足 query（输出受约束，校验不过即放弃）。"""
    from langchain_core.messages import HumanMessage
    from tools.llm_tool import get_small_model_name, stream_chat

    max_chars = int(rewrite_cfg().get("max_chars", 80) or 80)
    prompt = (
        "下面是一段客服对话历史，以及用户当前这一句。\n"
        "请把当前这一句改写成一个不依赖上文、单独也看得懂的问题："
        "把「这个 / 那个 / 它」这类指代替换成历史里明确提到的对象，补全省略的信息。\n"
        f"要求：只输出改写后的问题本身（不要解释、不要引号、不要加粗），不超过 {max_chars} 字；"
        "如果当前这一句本身已经完整，就原样输出。\n\n"
        f"对话历史：\n{history or '（无）'}\n\n"
        f"用户当前这一句：{query}"
    )
    try:
        out = "".join(stream_chat([HumanMessage(content=prompt)],
                                  model=get_small_model_name(), temperature=0.0))
    except Exception as e:  # noqa: BLE001
        logger.warning("[Rewrite] llm rewrite failed: %s", e)
        return None
    return _validate_rewrite(out, query, history, max_chars)


def rewrite_query(session_id: str, query: str) -> tuple[str, Literal["none"]] | tuple[str, Literal["rule"]] | tuple[
        str | None, Literal["llm"]]:
    """把依赖上文的追问改写成自足 query，返回 (effective_query, via)。"""
    conf = rewrite_cfg()
    if not enabled() or not conf.get("enabled", True):
        return query, "none"

    max_chars = int(conf.get("max_chars", 80) or 80)
    turns = _select_history_turns(session_id, query)
    if turns:
        budget = max(0, max_chars - len(query) - 1)
        hist = " ".join(t[:40] for t in turns)[:budget].strip()
        if hist:
            effective = f"{hist} {query}"
            logger.info("[Rewrite] via=rule origin=%s → effective=%s", query[:40], effective[:80])
            return effective, "rule"

    if str(conf.get("mode", "rule")).lower() == "rule+llm":
        history = history_block(session_id)
        result = _llm_rewrite(query, history)
        if result and result != query:
            logger.info("[Rewrite] via=llm origin=%s → effective=%s", query[:40], result[:80])
            return result, "llm"

    return query, "none"
