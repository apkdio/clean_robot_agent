"""运行状态与展示配置：给 Controller 提供只读视图。"""
from __future__ import annotations

from utils.config_tool import get_config


def agent_display_config() -> dict:
    """当前问答模式与模型名，供前端标识。"""
    agent_cfg = get_config("agent") or {}
    behavior = agent_cfg.get("behavior", {})
    retrieval_only = behavior.get("retrieval_only", True)
    return {
        "status": "ok",
        "mode": "retrieval_only" if retrieval_only else "rag_full",
        "mode_label": "RAG 检索模式" if retrieval_only else "RAG+LLM 模式",
        "model": agent_cfg.get("llm", {}).get("model", "?"),
    }
