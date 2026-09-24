"""知识库检索工具（LLM 工具编排用）。

与 `model_tool.find_models`（结构化直出）互补：这里返回**原文片段**，由生成环节组织成答案。
召回仍走既有的 HybridRetriever（双路 + RRF + 精排 + 阈值 + 域定向二次召回），工具不另开一套
——两套索引/阈值会各说各话。

设计见 notes/OPTIMIZATION_ROADMAP.md 的 P1-5 §10：模型只负责「选工具 + 选域 + 给检索式」。

`query_summarization` 是**改写的落点**：编排上线后改写搭检索这趟车（同一次调用里出结果），
所以不再有独立的改写工具 / 改写层；编排关闭时改写仍由 agent 侧的规则层 `_rewrite_query` 负责。
"""

from __future__ import annotations

from typing import Dict, List

from config.word_dict_config import DOMAIN_FILES


SEARCH_KB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": (
            "检索知识库原文（品牌介绍、选购、售后、维修、保养类的事实性问题）。"
            "检索式要写成**不依赖上文也看得懂**的问句：用户用「它」「这个」指代时，"
            "结合上下文换成明确对象（如「净界 P2 的滤网怎么清洗」）。"
            "不确定属于哪个知识域时不要填 domain。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户当前这一句的**原话**"},
                "query_summarization": {
                    "type": "string",
                    "description": "改写成自足问句（指代替换、省略补全）；本来已完整就照抄原话",
                },
                "domain": {
                    "type": "string",
                    "enum": list(DOMAIN_FILES.keys()),
                    "description": "可选：确信属于哪个知识域时才填，不填则全库检索",
                },
            },
            "required": ["query"],
        },
    },
}


def retrieval_query_of(args: Dict) -> str:
    """取真正用于检索的问句：改写句优先，没有就用原话。"""
    args = args or {}
    return str(args.get("query_summarization") or "").strip() or str(args.get("query") or "").strip()


def search_kb_by_args(args: Dict, retriever) -> List:
    """执行 search_kb，返回召回的 Document 列表（与 agent 既有链路同构）。

    `retriever` 由调用方注入：Agent 侧复用它那份 HybridRetriever 单例。工具内部**不自己建实例**
    ——带索引的检索器多一份会翻倍内存，还会与热更新后的索引不同步。

    domain 是**软约束**（与 agent 侧域路由同构）：域内精排 top1 低于 `rerank.score_threshold` 时，
    检索器会自己撤掉 filter 再跑一遍全库并合并两池（见 `hybrid_retriever.search` 的决策记录）。
    所以填了 domain 也可能返回其他域的片段——这是设计行为，不要把它“修”成硬过滤。
    domain 不在白名单里（模型编的）时按全库检索，不抛异常：漏召回比报错更好降级。
    """
    from config.word_dict_config import domain_filter

    query = retrieval_query_of(args)
    if not query:
        return []
    domain = str((args or {}).get("domain") or "").strip()
    where = {"file_name": domain_filter(domain)} if domain in DOMAIN_FILES else None
    return retriever.search(query, filter=where)
