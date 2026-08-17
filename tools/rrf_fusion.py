"""RRF（倒数排名融合）—— 将异构的排名列表合并为一个。

仅使用排名位置而非原始分数来合并稠密（向量相似度）与稀疏（BM25 关键词）结果列表，
这样不同分数尺度就不会相互干扰。

每个候选文档 d 的公式：
    RRF_score(d) = 对每个检索器 i 求和： w_i / (k + rank_i(d))

其中：
    k     = 平滑常数（通常为 60）
    w_i   = 检索器 i 的权重
    rank_i(d) = d 在检索器 i 中的排名（从 1 开始）
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.documents import Document

from config_tool import load_rag_config
from log_tool import get_logger

logger = get_logger(name="rrf_fusion")

_rag_cfg = load_rag_config()
_rrf_cfg = _rag_cfg.get("rrf", {})
_retrieval_cfg = _rag_cfg.get("retrieval", {})


def _doc_id(doc: Document) -> str:
    """使用 page_content 哈希作为文档身份用于去重。"""
    return doc.page_content


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Document, float]],
    sparse_results: List[Tuple[Document, float]],
    k: int | None = None,
    top_k: int | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
) -> List[Tuple[Document, float, Dict]]:
    """通过 RRF 融合稠密与稀疏检索结果。

    参数：
        dense_results:  [(doc, cosine_score), ...] 降序排列
        sparse_results: [(doc, bm25_score), ...] 降序排列
        k:              RRF 平滑常数（默认来自 rag.yaml）
        top_k:          返回结果数量（默认来自 rag.yaml）
        dense_weight:   稠密通道权重
        sparse_weight:  稀疏通道权重

    返回：
        [(doc, rrf_score, meta), ...]，按 RRF 分数降序排列。
        meta = {"dense_rank": int|None, "sparse_rank": int|None, "rrf_score": float}
    """
    k = k if k is not None else _rrf_cfg.get("k", 60)
    top_k = top_k if top_k is not None else _retrieval_cfg.get("final_top_k", 3)
    dense_weight = dense_weight if dense_weight is not None else _rrf_cfg.get("dense_weight", 1.0)
    sparse_weight = sparse_weight if sparse_weight is not None else _rrf_cfg.get("sparse_weight", 1.0)

    # 构建 {doc_id → rank} 映射
    dense_ranks: Dict[str, int] = {}
    for rank, (doc, _) in enumerate(dense_results, start=1):
        dense_ranks[_doc_id(doc)] = rank

    sparse_ranks: Dict[str, int] = {}
    for rank, (doc, _) in enumerate(sparse_results, start=1):
        sparse_ranks[_doc_id(doc)] = rank

    # 收集所有去重后的候选
    all_keys = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    # 缺失惩罚：rank = result_count + 100 → 贡献 ≈ 0
    d_penalty = len(dense_results) + 100
    s_penalty = len(sparse_results) + 100

    # {doc_id → Document} 查找表
    doc_map: Dict[str, Document] = {}
    for d, _ in dense_results:
        doc_map.setdefault(_doc_id(d), d)
    for d, _ in sparse_results:
        doc_map.setdefault(_doc_id(d), d)

    scored: List[Tuple[Document, float, Dict]] = []
    for key in all_keys:
        dr = dense_ranks.get(key, d_penalty)
        sr = sparse_ranks.get(key, s_penalty)
        rrf = dense_weight / (k + dr) + sparse_weight / (k + sr)
        scored.append((
            doc_map[key], rrf,
            {"dense_rank": dense_ranks.get(key), "sparse_rank": sparse_ranks.get(key), "rrf_score": rrf},
        ))

    scored.sort(key=lambda x: x[1], reverse=True)
    fused = scored[:top_k]

    logger.info("[RRF] Fused %d candidates → top-%d", len(scored), len(fused))
    for rank, (doc, rrf_score, meta) in enumerate(fused, 1):
        src = doc.metadata.get("file_name", "?")
        preview = doc.page_content[:60].replace("\n", " ")
        logger.debug(
            "  [RRF #%d score=%.6f] dense_rank=%s sparse_rank=%s | %s | %s",
            rank, rrf_score, meta.get("dense_rank", "-"), meta.get("sparse_rank", "-"),
            src, preview,
        )
    return fused
