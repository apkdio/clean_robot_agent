"""RRF（倒数排名融合）：按排名位置合并稠密与稀疏两路结果。

只用排名、不用原始分数，以避开两路分数尺度不可比的问题。
公式：score(d) = Σ w_i / (k + rank_i(d))，某文档缺失的一路取惩罚 rank。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.documents import Document

from config_tool import load_config
from log_tool import get_logger

logger = get_logger(name="rrf_fusion")

_rag_cfg = load_config("rag")
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
    """融合两路结果，返回按 RRF 分降序的 [(doc, score, meta), ...]。

    meta 含 dense_rank / sparse_rank / rrf_score；缺失的一路记 None（无排名）。
    两路入参都须按各自分数降序排列，rank 由列表位置推导。
    """
    k = k if k is not None else _rrf_cfg.get("k", 60)
    top_k = top_k if top_k is not None else _retrieval_cfg.get("final_top_k", 3)
    dense_weight = dense_weight if dense_weight is not None else _rrf_cfg.get("dense_weight", 1.0)
    sparse_weight = sparse_weight if sparse_weight is not None else _rrf_cfg.get("sparse_weight", 1.0)

    dense_ranks: Dict[str, int] = {}
    for rank, (doc, _) in enumerate(dense_results, start=1):
        dense_ranks[_doc_id(doc)] = rank

    sparse_ranks: Dict[str, int] = {}
    for rank, (doc, _) in enumerate(sparse_results, start=1):
        sparse_ranks[_doc_id(doc)] = rank

    all_keys = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    # 缺失惩罚 rank 取「该路结果数 + 100」，使其 RRF 贡献 ≈ 0；
    # 用实际结果数而非常数，惩罚强度随该路召回规模自适应。
    d_penalty = len(dense_results) + 100
    s_penalty = len(sparse_results) + 100

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
