"""RRF (Reciprocal Rank Fusion) — merge heterogeneous ranked lists into one.

Merges dense (vector similarity) and sparse (BM25 keyword) result lists
using only rank positions, not raw scores, so different score scales
don't interfere with each other.

Formula for each candidate document d:
    RRF_score(d) = sum over retrievers i of:  w_i / (k + rank_i(d))

where:
    k     = smoothing constant (typ. 60)
    w_i   = weight for retriever i
    rank_i(d) = rank of d in retriever i (starting from 1)
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
    """Use page_content hash as document identity for dedup."""
    return doc.page_content


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Document, float]],
    sparse_results: List[Tuple[Document, float]],
    k: int | None = None,
    top_k: int | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
) -> List[Tuple[Document, float, Dict]]:
    """Fuse dense and sparse retrieval results via RRF.

    Args:
        dense_results:  [(doc, cosine_score), ...] sorted descending
        sparse_results: [(doc, bm25_score), ...] sorted descending
        k:              RRF smoothing constant (default from rag.yaml)
        top_k:          number of results to return (default from rag.yaml)
        dense_weight:   weight for dense route
        sparse_weight:  weight for sparse route

    Returns:
        [(doc, rrf_score, meta), ...] sorted by RRF score descending.
        meta = {"dense_rank": int|None, "sparse_rank": int|None, "rrf_score": float}
    """
    k = k if k is not None else _rrf_cfg.get("k", 60)
    top_k = top_k if top_k is not None else _retrieval_cfg.get("final_top_k", 3)
    dense_weight = dense_weight if dense_weight is not None else _rrf_cfg.get("dense_weight", 1.0)
    sparse_weight = sparse_weight if sparse_weight is not None else _rrf_cfg.get("sparse_weight", 1.0)

    # Build {doc_id → rank} maps
    dense_ranks: Dict[str, int] = {}
    for rank, (doc, _) in enumerate(dense_results, start=1):
        dense_ranks[_doc_id(doc)] = rank

    sparse_ranks: Dict[str, int] = {}
    for rank, (doc, _) in enumerate(sparse_results, start=1):
        sparse_ranks[_doc_id(doc)] = rank

    # Collect all unique candidates
    all_keys = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    # Miss penalty: rank = result_count + 100 → contribution ≈ 0
    d_penalty = len(dense_results) + 100
    s_penalty = len(sparse_results) + 100

    # {doc_id → Document} lookup
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
