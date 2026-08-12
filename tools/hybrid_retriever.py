"""Hybrid retriever: dense (vector) + sparse (BM25) → RRF fusion.

Usage:
    hr = HybridRetriever()
    hr.ensure_sparse_index()       # build BM25 from Chroma chunks
    chunks = hr.search("边刷多久换")  # returns fused top-k Documents
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from log_tool import get_logger
from rrf_fusion import reciprocal_rank_fusion
from sparse_retriever import SparseRetriever
from vector_store import DenseRetriever, build_hybrid_index

logger = get_logger(name="hybrid_retriever")


class HybridRetriever:
    """Orchestrates dual-route retrieval: dense + sparse → RRF fusion."""

    def __init__(self) -> None:
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def ensure_sparse_index(self) -> int:
        """Build the BM25 index from Chroma-stored chunks (one-liner)."""
        n = build_hybrid_index(self.sparse)
        logger.info("[Hybrid] Sparse index ready: %d chunks", n)
        return n

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str) -> List[Document]:
        """Run dense + sparse retrieval, fuse via RRF, return Document list.

        Args:
            query: User question.

        Returns:
            Top-k fused Documents (count controlled by rag.yaml → final_top_k).
        """
        if not self.sparse.is_ready:
            self.ensure_sparse_index()

        # 1. Dense
        dense_results = self.dense.search(query)

        # 2. Sparse
        sparse_results = self.sparse.search(query)

        # 3. RRF fusion
        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        # 4. Return plain Document list
        documents = [doc for doc, _score, _meta in fused]
        logger.info("[Hybrid] query='%s' → %d fused result(s)", query[:40], len(documents))
        return documents
