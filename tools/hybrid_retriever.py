"""混合检索器：稠密（向量）+ 稀疏（BM25）→ RRF 融合。

用法：
    hr = HybridRetriever()
    hr.ensure_sparse_index()       # 从 Chroma chunk 构建 BM25
    chunks = hr.search("边刷多久换")  # 返回融合后的 top-k Document
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
    """编排双路召回：稠密 + 稀疏 → RRF 融合。"""

    def __init__(self) -> None:
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------

    def ensure_sparse_index(self) -> int:
        """从 Chroma 存储的 chunk 构建 BM25 索引（一行搞定）。"""
        n = build_hybrid_index(self.sparse)
        logger.info("[Hybrid] Sparse index ready: %d chunks", n)
        return n

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(self, query: str, filter: dict | None = None) -> List[Document]:
        """执行稠密 + 稀疏检索，经 RRF 融合，返回 Document 列表。

        参数：
            query: 用户问题。
            filter: 可选的 Chroma `where` 过滤器，用于结构化维度
                    （例如 {"min_price": {"$lte": 1000}}）。

        返回：
            Top-k 融合后的 Document（数量由 rag.yaml → final_top_k 控制）。
        """
        if not self.sparse.is_ready:
            self.ensure_sparse_index()

        # 1. 稠密（带 metadata 过滤）
        dense_results = self.dense.search(query, filter=filter)

        # 2. 稀疏（在 BM25 候选之上再套用过滤）
        sparse_results = self.sparse.search(query, filter=filter)

        # 3. RRF 融合
        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        # 4. 返回纯 Document 列表
        documents = [doc for doc, _score, _meta in fused]
        logger.info("[Hybrid] query='%s' → %d fused result(s)", query[:40], len(documents))
        return documents
