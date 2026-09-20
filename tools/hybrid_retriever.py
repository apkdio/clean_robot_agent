"""混合检索器：稠密（向量）+ 稀疏（BM25）→ RRF 融合 → Cross-Encoder 精排。

用法：
    hr = HybridRetriever()
    hr.ensure_sparse_index()       # 从 Chroma chunk 构建 BM25
    chunks = hr.search("边刷多久换")  # 返回精排后的 top-k Document

过滤链路（逐层收紧）：
    1. 稠密侧：低于 retrieval.score_threshold 的 chunk 直接丢弃（vector_store）
    2. 精排侧：RRF 候选交给 bge-reranker-v2-m3 重排，按 rerank.score_threshold 过滤
       （精排不可用时退回 RRF 顺序，并改按「无稠密支撑」兜底过滤）
    3. 截断：取 retrieval.final_top_k
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.documents import Document

from config_tool import load_config
from log_tool import get_logger
from reranker import rerank
from rrf_fusion import reciprocal_rank_fusion
from sparse_retriever import SparseRetriever
from vector_store import DenseRetriever, build_hybrid_index

logger = get_logger(name="hybrid_retriever")

_rag_cfg = load_config("rag")


class HybridRetriever:
    """编排召回链路：稠密 + 稀疏 → RRF 融合 → Cross-Encoder 精排。"""

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
        """执行稠密 + 稀疏检索，经 RRF 融合与精排，返回 Document 列表。

        参数：
            query: 用户问题。
            filter: 可选的 Chroma `where` 过滤器，用于结构化维度
                    （例如 {"min_price": {"$lte": 1000}}）。

        返回：
            精排后的 Document（数量由 rag.yaml → retrieval.final_top_k 控制）。
        """
        if not self.sparse.is_ready:
            self.ensure_sparse_index()

        retrieval_cfg = _rag_cfg.get("retrieval", {})
        rerank_cfg = _rag_cfg.get("rerank", {})
        final_top_k = retrieval_cfg.get("final_top_k", 5)
        rerank_on = bool(rerank_cfg.get("enabled", False))
        # 候选数与最终数解耦：精排开启时放宽 RRF 保留宽度，交给精排收窄
        candidate_top_k = rerank_cfg.get("candidate_top_k", final_top_k) if rerank_on else final_top_k

        # 1. 稠密（带 metadata 过滤；内部已按 retrieval.score_threshold 过滤）
        dense_results = self.dense.search(query, filter=filter)

        # 2. 稀疏（在 BM25 候选之上再套用过滤）
        sparse_results = self.sparse.search(query, filter=filter)

        # 3. RRF 融合
        fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=candidate_top_k)

        # 4. 精排：把「排名融合」升级为「语义相关性排序」
        reranked = rerank(query, fused) if rerank_on else None
        if reranked is not None:
            threshold = rerank_cfg.get("score_threshold", 0) or 0
            before = len(reranked)
            if threshold > 0:
                reranked = [item for item in reranked if item[1] >= threshold]
            logger.info(
                "[Hybrid] reranked %d → %d candidate(s) (threshold=%.2f)",
                before, len(reranked), threshold,
            )
            fused = reranked
        else:
            # 4'. 融合侧兜底（精排未启用或不可用）：丢弃无稠密支撑的候选。
            #    稠密侧已按阈值过滤，「不在稠密结果里」等价于「稠密相关性低于阈值」；
            #    不丢弃的话，领域外 query 会靠 BM25 的字面命中把噪声带进 Prompt。
            fused = self._drop_without_dense_support(fused, dense_results)

        documents = [doc for doc, _score, _meta in fused[:final_top_k]]
        logger.info("[Hybrid] query='%s' → %d result(s)", query[:40], len(documents))
        return documents

    @staticmethod
    def _drop_without_dense_support(
        fused: List[Tuple[Document, float, Dict]],
        dense_results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float, Dict]]:
        """丢弃没有被稠密路召回的候选（即稠密相关性低于阈值）。"""
        supported = {doc.page_content for doc, _score in dense_results}
        kept = [item for item in fused if item[0].page_content in supported]
        if len(kept) < len(fused):
            logger.info(
                "[Hybrid] dropped %d candidate(s) without dense support",
                len(fused) - len(kept),
            )
        return kept
