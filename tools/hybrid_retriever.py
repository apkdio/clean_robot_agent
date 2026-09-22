"""混合检索器：稠密（向量）+ 稀疏（BM25）→ RRF 融合 → Cross-Encoder 精排。

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

    def _retrieve(self, query: str, filter: dict | None) -> List[Tuple[Document, float, Dict]]:
        """一次召回：双路 → RRF → 精排，返回**未按阈值过滤**的有序候选。

        返回 [(doc, score, meta), ...]（分数降序），交给 search() 决定怎么用。
        """
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
            return reranked
        # 4'. 融合侧兜底（精排未启用或不可用）：丢弃无稠密支撑的候选。
        #     稠密侧已按阈值过滤，「不在稠密结果里」等价于「稠密相关性低于阈值」；
        #     不丢弃的话，领域外 query 会靠 BM25 的字面命中把噪声带进 Prompt。
        return self._drop_without_dense_support(fused, dense_results)

    def search(self, query: str, filter: dict | None = None) -> List[Document]:
        """双路召回 + RRF + 精排，返回 Document 列表（数量由 retrieval.final_top_k 控制）。

        `filter` 按**软约束**处理：带 filter 召回后若精排 top1 低于
        `rerank.score_threshold`，则撤掉 filter 再召回一次全库，两池交给 `_merge_pools`
        合并。这条路径不再按绝对阈值砍分。filter 为 None 时只有一遍召回，硬阈值照旧挡领域外。
        选型依据（两个直觉写法为何被否决）见 notes/project_detail.md 的决策记录。
        """
        if not self.sparse.is_ready:
            self.ensure_sparse_index()

        retrieval_cfg = _rag_cfg.get("retrieval", {})
        rerank_cfg = _rag_cfg.get("rerank", {})
        final_top_k = retrieval_cfg.get("final_top_k", 5)
        rerank_on = bool(rerank_cfg.get("enabled", False))
        threshold = rerank_cfg.get("score_threshold", 0) or 0

        ranked = self._retrieve(query, filter)
        top1 = ranked[0][1] if ranked else 0.0

        # 阈值只在精排启用时才有意义：关闭精排时 top1 是 RRF 分，量纲不同
        if rerank_on and threshold > 0 and top1 < threshold and filter is not None:
            logger.warning(
                "[Hybrid] in-domain top1=%.4f < threshold=%.2f → second pass over full KB",
                top1, threshold,
            )
            ranked = self._merge_pools(ranked, self._retrieve(query, None), final_top_k)
            logger.info(
                "[Hybrid] merged pools → %d candidate(s) (fallback quota=%d, no absolute cut), top score=%.4f",
                len(ranked), (final_top_k + 1) // 2, ranked[0][1] if ranked else 0.0,
            )
        elif rerank_on and threshold > 0:
            before = len(ranked)
            ranked = [item for item in ranked if item[1] >= threshold]
            logger.info(
                "[Hybrid] threshold cut %d → %d candidate(s) (threshold=%.2f)",
                before, len(ranked), threshold,
            )

        documents = [doc for doc, _score, _meta in ranked[:final_top_k]]
        logger.info("[Hybrid] query='%s' → %d result(s)", query[:40], len(documents))
        return documents

    @staticmethod
    def _merge_pools(
        primary: List[Tuple[Document, float, Dict]],
        fallback: List[Tuple[Document, float, Dict]],
        top_k: int,
    ) -> List[Tuple[Document, float, Dict]]:
        """合并第一遍（域内）与第二遍（全库兜底）候选：按精排分排序 + 兜底池限额 ⌈top_k/2⌉。

        两个直觉写法都实测否决：纯按精排分排序会让域内整池被另一个域的高分噪声整池挤掉；
        纯按位次融合会把高置信的正确结果拉下来。故取中间——按分排序保住高置信结果，
        同时限制兜底池名额（兜底只补漏，不该占掉半个以上的 Prompt）。
        另两条约束：**合并而非替换**（域内池有全库池没有的相关条目）；
        两池都召回的条目算第一遍池的，不占兜底名额。
        实测数据见 notes/BADCASES.md 与 notes/project_detail.md 的决策记录。
        """
        quota = (top_k + 1) // 2          # ⌈top_k/2⌉
        primary_keys = {doc.page_content for doc, _score, _meta in primary}
        fallback_only = {doc.page_content for doc, _score, _meta in fallback} - primary_keys

        best: Dict[str, Tuple[Document, float, Dict]] = {}
        for doc, score, meta in list(primary) + list(fallback):
            key = doc.page_content
            if key not in best or score > best[key][1]:
                best[key] = (doc, score, meta)

        merged: List[Tuple[Document, float, Dict]] = []
        used = 0
        for item in sorted(best.values(), key=lambda x: x[1], reverse=True):
            if item[0].page_content in fallback_only:
                if used >= quota:
                    continue
                used += 1
            merged.append(item)
            if len(merged) == top_k:
                break
        return merged

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
