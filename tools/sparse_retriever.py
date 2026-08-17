"""稀疏检索器：基于 BM25 的关键词检索。

BM25（Best Matching 25）是一种概率排序函数，是改进版的 TF-IDF。
它根据词频、逆文档频率以及文档长度归一化来衡量文档与查询的匹配程度。

中英文混合文本的分词策略：
- 拉丁字母 / 数字 → 作为完整 token 保留（转小写）
- 中文字符 → 每个字符作为一个独立 token
- 标点符号 → 丢弃

内部使用 rank_bm25.BM25Okapi。
"""

from __future__ import annotations

import os
import pickle
import re
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config_tool import load_chroma_config, load_rag_config
from log_tool import get_logger
from path_tool import get_abs_path

logger = get_logger(name="sparse_retriever")

_rag_cfg = load_rag_config()
_chroma_cfg = load_chroma_config()
_retrieval_cfg = _rag_cfg.get("retrieval", {})
_sparse_cfg = _rag_cfg.get("sparse", {})

_DEFAULT_K1: float = 1.5
_DEFAULT_B: float = 0.75

_BM25_CACHE_FILE = "data/pkl/bm25_index.pkl"


def _matches_filter(doc: Document, f: dict) -> bool:
    """检查 Document 的 metadata 是否满足 Chroma 风格的 where 过滤器。

    支持我们生成的那些子集：{"min_price": {"$lte": N}}、{"$and": [...]}、
    "$gte" / "$lte" / "$eq" 比较。未知运算符直接放行。
    """
    if not f:
        return True
    if "$and" in f:
        return all(_matches_filter(doc, sub) for sub in f["$and"])
    for key, cond in f.items():
        if key.startswith("$"):
            continue
        val = doc.metadata.get(key)
        if val is None:
            return False
        if isinstance(cond, dict):
            if "$lte" in cond and not (val <= cond["$lte"]):
                return False
            if "$gte" in cond and not (val >= cond["$gte"]):
                return False
            if "$eq" in cond and val != cond["$eq"]:
                return False
        else:
            if val != cond:
                return False
    return True


class SparseRetriever:
    """基于 BM25 的稀疏（关键词）检索器。"""

    def __init__(self) -> None:
        self.chunks: List[Document] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25_model: BM25Okapi | None = None

    # ------------------------------------------------------------------
    # 分词
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """对中英文混合文本进行分词。

        示例：
            "Python 在机器学习中的应用"
            → ["python", "在", "机", "器", "学", "习", "中", "的", "应", "用"]
        """
        tokens: List[str] = []
        for part in re.split(r"([a-zA-Z0-9]+)", text):
            if re.match(r"[a-zA-Z0-9]+", part):
                tokens.append(part.lower())
            else:
                tokens.extend(
                    ch for ch in part
                    if ch.strip()
                    and ch not in '，。！？；：""''（）【】《》、'
                )
        return tokens

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------

    def index_documents(self, documents: List[Document]) -> None:
        """从 Document chunk 列表构建 BM25 索引。

        1. 对每个 chunk 进行分词
        2. BM25Okapi 计算 DF/IDF 并构建评分模型
        """
        if not documents:
            logger.warning("[Sparse] No documents to index.")
            return
        self.chunks = documents
        self.corpus_tokens = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25_model = BM25Okapi(
            self.corpus_tokens,
            k1=_sparse_cfg.get("bm25_k1", _DEFAULT_K1),
            b=_sparse_cfg.get("bm25_b", _DEFAULT_B),
        )
        logger.info("[Sparse] BM25 index built: %d chunks", len(self.chunks))

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: int | None = None, filter: dict | None = None
    ) -> List[Tuple[Document, float]]:
        """执行 BM25 检索。

        参数：
            query: 用户查询字符串。
            top_k: 返回结果数量（默认来自 rag.yaml）。
            filter: 可选的 Chroma 风格 `where` 字典，应用于候选结果。

        返回：
            [(doc, bm25_score), ...]，按分数降序排列。
        """
        if self.bm25_model is None:
            logger.error("[Sparse] Index not built yet.")
            return []

        k = top_k if top_k is not None else _retrieval_cfg.get("sparse_top_k", 10)
        tokenized = self._tokenize(query)
        scores = self.bm25_model.get_scores(tokenized)

        scored = sorted(
            zip(self.chunks, scores), key=lambda x: x[1], reverse=True
        )
        # 在 BM25 候选上套用 metadata 过滤（先多取一些再过滤）
        if filter is not None:
            scored = [pair for pair in scored if _matches_filter(pair[0], filter)]
        scored = scored[:k]
        results = [(doc, float(score)) for doc, score in scored]
        logger.info("[Sparse] query='%s' filter=%s → %d results", query[:50], filter, len(results))
        for rank, (doc, score) in enumerate(results, 1):
            src = doc.metadata.get("file_name", "?")
            preview = doc.page_content[:60].replace("\n", " ")
            logger.debug("  [Sparse #%d score=%.4f] %s | %s", rank, score, src, preview)
        return results

    @property
    def is_ready(self) -> bool:
        return self.bm25_model is not None and len(self.chunks) > 0

    # ------------------------------------------------------------------
    # Pickle 持久化
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_path() -> str:
        return get_abs_path(_BM25_CACHE_FILE)

    @staticmethod
    def _fingerprint() -> dict:
        from vector_store import get_vector_store
        store = get_vector_store()
        return {
            "chunk_count": store._collection.count(),
            "collection_name": _chroma_cfg.get("collection_name", "clean_robot_kb"),
        }

    def save(self) -> None:
        path = self._cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "chunks": self.chunks,
            "corpus_tokens": self.corpus_tokens,
            "bm25_model": self.bm25_model,
            "fingerprint": self._fingerprint(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("[Sparse] BM25 cached: %s (%d chunks)", path, len(self.chunks))

    def load(self) -> bool:
        path = self._cache_path()
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning("[Sparse] Cache load failed: %s", e)
            return False
        fp = payload.get("fingerprint", {})
        cur = self._fingerprint()
        if fp.get("chunk_count") != cur.get("chunk_count"):
            logger.info("[Sparse] Cache stale (%d → %d chunks), rebuild.",
                        fp.get("chunk_count", 0), cur.get("chunk_count", 0))
            return False
        self.chunks = payload["chunks"]
        self.corpus_tokens = payload["corpus_tokens"]
        self.bm25_model = payload["bm25_model"]
        logger.info("[Sparse] BM25 loaded from cache: %d chunks", len(self.chunks))
        return True
