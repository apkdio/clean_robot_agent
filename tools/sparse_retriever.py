"""Sparse retriever: BM25 keyword-based retrieval.

BM25 (Best Matching 25) is a probabilistic ranking function, an improved TF-IDF.
It measures how well a document matches a query based on term frequency,
inverse document frequency, and document length normalization.

Tokenization strategy for Chinese + English mixed text:
- Latin letters / digits → kept as whole tokens (lowercased)
- Chinese characters → each character is a separate token
- Punctuation → discarded

Uses rank_bm25.BM25Okapi internally.
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
    """Check if a Document's metadata satisfies a Chroma-style where filter.

    Supports the subset we generate: {"min_price": {"$lte": N}}, {"$and": [...]},
    "$gte" / "$lte" / "$eq" comparisons. Unknown operators pass through.
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
    """BM25-based sparse (keyword) retriever."""

    def __init__(self) -> None:
        self.chunks: List[Document] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25_model: BM25Okapi | None = None

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize Chinese + English mixed text.

        Examples:
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
    # Index
    # ------------------------------------------------------------------

    def index_documents(self, documents: List[Document]) -> None:
        """Build BM25 index from a list of Document chunks.

        1. Tokenize every chunk
        2. BM25Okapi computes DF/IDF and builds the scoring model
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
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: int | None = None, filter: dict | None = None
    ) -> List[Tuple[Document, float]]:
        """Execute BM25 retrieval.

        Args:
            query: User query string.
            top_k: Number of results to return (default from rag.yaml).
            filter: Optional Chroma-style `where` dict applied to candidates.

        Returns:
            [(doc, bm25_score), ...] sorted by score descending.
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
        # Apply metadata filter over BM25 candidates (fetch a bit more first)
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
    # Pickle persistence
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
