"""向量库模块：把文档分块、向量化并持久化到 Chroma。

通过 OpenAI 兼容端点使用本地 Ollama embedding 模型（bge-m3），
Chroma 作为持久化向量库。
"""

import os
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools import trace_store
from tools.config_tool import load_config
from tools.file_tools import extract_file, get_file_sha256_hex
from tools.llm_tool import get_embedding_model
from tools.log_tool import get_logger
from tools.path_tool import get_abs_path

logger = get_logger(name="vector_store")

_chroma_cfg = load_config("chroma")
_rag_cfg = load_config("rag")


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """根据 rag.yaml 的分块配置构建文本切分器。"""
    chunk_cfg = _rag_cfg.get("chunk", {})
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg.get("chunk_size", 500),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 50),
        separators=chunk_cfg.get("separators", ["\n\n", "\n", "。", ".", " ", ""]),
        add_start_index=True,
    )


def _get_persist_dir() -> str:
    path = get_abs_path(_chroma_cfg.get("persist_dir", "data\\vector_store"))
    os.makedirs(path, exist_ok=True)
    return path


def _get_collection_name() -> str:
    return _chroma_cfg.get("collection_name", "clean_robot_kb")


def get_vector_store() -> Chroma:
    """返回一个基于持久化存储和本地 embedding 的 Chroma 实例。"""
    # 不再显式传 embedding 参数：来源统一在 llm_tool 里按 chroma.yaml 解析
    # （此前这里手传、intent_router 用默认值，两处会漂移）
    embedding = get_embedding_model()
    return Chroma(
        collection_name=_get_collection_name(),
        embedding_function=embedding,
        persist_directory=_get_persist_dir(),
        collection_metadata={"hnsw:space": "cosine"}
    )


def store_fingerprint() -> str:
    """索引指纹：逐 chunk 的 (文件名, 文件 sha256) 排序后的摘要。

    条数、文件名、文件内容任一变化都会变。只数 chunk 条数会漏掉「改了正文但条数不变」
    （改价格 / 改参数 / 改错字），那会让 models.pkl 的型号价格与 BM25 的关键词索引一直用旧数据。
    """
    import hashlib

    got = get_vector_store()._collection.get(where={"file_name": {"$ne": "__never__"}},
                                             include=["metadatas"])
    rows = sorted("%s\u0001%s" % ((m or {}).get("file_name") or "?", (m or {}).get("file_sha256") or "-")
                  for m in got.get("metadatas") or [])
    return hashlib.sha1("\n".join(rows).encode("utf-8")).hexdigest()[:12]


def ids_of_file(file_name: str) -> list:
    """某文件当前在索引里的全部 chunk id（替换式重入用）。"""
    got = get_vector_store()._collection.get(where={"file_name": file_name})
    return list(got.get("ids") or [])


def ingest_file(file_path: str, source_tag: str = "") -> dict:
    """解析、分块并把单个文件向量化入库，返回 {status, file, chunks, sha256, elapsed}。

    同一文件重复调用 = 替换：先抓旧 chunk 的 id，入库成功后再删——
    失败时旧 chunk 原样留着（不留「0 chunk」空洞），成功时不会叠加重复条目。
    """
    import time

    if not os.path.isfile(file_path):
        logger.error(f"[Ingest] File not found: {file_path}")
        return {"status": "error", "file": file_path, "message": "File not found."}

    sha256 = get_file_sha256_hex(file_path)
    if sha256 is None:
        return {"status": "error", "file": file_path, "message": "SHA256 calculation failed."}

    start = time.time()
    logger.info(f"[Ingest] Extracting: {file_path}")
    docs = extract_file(file_path)
    if not docs:
        return {"status": "error", "file": file_path, "message": "Extraction returned no content."}

    # 分块：优先用编号条目切分器，失败则回退到通用字符切分
    from tools.entry_splitter import split_numbered_entries
    file_name = os.path.basename(file_path)
    tag = source_tag or file_name

    chunks: list[Document] = []
    for doc in docs:
        entry_chunks = split_numbered_entries(
            doc.page_content,
            metadata={**doc.metadata, "source": tag, "file_name": file_name, "file_sha256": sha256},
        )
        chunks.extend(entry_chunks)
        # 有编号条目的文档，条目之外的非空正文（导语/章节说明/普通段落）不会成为 chunk——
        # 数出来告警，静默丢弃会让依赖概述的问答召回到一半却无处排查
        if entry_chunks:
            n_entries = len(entry_chunks)
            n_lines = len([ln for ln in doc.page_content.splitlines()
                           if ln.strip() and not ln.strip().startswith(("#", ">"))])
            dropped = n_lines - sum(len(c.page_content.splitlines()) for c in entry_chunks)
            if dropped > 0:
                logger.warning(
                    "[Ingest] %s 有 %d 行非编号正文未成 chunk（导语/概述段会被丢弃），条目数=%d",
                    file_name, dropped, n_entries,
                )

    if not chunks:
        # 没有编号条目 → 回退到通用字符切分器
        logger.info("[Ingest] No numbered entries, using generic splitter for %s", file_name)
        splitter = _get_splitter()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata.setdefault("source", tag)
            chunk.metadata["file_name"] = file_name
            chunk.metadata["file_sha256"] = sha256

    if not chunks:
        return {"status": "error", "file": file_path, "message": "Splitting returned no chunks."}

    # 给每个 chunk 打上结构化 metadata（价格、发布时间、通用规格参数）
    from tools.metadata_extractor import extract_price_metadata, extract_publish_date, extract_spec_metadata
    for chunk in chunks:
        chunk.metadata.setdefault("source", tag)
        chunk.metadata.setdefault("file_name", file_name)
        chunk.metadata["file_sha256"] = sha256
        chunk.metadata.update(extract_price_metadata(chunk.page_content))
        chunk.metadata.update(extract_publish_date(chunk.page_content))
        # 规格行 → param_<键> / param_<键>_num（通用，不预设参数名）
        chunk.metadata.update(extract_spec_metadata(chunk.page_content))

    # 持久化到 Chroma
    logger.info(f"[Ingest] Embedding {len(chunks)} chunk(s) from {file_name}")
    store = get_vector_store()
    old_ids = ids_of_file(file_name)
    store.add_documents(chunks)
    if old_ids:
        store._collection.delete(ids=old_ids)
        logger.info("[Ingest] Replaced %d old chunk(s) of %s", len(old_ids), file_name)

    elapsed = round(time.time() - start, 2)
    logger.info(f"[Ingest] Done {file_name}: {len(chunks)} chunks, {elapsed}s")
    return {
        "status": "ok",
        "file": file_name,
        "chunks": len(chunks),
        "sha256": sha256,
        "elapsed": elapsed,
    }


def ingest_directory(dir_path: str) -> list[dict]:
    """入库目录下所有支持的文件（非递归）。"""
    exts = tuple(_rag_cfg.get("supported_exts", [".txt", ".pdf"]))
    results = []
    for entry in sorted(os.listdir(dir_path)):
        fp = os.path.join(dir_path, entry)
        if os.path.isfile(fp) and entry.lower().endswith(exts):
            results.append(ingest_file(fp))
    return results


def ingest_data_dir(data_dir: str = None) -> list[dict]:
    """把知识库目录下所有支持的文件入库，返回每个文件一条结果 dict。"""
    if data_dir is None:
        from tools.config_tool import get_data_dir
        data_dir = get_data_dir()
    if not os.path.isdir(data_dir):
        logger.error(f"[IngestDataDir] {data_dir} is not a directory.")
        return [{"status": "error", "message": f"{data_dir} is not a directory."}]

    return ingest_directory(data_dir)


# ---------------------------------------------------------------------------
# DenseRetriever 类 —— 封装 Chroma，供双路召回流水线使用
# ---------------------------------------------------------------------------

class DenseRetriever:
    """基于项目 Chroma 向量库的稠密检索器，返回 (Document, score) 元组。"""

    def search(
        self, query: str, top_k: int | None = None, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        k = top_k if top_k is not None else _rag_cfg.get("retrieval", {}).get("dense_top_k", 10)
        store = get_vector_store()
        embedding_fn = getattr(store, "_embedding_function", None)
        if embedding_fn is not None:
            # 先单独计时 query embedding，再按向量查库；两步结果与
            # similarity_search_with_relevance_scores 等价，只是把耗时拆开。
            started = time.perf_counter()
            query_embedding = embedding_fn.embed_query(query)
            trace_store.add_ms("embed_query", started)
            started = time.perf_counter()
            results = store.similarity_search_by_vector_with_relevance_scores(
                query_embedding, k=k, filter=filter)
            trace_store.add_ms("chroma_dense", started)
            relevance = store._select_relevance_score_fn()
            results = [(doc, relevance(score)) for doc, score in results]
        else:
            started = time.perf_counter()
            results = store.similarity_search_with_relevance_scores(query, k=k, filter=filter)
            trace_store.add_ms("chroma_dense", started)

        # 相关性阈值过滤（0 或未配置 = 不过滤）。bge-m3 + cosine 实测：领域内 query 通常
        # 0.36+，领域外（天气/闲聊/乱码）落在 0.16~0.24，0.25 压在噪声底噪之上。
        threshold = _rag_cfg.get("retrieval", {}).get("score_threshold", 0) or 0
        kept = results
        if threshold > 0:
            kept = [(doc, score) for doc, score in results if score >= threshold]

        logger.info(
            "[Dense] query='%s' filter=%s → %d results (threshold=%.2f, kept=%d)",
            query[:50], filter, len(results), threshold, len(kept),
        )
        for rank, (doc, score) in enumerate(kept, 1):
            src = doc.metadata.get("file_name", "?")
            preview = doc.page_content[:60].replace("\n", " ")
            logger.debug("  [Dense #%d score=%.4f] %s | %s", rank, score, src, preview)
        return kept


def search_by_filter(filter: dict) -> list[Document]:
    """返回所有匹配 metadata 过滤条件的 chunk（不做向量排序，顺序不保证）。

    结构化查询（如预算）需要完整枚举每一条，不能按相似度取 top-k。
    """
    store = get_vector_store()
    data = store._collection.get(where=filter, include=["metadatas", "documents"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
    logger.info("[Filter] where=%s → %d chunks", filter, len(docs))
    return docs


def build_hybrid_index(sparse_retriever=None) -> int:
    """从 Chroma 全量 chunk 构建（或从 pickle 恢复）BM25 索引，返回 chunk 数。

    需在稠密入库完成后调用，保证两路拥有同一份 chunk 集合。
    """
    if sparse_retriever is None:
        from tools.sparse_retriever import SparseRetriever
        sparse_retriever = SparseRetriever()

    # 先尝试从 pickle 缓存恢复（避免重启后重建）
    if sparse_retriever.load():
        return len(sparse_retriever.chunks)

    # 缓存缺失或过期 —— 从 Chroma 重建
    store = get_vector_store()
    data = store._collection.get(include=["metadatas", "documents"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
    sparse_retriever.index_documents(docs)
    sparse_retriever.save()
    return len(docs)


def list_collections_info() -> dict:
    """返回当前 collection 的基本统计信息。"""
    store = get_vector_store()
    count = store._collection.count()
    return {
        "collection_name": _get_collection_name(),
        "chunk_count": count,
    }


def reset_collection() -> int:
    """清空当前 collection 的所有 chunk。返回清空前的数量。"""
    store = get_vector_store()
    count = store._collection.count()
    # 用不存在的 id 区间删除不可靠，改用 metadata 过滤删除全部
    store._collection.delete(where={"file_name": {"$ne": "__never__"}})
    logger.warning(f"[Reset] Cleared {count} chunk(s) from collection.")
    return count
