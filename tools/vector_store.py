"""向量库模块：把文档分块、向量化并持久化到 Chroma。

通过 OpenAI 兼容端点使用本地 Ollama embedding 模型（bge-m3），
Chroma 作为持久化向量库。
"""

import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config_tool import load_config
from file_tools import extract_file, get_file_md5_hex
from llm_tool import get_embedding_model
from log_tool import get_logger
from path_tool import get_abs_path

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
    embedding = get_embedding_model(
        model=_chroma_cfg["embedding"]["model"],
        base_url=_chroma_cfg["embedding"]["base_url"],
        api_key=_chroma_cfg["embedding"]["api_key"],
    )
    return Chroma(
        collection_name=_get_collection_name(),
        embedding_function=embedding,
        persist_directory=_get_persist_dir(),
        collection_metadata={"hnsw:space": "cosine"}
    )


def ingest_file(file_path: str, source_tag: str = "") -> dict:
    """解析、分块并把单个文件向量化入库。

    参数：
        file_path: 待入库文件的绝对路径。
        source_tag: 可选标签，作为 metadata 的 'source'（默认用文件名）。

    返回：
        含 status、file、chunks、md5、elapsed 的 dict。
    """
    import time

    if not os.path.isfile(file_path):
        logger.error(f"[Ingest] File not found: {file_path}")
        return {"status": "error", "file": file_path, "message": "File not found."}

    md5 = get_file_md5_hex(file_path)
    if md5 is None:
        return {"status": "error", "file": file_path, "message": "MD5 calculation failed."}

    start = time.time()
    logger.info(f"[Ingest] Extracting: {file_path}")
    docs = extract_file(file_path)
    if not docs:
        return {"status": "error", "file": file_path, "message": "Extraction returned no content."}

    # 分块：优先用编号条目切分器，失败则回退到通用字符切分
    from entry_splitter import split_numbered_entries
    file_name = os.path.basename(file_path)
    tag = source_tag or file_name

    chunks: list[Document] = []
    for doc in docs:
        entry_chunks = split_numbered_entries(
            doc.page_content,
            metadata={**doc.metadata, "source": tag, "file_name": file_name, "file_md5": md5},
        )
        chunks.extend(entry_chunks)

    if not chunks:
        # 没有编号条目 → 回退到通用字符切分器
        logger.info("[Ingest] No numbered entries, using generic splitter for %s", file_name)
        splitter = _get_splitter()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata.setdefault("source", tag)
            chunk.metadata["file_name"] = file_name
            chunk.metadata["file_md5"] = md5

    if not chunks:
        return {"status": "error", "file": file_path, "message": "Splitting returned no chunks."}

    # 给每个 chunk 打上结构化 metadata（价格、发布时间）
    from metadata_extractor import extract_price_metadata, extract_publish_date
    for chunk in chunks:
        chunk.metadata.setdefault("source", tag)
        chunk.metadata.setdefault("file_name", file_name)
        chunk.metadata["file_md5"] = md5
        chunk.metadata.update(extract_price_metadata(chunk.page_content))
        chunk.metadata.update(extract_publish_date(chunk.page_content))

    # 持久化到 Chroma
    logger.info(f"[Ingest] Embedding {len(chunks)} chunk(s) from {file_name}")
    store = get_vector_store()
    store.add_documents(chunks)

    elapsed = round(time.time() - start, 2)
    logger.info(f"[Ingest] Done {file_name}: {len(chunks)} chunks, {elapsed}s")
    return {
        "status": "ok",
        "file": file_name,
        "chunks": len(chunks),
        "md5": md5,
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
    """把知识库目录下所有支持的文件入库。

    参数：
        data_dir: 知识库目录路径，默认 project_root/data/knowledge。

    返回：
        入库结果列表，每个文件一个 dict。
    """
    if data_dir is None:
        data_dir = get_abs_path("data/knowledge")
    if not os.path.isdir(data_dir):
        logger.error(f"[IngestDataDir] {data_dir} is not a directory.")
        return [{"status": "error", "message": f"{data_dir} is not a directory."}]

    return ingest_directory(data_dir)


# ---------------------------------------------------------------------------
# DenseRetriever 类 —— 封装 Chroma，供双路召回流水线使用
# ---------------------------------------------------------------------------

class DenseRetriever:
    """基于项目 Chroma 向量库的稠密检索器。

    提供简洁的 search() 接口，返回 (Document, score) 元组，
    与 rrf_fusion 和双路召回编排层期望的接口一致。
    """

    def search(
        self, query: str, top_k: int | None = None, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        k = top_k if top_k is not None else _rag_cfg.get("retrieval", {}).get("dense_top_k", 10)
        store = get_vector_store()
        results = store.similarity_search_with_relevance_scores(query, k=k, filter=filter)
        logger.info("[Dense] query='%s' filter=%s → %d results", query[:50], filter, len(results))
        for rank, (doc, score) in enumerate(results, 1):
            src = doc.metadata.get("file_name", "?")
            preview = doc.page_content[:60].replace("\n", " ")
            logger.debug("  [Dense #%d score=%.4f] %s | %s", rank, score, src, preview)
        return results


def search_by_filter(filter: dict) -> list[Document]:
    """返回所有匹配 metadata 过滤条件的 chunk（不做向量排序）。

    用于结构化查询（如预算），需要完整枚举每一个预算内条目，
    而非按相似度排序取 top-k。

    参数：
        filter: Chroma `where` 过滤字典，如 {"min_price": {"$lte": 1000}}。

    返回：
        匹配过滤条件的 Document 列表（顺序不保证）。
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
    """读取 Chroma 里所有 chunk，构建 BM25（稀疏）索引。

    在稠密入库完成后调用，保证稀疏检索器拥有同一份 chunk 集合。

    参数：
        sparse_retriever: SparseRetriever 实例。懒加载导入以避免循环依赖。

    返回：
        稀疏检索器索引的 chunk 数量。
    """
    if sparse_retriever is None:
        from sparse_retriever import SparseRetriever
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
