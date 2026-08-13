"""Vector store module: chunk, embed, and persist documents into Chroma.

Uses the local Ollama embedding model (qwen3-embedding:0.6b) via the
OpenAI-compatible endpoint, and Chroma as the persistent vector store.
"""

import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config_tool import load_chroma_config, load_rag_config
from file_tools import extract_file, get_file_md5_hex
from llm_tool import get_embedding_model
from log_tool import get_logger
from path_tool import get_abs_path

logger = get_logger(name="vector_store")

_chroma_cfg = load_chroma_config()
_rag_cfg = load_rag_config()


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """Build a text splitter from rag.yaml chunk settings."""
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
    """Return a Chroma instance backed by the persistent store and local embeddings."""
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
    """Parse, chunk, and embed a single file into the vector store.

    Args:
        file_path: Absolute path to the file to ingest.
        source_tag: Optional label stored in metadata as 'source'
                    (defaults to the file name).

    Returns:
        dict with status, file, chunks, md5, elapsed.
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

    # Chunk the documents: prefer numbered-entry splitter, fall back to generic
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
        # No numbered entries → fall back to generic character splitter
        logger.info("[Ingest] No numbered entries, using generic splitter for %s", file_name)
        splitter = _get_splitter()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata.setdefault("source", tag)
            chunk.metadata["file_name"] = file_name
            chunk.metadata["file_md5"] = md5

    if not chunks:
        return {"status": "error", "file": file_path, "message": "Splitting returned no chunks."}

    # Stamp each chunk with structured metadata (price etc.)
    from metadata_extractor import extract_price_metadata
    for chunk in chunks:
        chunk.metadata.setdefault("source", tag)
        chunk.metadata.setdefault("file_name", file_name)
        chunk.metadata["file_md5"] = md5
        chunk.metadata.update(extract_price_metadata(chunk.page_content))

    # Persist to Chroma
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
    """Ingest all supported files in a directory (non-recursive)."""
    exts = tuple(_rag_cfg.get("supported_exts", [".txt", ".pdf"]))
    results = []
    for entry in sorted(os.listdir(dir_path)):
        fp = os.path.join(dir_path, entry)
        if os.path.isfile(fp) and entry.lower().endswith(exts):
            results.append(ingest_file(fp))
    return results


def ingest_data_dir(data_dir: str = None) -> list[dict]:
    """One-click: ingest all files from data/ (and data/external/) into the vector store.

    Args:
        data_dir: Path to the data directory. Defaults to project_root/data.

    Returns:
        List of ingest results, one dict per file.
    """
    if data_dir is None:
        data_dir = get_abs_path("data")
    if not os.path.isdir(data_dir):
        logger.error(f"[IngestDataDir] {data_dir} is not a directory.")
        return [{"status": "error", "message": f"{data_dir} is not a directory."}]

    results = []
    # Ingest files directly in data/
    results.extend(ingest_directory(data_dir))
    # Also ingest data/external/ if it exists
    external = os.path.join(data_dir, "external")
    if os.path.isdir(external):
        results.extend(ingest_directory(external))
    return results


# ---------------------------------------------------------------------------
# DenseRetriever class — wraps Chroma for use in the hybrid (dual-route) pipeline
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Dense retriever backed by the project's Chroma vector store.

    Provides a clean search() interface that returns (Document, score) tuples,
    matching the interface expected by rrf_fusion and the hybrid orchestrator.
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
    """Return ALL chunks matching a metadata filter (no vector ranking).

    Used for structured queries (e.g. budget) where we need full enumeration
    of every in-budget item rather than a similarity-ranked top-k.

    Args:
        filter: Chroma `where` filter dict, e.g. {"min_price": {"$lte": 1000}}.

    Returns:
        List of Documents matching the filter (order not guaranteed).
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
    """Load all stored Chroma chunks and build the BM25 (sparse) index.

    This is called after dense ingestion is complete so the sparse retriever
    has the same chunk set.

    Args:
        sparse_retriever: A SparseRetriever instance. Imported lazily to
                          avoid circular imports.

    Returns:
        Number of chunks indexed in the sparse retriever.
    """
    if sparse_retriever is None:
        from sparse_retriever import SparseRetriever
        sparse_retriever = SparseRetriever()

    # Try to restore from pickle cache first (avoids rebuild on restart)
    if sparse_retriever.load():
        return len(sparse_retriever.chunks)

    # Cache miss or stale — rebuild from Chroma
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
    """Return basic stats about the current collection."""
    store = get_vector_store()
    count = store._collection.count()
    return {
        "collection_name": _get_collection_name(),
        "chunk_count": count,
    }


def reset_collection() -> int:
    """Delete all chunks in the current collection. Returns previous count."""
    store = get_vector_store()
    count = store._collection.count()
    # Delete all by a non-existent id range is unreliable; use metadata filter instead
    store._collection.delete(where={"file_name": {"$ne": "__never__"}})
    logger.warning(f"[Reset] Cleared {count} chunk(s) from collection.")
    return count
