"""Cross-Encoder 精排：在 RRF 融合之后、截断之前，用 bge-reranker-v2-m3 对
(query, chunk) 逐对打分，把「排名融合」升级为「语义相关性排序」。

模型走本地 transformers 推理（Ollama 无 rerank 端点），首次调用懒加载；依赖缺失或
加载失败时降级为保持 RRF 顺序，并记 WARNING——降级是显式可观测的，不是静默跳过。
配置见 config/rag.yaml 的 rerank 段。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from config_tool import load_config
from langchain_core.documents import Document
from log_tool import get_logger

logger = get_logger(name="reranker")

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

_model = None
_tokenizer = None
_load_failed = False  # 加载失败后不再重试（避免每次检索都卡在加载上）


def _rerank_cfg() -> dict:
    """读取 rag.yaml 的 rerank 段（失败返回空 dict，走内置默认）。"""
    try:
        return load_config("rag").get("rerank", {}) or {}
    except Exception as e:
        logger.warning(f"[Rerank] load config failed: {e}")
        return {}


def _ensure_model() -> bool:
    """懒加载精排模型；成功返回 True，失败返回 False 并降级。"""
    global _model, _tokenizer, _load_failed
    if _model is not None:
        return True
    if _load_failed:
        return False
    model_name = _rerank_cfg().get("model", _DEFAULT_MODEL)
    try:
        # 
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("[Rerank] loading model=%s", model_name)
        # 优先只用本地缓存，避免每次启动都发起 HF 网络校验（国内环境容易挂起）；
        # 缓存未命中才回退到联网加载。
        tokenizer = model = None
        for local_only in (True, False):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, local_files_only=local_only
                )
                break
            except Exception as e:
                if local_only:
                    logger.info("[Rerank] not in local cache, try downloading: %s", str(e)[:120])
                    continue
                raise
        _tokenizer, _model = tokenizer, model
        _model.eval()
        logger.info("[Rerank] model ready")
        return True
    except Exception as e:
        logger.warning("[Rerank] model unavailable, fall back to RRF order: %s", e)
        _load_failed = True
        return False


def _score_batch(query: str, texts: List[str]) -> List[float]:
    """对 (query, text) 逐对打分，返回 sigmoid 归一化后的 [0,1] 分数。"""
    import torch

    cfg = _rerank_cfg()
    max_length = int(cfg.get("max_length", 512))
    batch_size = int(cfg.get("batch_size", 8))

    scores: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = _tokenizer(
            [query] * len(batch), batch,
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        with torch.no_grad():
            logits = _model(**inputs).logits.view(-1).float()
        scores.extend(torch.sigmoid(logits).tolist())
    return scores


def rerank(
    query: str,
    candidates: List[Tuple[Document, float, Dict]],
    top_k: int | None = None,
) -> List[Tuple[Document, float, Dict]] | None:
    """按语义相关性重排候选，返回 meta 内补了 rerank_score 的降序列表。

    模型不可用时返回 None，调用方据此退回 RRF 顺序。
    """
    if not candidates:
        return candidates
    if not _ensure_model():
        return None

    texts = [doc.page_content for doc, _score, _meta in candidates]
    try:
        scores = _score_batch(query, texts)
    except Exception as e:
        logger.warning("[Rerank] scoring failed, fall back to RRF order: %s", e)
        return None

    rescored: List[Tuple[Document, float, Dict]] = []
    for (doc, _rrf_score, meta), score in zip(candidates, scores):
        new_meta = dict(meta or {})
        new_meta["rerank_score"] = score
        rescored.append((doc, score, new_meta))
    rescored.sort(key=lambda x: x[1], reverse=True)

    top = rescored[:top_k] if top_k else rescored
    top1 = top[0][1] if top else 0.0
    logger.info("[Rerank] %d candidate(s) → top1=%.4f", len(top), top1)
    for rank, (doc, score, _meta) in enumerate(top, 1):
        src = doc.metadata.get("file_name", "?")
        preview = doc.page_content[:60].replace("\n", " ")
        logger.debug("  [Rerank #%d score=%.4f] %s | %s", rank, score, src, preview)
    return top
