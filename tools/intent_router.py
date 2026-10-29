"""意图路由：用本地 bge-m3 + 训练好的分类头判断用户意图。

四分类头（Linear 1024→4）由 data/datasets/ 训练得到，保存在 data/bgm_model/。
推理时用 Ollama bge-m3 把 query 向量化，再跑分类头前向——远比 LLM 分类快。

标签：robot / other / casual / unknown。
"""

from __future__ import annotations

import json
import random

from config.word_dict_config import GUESS_HINTS as _GUESS_HINTS
from tools.log_tool import get_logger

logger = get_logger(name="intent_router")

_MODEL_DIR = "data/bgm_model"
_HEAD_FILE = "classifier_head.pt"
_LABELS_FILE = "labels.json"

_head = None
_labels = None
_head_unavailable = False   # 加载失败后不再每轮重试（失败是秒级的）
_emb = None


def _get_emb():
    """懒加载 bge-m3 embedding 客户端（跨调用缓存复用）。"""
    global _emb
    if _emb is None:
        from tools.llm_tool import get_embedding_model
        _emb = get_embedding_model()
    return _emb


_QUERY_VEC_CACHE = {}          # query → 向量（意图分类与操作打分共用，同一条 query 只 embed 一次）
_QUERY_VEC_SIZE = 128


def get_query_embedding(query: str):
    """query 的 bge-m3 向量；不可用返回 None（调用方显式降级）。

    与意图分类走同一个 embed_documents 路径；同一条 query 在本进程内只算一次
    ——操作判别复用它，不给每轮多加一次 embedding。
    """
    global _emb_failed_logged
    if query in _QUERY_VEC_CACHE:
        return _QUERY_VEC_CACHE[query]
    try:
        vec = _get_emb().embed_documents([query])[0]
    except Exception as e:  # noqa: BLE001
        if not _emb_failed_logged:
            logger.warning("[IntentRouter] query embedding 不可用，操作打分退化为 BM25：%s", e)
            _emb_failed_logged = True
        return None
    if len(_QUERY_VEC_CACHE) >= _QUERY_VEC_SIZE:
        _QUERY_VEC_CACHE.clear()
    _QUERY_VEC_CACHE[query] = vec
    return vec


_emb_failed_logged = False


# 软引导话术（_GUESS_HINTS）见 config/word_dict_config.py


def _load_model():
    """懒加载训练好的分类头和标签映射；不可用时返回 (None, None)，由调用方显式降级。"""
    global _head, _labels, _head_unavailable
    if _head is not None:
        return _head, _labels
    if _head_unavailable:
        return None, None

    from tools.path_tool import get_abs_path

    # 先在局部装配完再赋值：失败时不留半个状态
    try:
        import torch          # import 放 try 内：torch 缺失也走这条降级，不能裸抛

        with open(get_abs_path(f"{_MODEL_DIR}/{_LABELS_FILE}"), encoding="utf-8") as f:
            meta = json.load(f)
        # 老格式是纯标签数组；新格式带 input_dim，推理按它建头（训练侧 X.shape[1]）
        if isinstance(meta, dict):
            labels, input_dim = meta.get("labels"), int(meta.get("input_dim") or 0)
        else:
            labels, input_dim = meta, 1024
        if not labels or input_dim <= 0:
            raise ValueError(f"labels.json 内容不完整：{type(meta).__name__}")
        head = torch.nn.Linear(input_dim, len(labels))
        head.load_state_dict(torch.load(get_abs_path(f"{_MODEL_DIR}/{_HEAD_FILE}"), map_location="cpu"))
        head.eval()
    except Exception as e:  # noqa: BLE001
        logger.warning("[IntentRouter] 分类头不可用，降级为 unknown：%s", e)
        _head_unavailable = True
        return None, None

    _head, _labels = head, labels
    logger.info("[IntentRouter] classifier head loaded (%s)", get_abs_path(_LABELS_FILE))
    return _head, _labels


def route_intent_with_margin(query: str) -> tuple[str, float]:
    """同 route_intent，但额外返回 margin（top1 - top2 概率差）。

    margin 衡量分类头有多确定：接近 0 说明它自己也拿不准——典型是追问句，
    指代对象在上文，单看这一句没有任何信号。调用方据此做低置信判定，
    而不是把 argmax 的结果当成确定结论。
    """
    head, labels = _load_model()
    if head is None:
        # 显式降级：unknown 会走软引导 + RAG（加载失败已记 WARNING）
        return "unknown", 0.0
    try:
        import numpy as np
        import torch

        # 用 embed_documents（与训练同一条路径）而非 embed_query，
        # 保证训练与推理的 embedding 一致；向量走缓存，操作判别直接复用。
        vec = get_query_embedding(query)
        if vec is None:
            return "unknown", 0.0
        x = torch.tensor(np.array([vec], dtype=np.float32))
        with torch.no_grad():
            probs = torch.softmax(head(x), dim=1)[0]
    except Exception as e:  # noqa: BLE001
        # embedding / 前向异常（Ollama 抖动、重启）与加载失败同走降级，
        # 不能把整轮打成 500——unknown 走软引导 + RAG 是设计承诺
        logger.warning("[IntentRouter] 推理失败，降级为 unknown：%s", e)
        return "unknown", 0.0
    top2 = torch.topk(probs, k=2).values
    margin = float(top2[0] - top2[1])

    intent = labels[int(probs.argmax().item())]
    logger.info("[IntentRouter] local → %s (margin=%.3f, %s)", intent, margin, query[:40])
    return intent, margin


def route_intent(query: str) -> str:
    """把 query 分类为 robot | other | casual | unknown（本地模型）。"""
    return route_intent_with_margin(query)[0]


def get_guess_hint() -> str:
    """为 unknown 类 query 随机返回一条软引导语。"""
    return random.choice(_GUESS_HINTS)
