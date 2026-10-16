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


# 软引导话术（_GUESS_HINTS）见 config/word_dict_config.py


def _load_model():
    """懒加载训练好的分类头和标签映射；不可用时返回 (None, None)，由调用方显式降级。"""
    global _head, _labels, _head_unavailable
    if _head is not None:
        return _head, _labels
    if _head_unavailable:
        return None, None

    import torch
    from tools.path_tool import get_abs_path

    # 先在局部装配完再赋值：失败时不留半个状态
    head = torch.nn.Linear(1024, 4)
    try:
        head.load_state_dict(torch.load(get_abs_path(f"{_MODEL_DIR}/{_HEAD_FILE}"), map_location="cpu"))
        head.eval()
        with open(get_abs_path(f"{_MODEL_DIR}/{_LABELS_FILE}"), encoding="utf-8") as f:
            labels = json.load(f)
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
    import numpy as np
    import torch

    head, labels = _load_model()
    if head is None:
        # 显式降级：unknown 会走软引导 + RAG（加载失败已记 WARNING）
        return "unknown", 0.0
    emb = _get_emb()
    # 用 embed_documents（与训练同一条路径）而非 embed_query，
    # 保证训练与推理的 embedding 一致。
    vec = emb.embed_documents([query])[0]

    x = torch.tensor(np.array([vec], dtype=np.float32))
    with torch.no_grad():
        probs = torch.softmax(head(x), dim=1)[0]
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
