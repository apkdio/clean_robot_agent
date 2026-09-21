"""意图路由：用本地 bge-m3 + 训练好的分类头判断用户意图。

四分类头（Linear 1024→4）由 data/datasets/ 训练得到，保存在 data/bgm_model/。
推理时用 Ollama bge-m3 把 query 向量化，再跑分类头前向——远比 LLM 分类快。

标签：robot / other / casual / unknown。
"""

from __future__ import annotations

import json
import random

from log_tool import get_logger

logger = get_logger(name="intent_router")

_MODEL_DIR = "data/bgm_model"
_HEAD_FILE = "classifier_head.pt"
_LABELS_FILE = "labels.json"

_head = None
_labels = None
_emb = None


def _get_emb():
    """懒加载 bge-m3 embedding 客户端（跨调用缓存复用）。"""
    global _emb
    if _emb is None:
        from llm_tool import get_embedding_model
        _emb = get_embedding_model()
    return _emb


_GUESS_HINTS = [
    "我猜你可能想问扫地机器人相关的问题吧，帮你看看～",
    "我主要擅长扫地机器人，先按机器人帮你查一下吧～",
    "这个我不太确定是不是机器人的问题，帮你找找相关答案～",
]


def _load_model():
    """懒加载训练好的分类头和标签映射。"""
    global _head, _labels
    if _head is not None:
        return _head, _labels

    import torch
    from path_tool import get_abs_path

    head_path = get_abs_path(f"{_MODEL_DIR}/{_HEAD_FILE}")
    labels_path = get_abs_path(f"{_MODEL_DIR}/{_LABELS_FILE}")

    _head = torch.nn.Linear(1024, 4)
    _head.load_state_dict(torch.load(head_path, map_location="cpu"))
    _head.eval()

    with open(labels_path, "r", encoding="utf-8") as f:
        _labels = json.load(f)

    logger.info("[IntentRouter] classifier head loaded (%s)", labels_path)
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
