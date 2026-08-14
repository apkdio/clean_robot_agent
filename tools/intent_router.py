"""Intent router: classify query with a local bge-m3 + trained classifier head.

The 4-class head (Linear 1024→4) was trained on data/datasets/ and saved to
data/bgm_model/. At inference we embed the query with Ollama bge-m3 and run
the head forward — far faster than an LLM classifier.

Labels: robot / other / casual / unknown.
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
    """Lazily init the bge-m3 embedding client (cached across calls)."""
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
    """Lazily load the trained classifier head and label mapping."""
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


def route_intent(query: str) -> str:
    """Classify query → robot | other | casual | unknown (local model)."""
    import numpy as np
    import torch

    head, labels = _load_model()
    emb = _get_emb()
    # Use embed_documents (same path as training) rather than embed_query,
    # to keep the embeddings consistent between train and inference.
    vec = emb.embed_documents([query])[0]

    x = torch.tensor(np.array([vec], dtype=np.float32))
    with torch.no_grad():
        logits = head(x)
        idx = int(logits.argmax(dim=1).item())

    intent = labels[idx]
    logger.info("[IntentRouter] local → %s (%s)", intent, query[:40])
    return intent


def get_guess_hint() -> str:
    """Return a random soft-guess hint for 'unknown' queries."""
    return random.choice(_GUESS_HINTS)
