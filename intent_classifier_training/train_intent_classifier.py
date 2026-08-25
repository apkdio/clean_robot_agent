"""在 bge-m3 embedding 之上训练一个 4 分类的意图分类头。

流程：
  1. 加载 data/datasets/intent_dataset.jsonl
  2. 使用本地 Ollama bge-m3（1024 维）对每个样本进行 embedding
  3. 划分训练集/测试集（80/20）
  4. 使用 torch 训练一个 Linear(1024 → 4) 分类头
  5. 评估准确率 + 每个类别的 F1
  6. 将分类头权重与标签映射保存到 data/bgm_model/
"""

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from tools.llm_tool import get_embedding_model

_DATASET_PATH = "data/datasets/intent_dataset.jsonl"
_MODEL_DIR = "data/bgm_model"
_LABELS = ["robot", "other", "casual", "unknown"]  # 固定顺序

_SEED = 42
_EPOCHS = 500
_LR = 1e-3
_BATCH = 64
_TEST_RATIO = 0.2


def _abs(p):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p)


def load_dataset():
    rows = []
    with open(_abs(_DATASET_PATH), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)

    rows = load_dataset()
    texts = [r["text"] for r in rows]
    labels = [_LABELS.index(r["label"]) for r in rows]
    print(f"数据集: {len(rows)} 条")

    # 通过 Ollama bge-m3 对所有样本进行 embedding（分批处理以避免分词崩溃）
    print("生成 bge-m3 embedding...")
    emb = get_embedding_model()
    embs = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        embs.extend(emb.embed_documents(batch))
        if (i // 64 + 1) % 3 == 0 or i + 64 >= len(texts):
            print(f"  embedded {min(i + 64, len(texts))}/{len(texts)}")
    X = np.array(embs, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    print(f"embedding 形状: {X.shape}")

    # 分层划分训练/测试集：确保每个类别都同时出现在两个集合中
    n = len(rows)
    train_idx, test_idx = [], []
    for label in range(len(_LABELS)):
        label_idx = [i for i in range(n) if y[i] == label]
        random.shuffle(label_idx)
        n_test_label = max(1, int(len(label_idx) * _TEST_RATIO))
        test_idx.extend(label_idx[:n_test_label])
        train_idx.extend(label_idx[n_test_label:])
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"train={len(train_idx)} test={len(test_idx)}")

    # 模型：Linear(1024 -> 4)
    model = nn.Linear(X.shape[1], len(_LABELS))
    # 类别平衡权重，用于抵消 robot 类样本占比过高带来的影响。
    # 防止出现零样本类别（否则权重会爆炸）。
    class_counts = np.bincount(y_train, minlength=len(_LABELS)).astype(np.float32)
    safe_counts = np.where(class_counts > 0, class_counts, 1.0)
    class_weights = safe_counts.sum() / (len(_LABELS) * safe_counts)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
    opt = torch.optim.Adam(model.parameters(), lr=_LR)

    Xt = torch.tensor(X_train)
    yt = torch.tensor(y_train)

    print("训练中...")
    for epoch in range(_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt))
        total_loss = 0.0
        for i in range(0, len(Xt), _BATCH):
            bi = perm[i:i + _BATCH]
            xb, yb = Xt[bi], yt[bi]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{_EPOCHS}  loss={total_loss/len(Xt):.4f}")

    # 评估
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        pred = logits.argmax(dim=1).numpy()

    acc = (pred == y_test).mean()
    print(f"\n测试集准确率: {acc:.4f}")

    # 每个类别的 F1
    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for p, t in zip(pred, y_test):
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    print("各类 F1:")
    for i, name in enumerate(_LABELS):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        rec = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {name:8s}  F1={f1:.4f}  (tp={tp[i]}, fp={fp[i]}, fn={fn[i]})")

    # 保存模型 + 标签映射
    os.makedirs(_abs(_MODEL_DIR), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(_abs(_MODEL_DIR), "classifier_head.pt"))
    with open(os.path.join(_abs(_MODEL_DIR), "labels.json"), "w", encoding="utf-8") as f:
        json.dump(_LABELS, f, ensure_ascii=False)
    print(f"\n模型已保存到 {_abs(_MODEL_DIR)}")


if __name__ == "__main__":
    main()
