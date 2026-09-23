"""在 bge-m3 embedding 之上训练一个 4 分类的意图分类头。

流程：
  1. 加载 data/datasets/intent_dataset.jsonl（字段口径见 data/datasets/README.md）
  2. 使用本地 Ollama bge-m3（1024 维）对每个样本进行 embedding
  3. 划分训练集/测试集：按类别分层抽 20%；`multiturn_real`（真实会话追问）全部进测试集
  4. 训练 Linear(1024 → 4)，用训练集里再切出的 15% 验证集选最佳权重
  5. 评估准确率 + 每个类别的 F1 + **按来源分组的准确率**
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
    sources = [r.get("source", "base") for r in rows]
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

    # 分层划分训练/测试集：确保每个类别都同时出现在两个集合中。
    # 例外：source=multiturn_real 的样本（真实会话里的追问）**全部划入测试集**——
    # 它们是最贴近线上多轮场景的一小撮，混进训练集就等于拿同一批样本自证。
    n = len(rows)
    train_idx, test_idx = [], []
    for label in range(len(_LABELS)):
        label_idx = [i for i in range(n) if y[i] == label]
        forced = [i for i in label_idx if sources[i] == "multiturn_real"]
        pool = [i for i in label_idx if sources[i] != "multiturn_real"]
        random.shuffle(pool)
        n_test_label = max(0, max(1, int(len(label_idx) * _TEST_RATIO)) - len(forced))
        test_idx.extend(forced + pool[:n_test_label])
        train_idx.extend(pool[n_test_label:])
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    src_test = [sources[i] for i in test_idx]
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

    # 从训练集里再留 15% 做验证集，用它选最佳权重。
    # 数据集只有几百条、头又是单层线性，跑满 500 轮 loss 降得很低（≈0.07）实际上是在背样本，
    # 边界会随数据小幅增删来回摆；选验证损失最低的那一版能明显压住这种抖动。
    order = np.random.permutation(len(y_train))
    n_val = max(1, int(len(y_train) * 0.15))
    val_idx, tr_idx = order[:n_val], order[n_val:]
    Xt = torch.tensor(X_train[tr_idx])
    yt = torch.tensor(y_train[tr_idx])
    Xv = torch.tensor(X_train[val_idx])
    yv = torch.tensor(y_train[val_idx])
    print(f"train={len(tr_idx)} val={len(val_idx)}")

    print("训练中...")
    best_loss, best_state = float("inf"), None
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
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xv), yv))
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{_EPOCHS}  loss={total_loss/len(Xt):.4f}  val={vloss:.4f}")

    model.load_state_dict(best_state)
    print(f"选用验证损失最低的权重: val={best_loss:.4f}")

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

    # 按来源分组：multiturn_real 是唯一没有被训练过的真实多轮追问，最值得看
    print("按来源分组的测试集准确率:")
    for s in sorted(set(src_test)):
        hit = [k for k in range(len(y_test)) if src_test[k] == s]
        if not hit:
            continue
        acc_s = sum(1 for k in hit if pred[k] == y_test[k]) / len(hit)
        print(f"  {s:16s} n={len(hit):3d}  acc={acc_s:.4f}")

    # 保存模型 + 标签映射
    os.makedirs(_abs(_MODEL_DIR), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(_abs(_MODEL_DIR), "classifier_head.pt"))
    with open(os.path.join(_abs(_MODEL_DIR), "labels.json"), "w", encoding="utf-8") as f:
        json.dump(_LABELS, f, ensure_ascii=False)
    print(f"\n模型已保存到 {_abs(_MODEL_DIR)}")


if __name__ == "__main__":
    main()
