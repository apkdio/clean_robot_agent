"""意图分类评测：逐条断言 `route_intent` 的四分类标签。

口径以 `data/datasets/README.md`（2026-09-23 修订）为准，本评测集**不另立标准**。
评测集在 `data/eval/intent/`（**本地数据，已 gitignore**）。

与训练脚本内置测试集的区别：
  - **独立来源**：真实会话用户轮 + 人工清单 + 按口径构造；与训练集**零重叠**（本模块硬断言）
  - **固定可复现、带基线**：换分类头 / 换训练集即换基线（环境快照记两者指纹）
  - **逐条可归因**：混淆矩阵 + 按来源分组的准确率 + 「高置信错判」清单

本轨只评「这句话该判成什么」，**不评答得好不好**——答案质量属下游，另议。

指标：
  - 总体准确率与各类 P/R/F1（只统计 `status=confirmed`；`tentative` 单列、不计入）
  - 混淆矩阵（行=期望，列=实判）
  - 高置信错判：判错 **且** `margin ≥ intent.high_conf_margin` —— 模型很确信地错，最危险

运行（需 torch + Ollama bge-m3 + 已训练的分类头）：
  .venv\\Scripts\\python.exe test_scripts/eval/test_intent_eval.py --e2e
  .venv\\Scripts\\python.exe test_scripts/eval/test_intent_eval.py --e2e --save-baseline   # 存基线
  .venv\\Scripts\\python.exe test_scripts/eval/test_intent_eval.py --e2e --compare         # 对比基线
"""
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_SCRIPT_DIR) not in sys.path:   # test_scripts/：供 `_runner` 导入
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))
from _runner import *

_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))   # 上两级：test_scripts/eval/ → 项目根
_GOLDEN_FILE = os.path.join(_ROOT, "data", "eval", "intent", "golden.jsonl")
_BASELINE_FILE = os.path.join(_ROOT, "data", "eval", "intent", "baseline.json")
_PREV_BASELINE_FILE = os.path.join(_ROOT, "data", "eval", "intent", "baseline.prev.json")
_TRAIN_FILE = os.path.join(_ROOT, "data", "datasets", "intent_dataset.jsonl")
_HEAD_FILE = os.path.join(_ROOT, "data", "bgm_model", "classifier_head.pt")

# 与 intent_classifier_training/train_intent_classifier.py 的 _LABELS 同序
_LABELS = ("robot", "other", "casual", "unknown")


# ──────────────────────────────────────────────────────────────
# 数据与判定
# ──────────────────────────────────────────────────────────────

def _load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _conf_bands():
    """置信分档阈值：与线上同源（config/context.yaml），不在评测里另立标准。"""
    from tools.config_tool import load_config
    try:
        cfg = load_config("context") or {}
    except OSError as exc:
        print(f"  [注意] 读不到 context 配置（{exc}），回退默认阈值 0.2 / 0.5")
        return 0.2, 0.5
    intent = cfg.get("intent") or {}
    return float(intent.get("low_conf_margin", 0.2)), float(intent.get("high_conf_margin", 0.5))


def _evaluate(rows):
    from tools.intent_router import route_intent_with_margin
    records = []
    for row in rows:
        label, margin = route_intent_with_margin(row["query"])
        records.append({"id": row["id"], "query": row["query"], "expect": row["expect"],
                        "got": label, "margin": round(float(margin), 4),
                        "origin": row.get("origin", "unknown"), "note": row.get("note", ""),
                        "status": row.get("status", "confirmed"),
                        "ok": label == row["expect"]})
    return records


def _metrics(records):
    scored = [r for r in records if r["status"] == "confirmed"]
    n = len(scored)
    conf = {(e, g): 0 for e in _LABELS for g in _LABELS}
    for r in scored:
        conf[(r["expect"], r["got"])] += 1
    per = {}
    for lab in _LABELS:
        tp = conf[(lab, lab)]
        fp = sum(conf[(e, lab)] for e in _LABELS if e != lab)
        fn = sum(conf[(lab, g)] for g in _LABELS if g != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[lab] = {"n": tp + fn, "precision": prec, "recall": rec,
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    by_origin = defaultdict(lambda: [0, 0])
    for r in scored:
        by_origin[r["origin"]][1] += 1
        if r["ok"]:
            by_origin[r["origin"]][0] += 1
    return {"n": n, "accuracy": (sum(1 for r in scored if r["ok"]) / n) if n else 0.0,
            "per_class": per, "confusion": conf, "by_origin": dict(by_origin),
            "tentative": len(records) - n}


def _high_conf_errors(records, high):
    return [r for r in records if not r["ok"] and r["margin"] >= high]


# ──────────────────────────────────────────────────────────────
# 输出
# ──────────────────────────────────────────────────────────────

def _print_table(records):
    print("── 逐条明细 ──")
    for r in records:
        mark = "OK  " if r["ok"] else "FAIL"
        tag = "" if r["status"] == "confirmed" else "  [tentative]"
        print(f"  {mark} [{r['id']}] 期望 {r['expect']:<7} 实判 {r['got']:<7}"
              f" margin={r['margin']:.3f}{tag}  {r['query']}")


def _print_metrics(m):
    print()
    print("=" * 72)
    print("意图分类评测汇总")
    print("=" * 72)
    tail = f"，另有 tentative {m['tentative']} 条不计入" if m["tentative"] else ""
    print(f"  样本（confirmed）: {m['n']} 条{tail}")
    print(f"  总体准确率        : {m['accuracy'] * 100:.1f}%")
    print("  每类 P / R / F1   :")
    for lab in _LABELS:
        s = m["per_class"][lab]
        print(f"    {lab:<8} n={s['n']:<4} P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}")
    print("  按来源分组准确率  :")
    for origin in sorted(m["by_origin"]):
        hit, total = m["by_origin"][origin]
        print(f"    {origin:<14} {hit}/{total} = {hit / total * 100:.1f}%")
    print("  混淆矩阵（行=期望，列=实判）:")
    print("            " + "".join(f"{lab:>9}" for lab in _LABELS))
    for e in _LABELS:
        print(f"    {e:<8}" + "".join(f"{m['confusion'][(e, g)]:>9}" for g in _LABELS))
    print("=" * 72)


# ──────────────────────────────────────────────────────────────
# 环境快照与基线
# ──────────────────────────────────────────────────────────────

def _file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _env_meta():
    """**本轨的关键环境就是「哪个分类头 + 哪版训练集」** —— 任一变化即应视为新基线。"""
    def short(path):
        return _file_md5(path)[:8] if os.path.isfile(path) else "missing"
    return {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "head": short(_HEAD_FILE), "dataset": short(_TRAIN_FILE), "golden": short(_GOLDEN_FILE)}


def _baseline_payload(m, records):
    return {"accuracy": round(m["accuracy"], 4),
            "per_class": {lab: {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in m["per_class"][lab].items()} for lab in _LABELS},
            "ids": [r["id"] for r in records],
            "per_id": {r["id"]: r["got"] for r in records},
            "env": _env_meta()}


def _diff_baseline(records, base):
    """按 id 比实判标签；基线是旧格式（无 per_id）时返回 None。"""
    per_id = base.get("per_id")
    if not isinstance(per_id, dict):
        return None
    cur = {r["id"]: r["got"] for r in records}
    changed = sorted(i for i in cur if i in per_id and per_id[i] != cur[i])
    added = sorted(i for i in cur if i not in per_id)
    missing = sorted(i for i in per_id if i not in cur)
    return changed, added, missing


def _save_baseline(m, records):
    os.makedirs(os.path.dirname(_BASELINE_FILE), exist_ok=True)
    payload = _baseline_payload(m, records)
    rotated = os.path.isfile(_BASELINE_FILE)
    if rotated:
        shutil.copyfile(_BASELINE_FILE, _PREV_BASELINE_FILE)
    with open(_BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    env = payload["env"]
    print(f"\n[基线] 已写入 {_BASELINE_FILE}（{len(payload['ids'])} 条，准确率 {payload['accuracy']:.4f}）")
    print(f"       环境：分类头 {env['head']}；训练集 {env['dataset']}；评测集 {env['golden']}")
    if rotated:
        print(f"       上一版留存为 {_PREV_BASELINE_FILE}")


def _print_env_compare(env, base_env):
    print("\n[基线环境]")
    if not base_env:
        print("  基线未记录环境（旧格式）—— 跑一次 --save-baseline 即可核对分类头 / 训练集")
        return
    print(f"  基线采集于 {base_env.get('created_at', '?')}")
    for key, name in (("head", "分类头"), ("dataset", "训练集"), ("golden", "评测集")):
        same = env[key] == base_env.get(key)
        print(f"  {name:<6} : {env[key]}  vs  基线 {base_env.get(key, '?')}" + ("" if same else "  ⚠ 已变"))
    if env["head"] != base_env.get("head"):
        print("  ⚠ 分类头变了 —— 准确率差异首先归因于换模型，而不是评测集")


def _compare_baseline(m, records):
    if not os.path.isfile(_BASELINE_FILE):
        print(f"\n[基线] 无基线文件 {_BASELINE_FILE}，跳过对比")
        return
    with open(_BASELINE_FILE, encoding="utf-8") as f:
        base = json.load(f)
    print(f"\n[基线对比] 准确率 {m['accuracy']:.4f}  vs  {base.get('accuracy', float('nan')):.4f}"
          f"  ({m['accuracy'] - base.get('accuracy', 0.0):+.4f})")
    _print_env_compare(_env_meta(), base.get("env"))

    diff = _diff_baseline(records, base)
    if diff is None:
        print("\n[基线逐条] 基线是旧格式（无逐条结果），只能看聚合；跑一次 --save-baseline 后即可逐条比。")
        return
    changed, added, missing = diff
    def show(label, ids):
        if not ids:
            print(f"{label}：无")
        elif len(ids) <= 10:
            print(f"{label}：{'、'.join(ids)}")
        else:
            print(f"{label}：{'、'.join(ids[:10])} …（共 {len(ids)} 条）")
    print("\n[基线逐条]（按 id 比实判标签）")
    show("  标签变了", changed)
    show("  新增（基线无记录）", added)
    show("  未跑（基线有、本次没跑）", missing)


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────

def run():
    reset()
    if not E2E:
        _skip("意图分类评测需要 torch + Ollama bge-m3 + 已训练的分类头（加 --e2e 运行）")
        return stats()
    if not os.path.isfile(_GOLDEN_FILE):
        _skip(f"未找到评测集 {_GOLDEN_FILE}（本地数据目录 data/eval，已 gitignore）")
        return stats()
    if not os.path.isfile(_HEAD_FILE):
        _skip(f"未找到分类头 {_HEAD_FILE}（由 intent_classifier_training/train_intent_classifier.py 产出）")
        return stats()

    rows = _load_jsonl(_GOLDEN_FILE)
    _assert(len(rows) > 0, f"评测集加载成功（{len(rows)} 条）")

    # 与训练集重叠 = 拿训练样本自证，准确率会虚高 → 硬失败（不是"测量"）
    train_texts = {r["text"] for r in _load_jsonl(_TRAIN_FILE)}
    _assert(len(train_texts) > 0, f"训练集可读（{len(train_texts)} 条，用于重叠检查）")
    overlap = sorted({r["query"] for r in rows if r["query"] in train_texts})
    _assert(not overlap, f"评测集与训练集零重叠（重叠 {len(overlap)} 条：{'、'.join(overlap[:5])}）")

    bad = sorted({r["expect"] for r in rows} - set(_LABELS))
    _assert(not bad, f"期望标签都在闭集内（越界：{bad}）")
    thin = [lab for lab in _LABELS if not any(r["expect"] == lab and r.get("status", "confirmed") == "confirmed"
                                              for r in rows)]
    _assert(not thin, f"每类都有 confirmed 样本（缺：{thin}）")

    low, high = _conf_bands()
    print(f"\n置信阈值：low_conf_margin={low}  high_conf_margin={high}（取自 config/context.yaml）")

    records = _evaluate(rows)
    _print_table(records)
    m = _metrics(records)
    _print_metrics(m)

    errs = _high_conf_errors(records, high)
    print(f"\n高置信错判（判错且 margin ≥ {high}）：{len(errs)} 条")
    for r in errs:
        print(f"  {r['id']}  期望 {r['expect']:<7} 实判 {r['got']:<7} margin={r['margin']:.3f}  {r['query']}")

    if "--save-baseline" in sys.argv:
        _save_baseline(m, records)
    if "--compare" in sys.argv:
        _compare_baseline(m, records)

    return stats()


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
