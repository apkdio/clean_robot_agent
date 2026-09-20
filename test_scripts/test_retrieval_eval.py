"""检索质量评测：golden set + 核心指标（P1-1）。

评测集位于 `data/eval/`（**本地数据目录，已 gitignore，不随仓库发布**）：
  - `retrieval_golden.jsonl`：每条 `{query, type, ...}`，覆盖五类：
      - domain        ：域路由 + 域过滤检索（expect_file + expect_keyword）
      - model         ：型号精准（expect_model，检索具体型号域）
      - budget        ：预算过滤（expect_models，结构化直出）
      - time          ：时间过滤（expect_models，结构化直出）
      - out_of_domain ：领域外（期望零召回）
  - `retrieval_baseline.json`：基线快照（`--save-baseline` 写、`--compare` 读）
  - `README.md`：构建准则与扩充方法（同目录本地文档）

评测集缺失时本模块自动跳过、不报错，不影响 `run_tests.py` 其它模块。

指标：
  - 域路由准确率      ：domain 类 query 的 _route_domain 命中 expect_file 比例
  - hit@1 / @3 / @5  ：检索轨（domain + model）中正确 chunk 进入前 k 的比例
  - MRR               ：检索轨的平均倒数排名
  - 无召回率          ：out_of_domain 类中召回为 0 的比例（越高越好）
  - 结构化直出命中率  ：budget + time 类中 enumerate_models 结果精确匹配期望型号集合的比例

运行（需 Ollama + Chroma + 本地 reranker 模型）：
  .venv\\Scripts\\python.exe test_retrieval_eval.py --e2e
  .venv\\Scripts\\python.exe test_retrieval_eval.py --e2e --save-baseline   # 保存基线
  .venv\\Scripts\\python.exe test_retrieval_eval.py --e2e --compare         # 对比基线
"""
import json
import os
import sys
from collections import defaultdict
from _runner import *

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GOLDEN_FILE = os.path.join(_ROOT, "data", "eval", "retrieval_golden.jsonl")
_BASELINE_FILE = os.path.join(_ROOT, "data", "eval", "retrieval_baseline.json")

_BRAND_PREFIX = "不染一尘"
_MODEL_FILE = "不染一尘具体型号.txt"


# ──────────────────────────────────────────────────────────────
# 数据与辅助
# ──────────────────────────────────────────────────────────────

def _load_golden():
    entries = []
    with open(_GOLDEN_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _short_name(name):
    return (name or "").replace(_BRAND_PREFIX, "").strip()


def _first_rank(docs, pred):
    """返回第一个满足 pred 的 doc 的 1-based 排名；未命中返回 None。"""
    for i, c in enumerate(docs, 1):
        if pred(c):
            return i
    return None


def _hit_at_k(ranks, ks=(1, 3, 5)):
    """ranks: list[int|None]。返回 {k: 命中率}。"""
    n = len(ranks)
    if n == 0:
        return {k: 0.0 for k in ks}
    return {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}


def _mrr(ranks):
    n = len(ranks)
    if n == 0:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / n


def _fmt(ratio):
    return f"{ratio * 100:.1f}%"


def _compute(entries, hr):
    from tools.agent import _route_domain, _resolve_date_filter
    from tools.metadata_extractor import enumerate_models, resolve_budget_filter

    records = []
    ranks = []                 # 检索轨（domain + model）每条 rank
    route_correct = route_total = 0
    struct_hits = struct_total = 0
    ood_zero = ood_total = 0
    strict_ood_fail = []       # 应零召回却召回了的严格领域外

    for e in entries:
        q, t = e["query"], e["type"]
        rec = {"query": q, "type": t, "ok": False, "detail": ""}

        if t == "domain":
            route_total += 1
            route = _route_domain(q)
            route_ok = route == e["expect_file"]
            if route_ok:
                route_correct += 1
            # 忠实于 agent 实际行为：域路由命中则域过滤，识别不准则全库兜底
            docs = hr.search(q, filter={"file_name": route} if route else None)
            rank = _first_rank(docs, lambda c: e["expect_keyword"] in c.page_content)
            ranks.append(rank)
            rec["ok"] = rank is not None and rank <= 5
            rec["detail"] = (
                f"路由{'OK' if route_ok else 'FAIL(' + str(route) + ')'}，"
                f"期望关键词 rank={rank}，召回 {len(docs)} 条"
            )

        elif t == "model":
            docs = hr.search(q, filter={"file_name": _MODEL_FILE})
            rank = _first_rank(docs, lambda c: e["expect_model"] in c.page_content)
            ranks.append(rank)
            rec["ok"] = rank is not None and rank <= 5
            rec["detail"] = f"型号 rank={rank}，召回 {len(docs)} 条"

        elif t == "out_of_domain":
            ood_total += 1
            docs = hr.search(q)
            recall = len(docs)
            if recall == 0:
                ood_zero += 1
            rec["ok"] = recall == 0
            rec["detail"] = f"召回 {recall} 条"
            if e.get("strict") and recall > 0:
                strict_ood_fail.append(q)

        elif t == "budget":
            struct_total += 1
            f = resolve_budget_filter(q)
            models = enumerate_models(f) if f is not None else []
            names = {_short_name(m["name"]) for m in models}
            rec["ok"] = names == set(e["expect_models"])
            if rec["ok"]:
                struct_hits += 1
            rec["detail"] = f"命中 {sorted(names)}"

        elif t == "time":
            struct_total += 1
            f = _resolve_date_filter(q)
            models = enumerate_models(f) if f is not None else []
            names = {_short_name(m["name"]) for m in models}
            rec["ok"] = names == set(e["expect_models"])
            if rec["ok"]:
                struct_hits += 1
            rec["detail"] = f"命中 {sorted(names)}"

        records.append(rec)

    metrics = {
        "route_accuracy": (route_correct, route_total),
        "hit_at_k": _hit_at_k(ranks, (1, 3, 5)),
        "mrr": _mrr(ranks),
        "no_recall_rate": (ood_zero, ood_total),
        "structured_hit_rate": (struct_hits, struct_total),
    }
    return records, metrics, strict_ood_fail


def _print_table(records):
    print("── 逐条明细 ──")
    for r in records:
        mark = "OK  " if r["ok"] else "FAIL"
        print(f"  {mark} [{r['type']:<13}] {r['query']}  → {r['detail']}")


def _print_metrics(metrics):
    print()
    print("=" * 72)
    print("检索质量评测汇总")
    print("=" * 72)
    rc, rt = metrics["route_accuracy"]
    hit = metrics["hit_at_k"]
    zr, zt = metrics["no_recall_rate"]
    sh, st = metrics["structured_hit_rate"]
    print(f"  域路由准确率      : {rc}/{rt} = {_fmt(rc / rt) if rt else 'N/A'}")
    print(f"  hit@1 / @3 / @5   : {_fmt(hit[1])} / {_fmt(hit[3])} / {_fmt(hit[5])}")
    print(f"  MRR               : {metrics['mrr']:.4f}")
    print(f"  无召回率          : {zr}/{zt} = {_fmt(zr / zt) if zt else 'N/A'}")
    print(f"  结构化直出命中率  : {sh}/{st} = {_fmt(sh / st) if st else 'N/A'}")
    print("=" * 72)


def _baseline_payload(metrics):
    return {
        "route_accuracy": {"correct": metrics["route_accuracy"][0], "total": metrics["route_accuracy"][1]},
        "hit_at_1": round(metrics["hit_at_k"][1], 4),
        "hit_at_3": round(metrics["hit_at_k"][3], 4),
        "hit_at_5": round(metrics["hit_at_k"][5], 4),
        "mrr": round(metrics["mrr"], 4),
        "no_recall_rate": {"zero": metrics["no_recall_rate"][0], "total": metrics["no_recall_rate"][1]},
        "structured_hit_rate": {"hit": metrics["structured_hit_rate"][0], "total": metrics["structured_hit_rate"][1]},
    }


def _save_baseline(metrics):
    os.makedirs(os.path.dirname(_BASELINE_FILE), exist_ok=True)
    with open(_BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(_baseline_payload(metrics), f, ensure_ascii=False, indent=2)
    print(f"\n[基线] 已写入 {_BASELINE_FILE}")


def _compare_baseline(metrics):
    if not os.path.isfile(_BASELINE_FILE):
        print(f"\n[基线] 无基线文件 {_BASELINE_FILE}，跳过对比")
        return
    with open(_BASELINE_FILE, encoding="utf-8") as f:
        base = json.load(f)
    cur = _baseline_payload(metrics)
    print("\n[基线对比]（当前 - 基线，负值=回退）")
    print(f"  hit@1   : {cur['hit_at_1']:.4f}  vs  {base['hit_at_1']:.4f}  ({cur['hit_at_1'] - base['hit_at_1']:+.4f})")
    print(f"  hit@3   : {cur['hit_at_3']:.4f}  vs  {base['hit_at_3']:.4f}  ({cur['hit_at_3'] - base['hit_at_3']:+.4f})")
    print(f"  hit@5   : {cur['hit_at_5']:.4f}  vs  {base['hit_at_5']:.4f}  ({cur['hit_at_5'] - base['hit_at_5']:+.4f})")
    print(f"  MRR     : {cur['mrr']:.4f}  vs  {base['mrr']:.4f}  ({cur['mrr'] - base['mrr']:+.4f})")
    print(f"  无召回率: {cur['no_recall_rate']['zero']}/{cur['no_recall_rate']['total']}  vs  {base['no_recall_rate']['zero']}/{base['no_recall_rate']['total']}")
    print(f"  结构化直出: {cur['structured_hit_rate']['hit']}/{cur['structured_hit_rate']['total']}  vs  {base['structured_hit_rate']['hit']}/{base['structured_hit_rate']['total']}")


def run():
    reset()
    if not E2E:
        _skip("检索质量评测需要 Ollama + Chroma + reranker 模型（加 --e2e 运行）")
        return stats()

    # 评测集是本地数据（data/eval 已 gitignore）：缺失则跳过，不阻塞聚合运行
    if not os.path.isfile(_GOLDEN_FILE):
        _skip(f"未找到评测集 {_GOLDEN_FILE}（本地数据目录 data/eval，已 gitignore）")
        return stats()

    entries = _load_golden()
    _assert(len(entries) > 0, f"评测集加载成功（{len(entries)} 条）")

    from tools.metadata_extractor import get_all_models
    _assert(len(get_all_models()) == 16, "型号数据就绪（16 个型号）")

    from tools.hybrid_retriever import HybridRetriever
    hr = HybridRetriever()
    hr.ensure_sparse_index()

    records, metrics, strict_ood_fail = _compute(entries, hr)
    _print_table(records)
    _print_metrics(metrics)

    # 回归护栏：严格领域外 query 应零召回（依赖 P0-1 阈值 + P0-2 精排）
    for q in strict_ood_fail:
        _assert(False, f"严格领域外应零召回: {q}")
    if not strict_ood_fail:
        _assert(True, "严格领域外 query 均零召回")

    if "--save-baseline" in sys.argv:
        _save_baseline(metrics)
    if "--compare" in sys.argv:
        _compare_baseline(metrics)

    return stats()


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
