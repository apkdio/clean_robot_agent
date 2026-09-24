"""检索质量评测：golden set + 核心指标（P1-1）。

评测集位于 `data/eval/retrieval/`（**本地数据目录，已 gitignore，不随仓库发布**）：
  - `golden.jsonl`：每条 `{id, query, type, ...}`，覆盖五类：
      - domain        ：域路由 + 域过滤检索（expect_file + expect_keyword）
      - model         ：型号精准（expect_model，检索具体型号域）
      - budget        ：预算过滤（expect_models，结构化直出）
      - time          ：时间过滤（expect_models，结构化直出）
      - out_of_domain ：领域外（期望零召回）
  - `baseline.json`：基线快照（`--save-baseline` 写、`--compare` 读），记录覆盖的 id 段、逐条结果，
    以及采集时的环境（代码提交 + 未提交改动、知识库指纹、配置指纹与关键阈值、索引是否跟上知识库）
  - `README.md`：构建准则与扩充方法（同目录本地文档）

id 是版本锚点：评测集**只增不删**，因此 id 单调递增、一次分配永不变更 —— **id 区间即批次**
（批次 1 = R001-R070），新增条目从 R071 起。`--only` 按区间跑，`--compare` 按 id 逐条回归。

评测集缺失时本模块自动跳过、不报错，不影响 `run_tests.py` 其它模块。

指标：
  - 域路由准确率      ：domain 类 query 的 top1 域命中 expect_file 比例（口径同旧基线）
  - 域覆盖命中率      ：实际搜索的文件（含双域 $in）包含 expect_file 的比例
  - hit@1 / @3 / @5  ：检索轨（domain + model）中正确 chunk 进入前 k 的比例
  - MRR               ：检索轨的平均倒数排名
  - 无召回率          ：out_of_domain 类中召回为 0 的比例（越高越好）
  - 结构化直出命中率  ：budget + time 类中 enumerate_models 结果精确匹配期望型号集合的比例

运行（需 Ollama + Chroma + 本地 reranker 模型）：
  .venv\\Scripts\\python.exe test_retrieval_eval.py --e2e
  .venv\\Scripts\\python.exe test_retrieval_eval.py --e2e --save-baseline   # 保存基线（旧版自动转 .prev.json）
  .venv\\Scripts\\python.exe test_retrieval_eval.py --e2e --compare         # 对比基线（按 id 逐条）
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from _runner import *

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GOLDEN_FILE = os.path.join(_ROOT, "data", "eval", "retrieval", "golden.jsonl")
_BASELINE_FILE = os.path.join(_ROOT, "data", "eval", "retrieval", "baseline.json")
_PREV_BASELINE_FILE = os.path.join(_ROOT, "data", "eval", "retrieval", "baseline.prev.json")

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


# ──────────────────────────────────────────────────────────────
# id：版本锚点（只增不删 → id 区间即批次）
# ──────────────────────────────────────────────────────────────

def _id_num(entry_id):
    """`R012` → 12；非本格式返回 None。"""
    m = re.fullmatch(r"R(\d+)", entry_id or "")
    return int(m.group(1)) if m else None


def _id_sort_key(entry_id):
    return _id_num(entry_id) or 0


def _parse_id_spec(spec):
    """`R001-R070` / `R005` / `R001-R010,R020` → [(lo, hi), ...]（闭区间，按数字比较）。"""
    ranges = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        lo_s, sep, hi_s = part.partition("-")
        lo = _id_num(lo_s)
        hi = _id_num(hi_s) if sep else lo
        if lo is None or hi is None:
            raise ValueError(f"非法的 id 写法：{part}（应为 R001-R070 或 R005）")
        ranges.append((min(lo, hi), max(lo, hi)))
    if not ranges:
        raise ValueError("--only 未给出有效 id")
    return ranges


def _filter_by_id(entries, spec):
    ranges = _parse_id_spec(spec)
    return [e for e in entries
            if _id_num(e.get("id")) is not None
            and any(lo <= _id_num(e["id"]) <= hi for lo, hi in ranges)]


def _validate_ids(entries):
    """id 必须存在且唯一 —— 新增条目最常见的手误是「复制上一行忘了改 id」。"""
    ids = [e.get("id") for e in entries]
    missing = sum(1 for i in ids if not i)
    dup = sorted((i for i, n in Counter(ids).items() if i and n > 1), key=_id_sort_key)
    return missing, dup


def _arg_value(flag):
    """取 `--flag value` 形式的参数值（无值时返回 None）。"""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _short_name(name):
    return (name or "").replace(_BRAND_PREFIX, "").strip()


def _searched_files(domain_value):
    """把 `_domain_filter` 的取值还原成被搜索的文件集合（字符串 / `$in` / None）。"""
    if not domain_value:
        return set()
    if isinstance(domain_value, dict):
        return set(domain_value.get("$in") or [])
    return {domain_value}


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
    from tools.agent import _domain_filter, _resolve_date_filter, _route_domain
    from tools.metadata_extractor import enumerate_models, resolve_budget_filter

    records = []
    ranks = []                 # 检索轨（domain + model）每条 rank
    route_correct = route_total = 0
    route_covered = 0
    via_count = defaultdict(int)
    struct_hits = struct_total = 0
    ood_zero = ood_total = 0
    strict_ood_fail = []       # 应零召回却召回了的严格领域外

    for e in entries:
        q, t = e["query"], e["type"]
        rec = {"id": e.get("id", "?"), "query": q, "type": t, "ok": False, "detail": ""}

        if t == "domain":
            route_total += 1
            # 忠实于 agent 的**实际检索路径**：走 _domain_filter（top1 领先则单域；
            # 两域咬得近则 $in 双域；无域则全库兜底），而不是旧的单文件 filter。
            domain_value, via = _domain_filter(q)
            route = _route_domain(q)
            route_ok = route == e["expect_file"]           # top1 命中（口径与旧基线一致）
            covered = e["expect_file"] in _searched_files(domain_value)
            if route_ok:
                route_correct += 1
            if covered:
                route_covered += 1
            via_count[via] += 1
            docs = hr.search(q, filter={"file_name": domain_value} if domain_value else None)
            rank = _first_rank(docs, lambda c: e["expect_keyword"] in c.page_content)
            ranks.append(rank)
            rec["ok"] = rank is not None and rank <= 5
            rec["detail"] = (
                f"路由{'OK' if route_ok else 'FAIL(' + str(route) + ')'}"
                f"{'' if covered else '｜未覆盖期望文件'}({via})，"
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
        "route_recall": (route_covered, route_total),
        "route_via": dict(via_count),
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
        print(f"  {mark} [{r['id']}] [{r['type']:<13}] {r['query']}  → {r['detail']}")


def _print_metrics(metrics):
    print()
    print("=" * 72)
    print("检索质量评测汇总")
    print("=" * 72)
    rc, rt = metrics["route_accuracy"]
    vc, _vt = metrics["route_recall"]
    hit = metrics["hit_at_k"]
    zr, zt = metrics["no_recall_rate"]
    sh, st = metrics["structured_hit_rate"]
    print(f"  域路由准确率(top1): {rc}/{rt} = {_fmt(rc / rt) if rt else 'N/A'}")
    print(f"  域覆盖命中率      : {vc}/{rt} = {_fmt(vc / rt) if rt else 'N/A'}"
          f"（实际搜索文件含期望文件）  路径分布 {metrics['route_via']}")
    print(f"  hit@1 / @3 / @5   : {_fmt(hit[1])} / {_fmt(hit[3])} / {_fmt(hit[5])}")
    print(f"  MRR               : {metrics['mrr']:.4f}")
    print(f"  无召回率          : {zr}/{zt} = {_fmt(zr / zt) if zt else 'N/A'}")
    print(f"  结构化直出命中率  : {sh}/{st} = {_fmt(sh / st) if st else 'N/A'}")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────
# 环境快照：基线记的是「某版代码 + 某版知识库 + 某套配置」下的表现
# ──────────────────────────────────────────────────────────────

_CONFIG_FILES = ("config/rag.yaml", "config/chroma.yaml", "config/agent.yaml", "config/context.yaml")


def _file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _digest(pairs):
    """把 (名称, md5) 序列折成一个短指纹 —— 任一项变化都会改变它。"""
    text = "\n".join(f"{k}:{v}" for k, v in sorted(pairs))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _git(*args):
    """跑一条 git 命令，返回**原始** stdout（不做 strip：porcelain 首行行首是状态位）；失败返回 None。"""
    try:
        out = subprocess.run(["git", "-C", _ROOT, *args], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _code_label(code):
    n = len(code.get("changed") or [])
    return f"{code.get('commit', '?')}" + (f"（+{n} 未提交）" if n else "（干净）")


def _code_meta():
    """代码工程状态：提交号 + 未提交改动清单（本项目常在未提交状态下跑评测，必须记下来）。"""
    porcelain = _git("status", "--porcelain") or ""
    # porcelain 每行是 `XY path`：行首两个状态位不能先 strip 掉，否则路径第一个字符被吃掉
    files = sorted(line[3:].strip() for line in porcelain.splitlines() if len(line) > 3)
    commit = (_git("rev-parse", "--short", "HEAD") or "").strip() or "unknown"
    return {"commit": commit, "dirty": bool(files), "changed": files}


def _kb_meta():
    """知识库内容指纹（复用热更新的 scan_files：逐文件 md5 汇总）。"""
    from tools.config_tool import get_data_dir
    from tools.hot_ingest import scan_files

    kb_dir = get_data_dir()
    files = scan_files(kb_dir) if os.path.isdir(kb_dir) else {}
    return {"dir": os.path.relpath(kb_dir, _ROOT).replace("\\", "/"),
            "files": len(files), "fingerprint": _digest(files.items())}


def _config_meta():
    """行为相关配置：指纹 + 关键旋钮现值（阈值/开关改一下，指标就变）。"""
    from tools.config_tool import load_config

    pairs = [(rel, _file_md5(os.path.join(_ROOT, rel)))
             for rel in _CONFIG_FILES if os.path.isfile(os.path.join(_ROOT, rel))]
    rag = load_config("rag")
    key = {"retrieval.score_threshold": rag["retrieval"]["score_threshold"],
           "retrieval.final_top_k": rag["retrieval"]["final_top_k"],
           "retrieval.domain_margin": rag["retrieval"].get("domain_margin", 0),
           "rerank.enabled": rag["rerank"]["enabled"],
           "rerank.score_threshold": rag["rerank"]["score_threshold"],
           "rerank.candidate_top_k": rag["rerank"]["candidate_top_k"],
           "chunk.chunk_size": rag["chunk"]["chunk_size"]}
    return {"files": [rel for rel, _ in pairs], "fingerprint": _digest(pairs), "key": key}


def _index_meta():
    """索引有没跟上知识库：拿热更新的 md5 快照（data/state/ingest_snapshot.json）与当前知识库比。

    不用文件 mtime —— Chroma 读一次也会动文件时间，mtime 判不出「索引落后」。
    """
    from tools.config_tool import get_data_dir
    from tools.hot_ingest import diff_snapshot, load_snapshot, scan_files

    snapshot = load_snapshot()
    if not snapshot:
        return {"in_sync": None, "note": "无 ingest 快照（data/state/ingest_snapshot.json）"}
    added, changed, removed = diff_snapshot(snapshot, scan_files(get_data_dir()))
    if not (added or changed or removed):
        return {"in_sync": True}
    return {"in_sync": False,
            "pending": {"added": sorted(added), "changed": sorted(changed), "removed": sorted(removed)}}


def _index_label(idx):
    if idx.get("in_sync") is True:
        return "已同步"
    if idx.get("in_sync") is False:
        pending = idx.get("pending") or {}
        return f"未同步（{sum(len(v) for v in pending.values())} 项待摄取）"
    return idx.get("note", "状态未知")


def _env_meta():
    return {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "code": _code_meta(), "kb": _kb_meta(),
            "config": _config_meta(), "index": _index_meta()}


def _print_env_compare(env, base_env):
    """核对基线采集时的代码 / 知识库 / 配置 —— 指标漂了先看这里，再谈回归。"""
    print("\n[基线环境]")
    if not base_env:
        print("  基线未记录环境（旧格式）—— 跑一次 --save-baseline 即可核对代码 / 知识库 / 配置")
        return
    print(f"  基线采集于 {base_env.get('created_at', '?')}")
    print(f"  代码    : {_code_label(env['code'])}  vs  基线 {_code_label(base_env.get('code') or {})}")
    kb, base_kb = env["kb"], base_env.get("kb") or {}
    cfg, base_cfg = env["config"], base_env.get("config") or {}
    kb_same = kb.get("fingerprint") == base_kb.get("fingerprint")
    cfg_same = cfg.get("fingerprint") == base_cfg.get("fingerprint")
    print(f"  知识库  : {kb['files']} 文件 fp={kb['fingerprint']}"
          f"  vs  基线 {base_kb.get('files', '?')} 文件 fp={base_kb.get('fingerprint', '?')}"
          + ("" if kb_same else "  ⚠ 已变"))
    print(f"  配置    : fp={cfg['fingerprint']}  vs  基线 fp={base_cfg.get('fingerprint', '?')}"
          + ("" if cfg_same else "  ⚠ 已变"))
    print(f"  索引    : {_index_label(env['index'])}  vs  基线 {_index_label(base_env.get('index') or {})}")
    if not (kb_same and cfg_same):
        print("  ⚠ 知识库 / 配置与基线不同 —— 指标变化可能来自它们，而不只是代码改动")
    if env["index"].get("in_sync") is False:
        print("  ⚠ 知识库有变更尚未摄取 —— 索引落后于知识库，先重建再谈指标")


def _baseline_payload(metrics, records):
    return {
        "route_accuracy": {"correct": metrics["route_accuracy"][0], "total": metrics["route_accuracy"][1]},
        "route_recall": {"covered": metrics["route_recall"][0], "total": metrics["route_recall"][1]},
        "route_via": metrics["route_via"],
        "hit_at_1": round(metrics["hit_at_k"][1], 4),
        "hit_at_3": round(metrics["hit_at_k"][3], 4),
        "hit_at_5": round(metrics["hit_at_k"][5], 4),
        "mrr": round(metrics["mrr"], 4),
        "no_recall_rate": {"zero": metrics["no_recall_rate"][0], "total": metrics["no_recall_rate"][1]},
        "structured_hit_rate": {"hit": metrics["structured_hit_rate"][0], "total": metrics["structured_hit_rate"][1]},
        "ids": [r["id"] for r in records],
        "per_id": {r["id"]: bool(r["ok"]) for r in records},
        "env": _env_meta(),
    }


def _diff_baseline(records, base):
    """按 id 逐条对比基线；基线是旧格式（无 per_id）时返回 None。"""
    per_id = base.get("per_id")
    if not isinstance(per_id, dict):
        return None
    cur = {r["id"]: bool(r["ok"]) for r in records}
    regress = sorted((i for i in cur if i in per_id and per_id[i] and not cur[i]), key=_id_sort_key)
    improve = sorted((i for i in cur if i in per_id and not per_id[i] and cur[i]), key=_id_sort_key)
    added = sorted((i for i in cur if i not in per_id), key=_id_sort_key)
    missing = sorted((i for i in per_id if i not in cur), key=_id_sort_key)
    return regress, improve, added, missing


def _print_id_list(label, ids):
    if not ids:
        print(f"{label}：无")
    elif len(ids) <= 12:
        print(f"{label}：{'、'.join(ids)}")
    else:
        print(f"{label}：{'、'.join(ids[:12])} …（共 {len(ids)} 条）")


def _save_baseline(metrics, records):
    os.makedirs(os.path.dirname(_BASELINE_FILE), exist_ok=True)
    payload = _baseline_payload(metrics, records)
    rotated = os.path.isfile(_BASELINE_FILE)
    if rotated:
        shutil.copyfile(_BASELINE_FILE, _PREV_BASELINE_FILE)
    with open(_BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    ids, env = payload["ids"], payload["env"]
    print(f"\n[基线] 已写入 {_BASELINE_FILE}（{ids[0]}–{ids[-1]}，{len(ids)} 条）")
    print(f"       环境：代码 {_code_label(env['code'])}；知识库 {env['kb']['files']} 文件"
          f" fp={env['kb']['fingerprint']}；配置 fp={env['config']['fingerprint']}")
    print(f"       索引：{_index_label(env['index'])}（按 ingest 快照核对）")
    if rotated:
        print(f"       上一版留存为 {_PREV_BASELINE_FILE}")


def _compare_baseline(metrics, records):
    if not os.path.isfile(_BASELINE_FILE):
        print(f"\n[基线] 无基线文件 {_BASELINE_FILE}，跳过对比")
        return
    with open(_BASELINE_FILE, encoding="utf-8") as f:
        base = json.load(f)
    cur = _baseline_payload(metrics, records)

    base_ids, cur_ids = base.get("ids"), [r["id"] for r in records]
    # 聚合值只在 id 集合一致时可比；集合不同就明说不可比，回归看逐条
    if not base_ids:
        warn = "  ⚠ 基线未记录 id（旧格式），无法核对集合范围"
    elif set(base_ids) != set(cur_ids):
        warn = (f"  ⚠ id 集合与基线不同：基线 {base_ids[0]}–{base_ids[-1]}({len(base_ids)})"
                f" vs 本次 {cur_ids[0]}–{cur_ids[-1]}({len(cur_ids)}) —— 聚合值不可直接比较，回归看下方逐条")
    else:
        warn = None
    print("\n[基线对比]（当前 - 基线，负值=回退）")
    if warn:
        print(warn)
    ra, rb = cur["route_accuracy"], base.get("route_accuracy") or {}
    print(f"  域路由top1: {ra['correct']}/{ra['total']}  vs  {rb.get('correct', '?')}/{rb.get('total', '?')}"
          f"   覆盖率 {cur['route_recall']['covered']}/{cur['route_recall']['total']}"
          f"  vs  {(base.get('route_recall') or {}).get('covered', '?')}/{(base.get('route_recall') or {}).get('total', '?')}")
    print(f"  hit@1   : {cur['hit_at_1']:.4f}  vs  {base['hit_at_1']:.4f}  ({cur['hit_at_1'] - base['hit_at_1']:+.4f})")
    print(f"  hit@3   : {cur['hit_at_3']:.4f}  vs  {base['hit_at_3']:.4f}  ({cur['hit_at_3'] - base['hit_at_3']:+.4f})")
    print(f"  hit@5   : {cur['hit_at_5']:.4f}  vs  {base['hit_at_5']:.4f}  ({cur['hit_at_5'] - base['hit_at_5']:+.4f})")
    print(f"  MRR     : {cur['mrr']:.4f}  vs  {base['mrr']:.4f}  ({cur['mrr'] - base['mrr']:+.4f})")
    print(f"  无召回率: {cur['no_recall_rate']['zero']}/{cur['no_recall_rate']['total']}  vs  {base['no_recall_rate']['zero']}/{base['no_recall_rate']['total']}")
    print(f"  结构化直出: {cur['structured_hit_rate']['hit']}/{cur['structured_hit_rate']['total']}  vs  {base['structured_hit_rate']['hit']}/{base['structured_hit_rate']['total']}")

    _print_env_compare(cur["env"], base.get("env"))

    diff = _diff_baseline(records, base)
    if diff is None:
        print("\n[基线逐条] 基线是旧格式（无逐条结果），只能看聚合；跑一次 --save-baseline 后即可按 id 逐条回归。")
        return
    regress, improve, added, missing = diff
    print("\n[基线逐条]（按 id）")
    _print_id_list("  回归（基线过 → 本次不过）", regress)
    _print_id_list("  改善（基线不过 → 本次过）", improve)
    _print_id_list("  新增（基线无记录，首次）", added)
    _print_id_list("  未跑（基线有、本次没跑）", missing)


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

    missing_id, dup_id = _validate_ids(entries)
    _assert(missing_id == 0, f"评测集每条都有 id（缺 {missing_id} 条）")
    _assert(not dup_id, f"评测集 id 唯一（重复：{'、'.join(dup_id) if dup_id else '无'}）")

    # --only 只跑某段 id：批次回归用（如 --only R001-R070 只跑首批）
    only = _arg_value("--only")
    if only:
        try:
            entries = _filter_by_id(entries, only)
        except ValueError as exc:
            _assert(False, f"--only 参数非法：{exc}")
            return stats()
        _assert(len(entries) > 0, f"--only {only} 命中 {len(entries)} 条")
    if entries:
        print(f"\n评测范围：{entries[0]['id']}–{entries[-1]['id']}（{len(entries)} 条）")

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
        _save_baseline(metrics, records)
    if "--compare" in sys.argv:
        _compare_baseline(metrics, records)

    return stats()


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
