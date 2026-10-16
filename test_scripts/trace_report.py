"""读主链路 trace，按会话列出每一轮的走向——「从哪一轮开始跑偏」的一眼视图。

trace 由 `tools/trace_store.py` 落盘（默认 `logs/trace/<YYYYMMDD>/<session_id>_<YYYYMMDD>_<HHMM>/<同名>.jsonl`，
日期目录 + 一会话一目录、当天一份文件；`TRACE_DIR` 可覆盖）。本脚本**只读**，不改任何数据、不调模型。

用法：解释器用 .venv\\Scripts\\python.exe

    # 列某天有 trace 的会话（默认今天）
    python test_scripts/trace_report.py --list
    # 看某个会话每一轮（默认跨天全看，按天分段）
    python test_scripts/trace_report.py --session <session_id>
    # 看该会话每轮召回的片段全文 / 影子决策 / 用户评价（旁路文件）
    python test_scripts/trace_report.py --session <session_id> --chunks
    python test_scripts/trace_report.py --session <session_id> --shadow
    python test_scripts/trace_report.py --session <session_id> --feedback
    # 只看某一天 / 指定目录（评测跑出来的 trace 在 temp/trace_test）
    python test_scripts/trace_report.py --session abc12345 --date 2026-09-29 --dir temp/trace_test
    # 按平铺形态打印
    python test_scripts/trace_report.py --session abc12345 --flat
    # 把某天渲染成自包含 HTML 报告（只出报告，不打印 trace 文本）
    python test_scripts/trace_report.py --html                       # 最近有 trace 的一天
    python test_scripts/trace_report.py --html --date 2026-10-15 --out-dir temp/reports
    python test_scripts/trace_report.py --html --dir temp/trace_test --out /tmp/x.html
默认落点 reports/<YYYYMMDD>/trace_<YYYYMMDD>.html（`--out-dir` 换根目录，`--out` 直接给整路径，不再加日期层）。

默认按分层视图打印：进门判断 / 有序路径 / 检索各遍与来源分布 / SOP 动作链 / 出口；
没有分层块的记录自动回退到平铺视图。字段含义见 notes/TRACE.md。
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 刻意不 import `_runner`：它会设 TRACE_DIR 等测试隔离变量，而这里读的是真实 trace（--dir 可覆盖）。
from tools import trace_store  # noqa: E402
from test_scripts import trace_reader  # noqa: E402

# 读取层（本脚本的文本视图与 HTML 报告都用）与 trace_reader 共用，避免两套读法漂移
_root_dir = trace_reader.root_dir
_load = trace_reader.load
_session_dir = trace_reader.session_dir
_session_files = trace_reader.session_files

_SLOW_MS = 30000   # 单轮超过这个耗时就在摘要里标「慢」


def _flags(rec: dict) -> list:
    """值得多看一眼的标记（启发式，不是判定）。"""
    steps = rec.get("steps") or {}
    retrieve = steps.get("retrieve") or {}
    ms = rec.get("ms") or {}
    out = []
    if rec.get("aborted"):
        out.append("aborted 未跑完")
    if rec.get("error"):
        out.append("error")
    if rec.get("retry"):
        out.append("retry 重答")
    if retrieve.get("n_chunks") == 0 or rec.get("tag") == "no_answer_fallback":
        out.append("0 召回/兜底")
    ret = rec.get("retrieval") or {}
    if (ret.get("twice_retrieval") or {}).get("active"):
        out.append("撤域二遍召回")
    if any((p.get("rerank") or {}).get("degraded") for p in ret.get("passes") or []):
        out.append("精排降级")
    if isinstance(ms.get("total"), int) and ms["total"] >= _SLOW_MS:
        out.append("慢 %ds" % round(ms["total"] / 1000))
    return out


def _turn_lines(rec: dict) -> list:
    steps = rec.get("steps") or {}
    bits = ["tag=%s" % (rec.get("tag") or "-")]
    intent = steps.get("intent") or {}
    if intent:
        margin = intent.get("margin")
        bits.append("意图=%s%s" % (intent.get("intent"),
                                 "(%.3f)" % float(margin) if isinstance(margin, (int, float)) else ""))
    rewrite = steps.get("rewrite") or {}
    if rewrite and rewrite.get("via") not in (None, "none"):
        bits.append("改写=%s→「%s」" % (rewrite.get("via"), rewrite.get("effective_query")))
    retrieve = steps.get("retrieve") or {}
    if retrieve:
        bits.append("检索=%s/%s条" % (retrieve.get("domain") or "全库", retrieve.get("n_chunks")))
    ms = rec.get("ms") or {}
    if ms:
        bits.append("耗时=" + "/".join("%s:%s" % (k, v) for k, v in sorted(ms.items())) + "ms")
    line = "  #%s  %s\n      %s" % (rec.get("turn_index"), rec.get("query") or "", "  ".join(bits))
    flags = _flags(rec)
    if flags:
        line += "\n      ⚠ " + "、".join(flags)
    if rec.get("tags") and len(rec["tags"]) > 1:
        line += "\n      行为序列: %s" % "→".join(rec["tags"])
    return [line]


def _dur(v) -> str:
    """耗时（毫秒）→ 人看的写法。"""
    if not isinstance(v, (int, float)):
        return "-"
    return "%dms" % v if v < 1000 else "%.1fs" % (v / 1000.0)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return ("%.3f" % v).rstrip("0").rstrip(".")
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _kv(d) -> str:
    if not isinstance(d, dict) or not d:
        return ""
    return "(" + " ".join("%s=%s" % (k, _fmt(v)) for k, v in d.items()) + ")"


def _turn_lines_nested(rec: dict, labels: dict) -> list:
    """分层视图：进门判断 / 有序路径 / 检索各遍与分布 / SOP 动作链 / 出口。"""
    out = ["  #%s  %s" % (rec.get("turn_index"), rec.get("query") or "")]
    head = ["tag=%s" % (rec.get("tag") or "-")]
    if rec.get("branch"):
        head.append("出口=%s" % rec["branch"])
    head.append("耗时=%s" % _dur((rec.get("ms") or {}).get("total")))
    out.append("      " + "  ".join(head))

    route = rec.get("route") or {}
    gate = "  ".join("%s%s" % (labels.get("route.%s" % k, k),
                                 "命中" if (route.get(k) or {}).get("hit") else "无")
                     for k in ("inject", "danger", "emotion"))
    intent = route.get("intent") or {}
    if intent:
        gate += "  |  %s=%s" % (labels.get("route.intent", "意图"), intent.get("label"))
        if isinstance(intent.get("margin"), (int, float)):
            gate += "(%.3f)" % intent["margin"]
        if intent.get("downgrade"):
            gate += "→%s" % intent["downgrade"]
    out.append("      进门: " + gate)

    path = rec.get("path") or []
    if path:
        steps = []
        for n in path:
            bit = labels.get(n.get("node"), n.get("node"))
            if n.get("status") == "skipped":
                bit += "·跳过(%s)" % (n.get("by") or "-")
            if n.get("ms"):
                bit += " %s" % _dur(n["ms"])
            steps.append(bit)
        out.append("      路径: " + " → ".join(steps))
        facts = ["%s%s" % (labels.get(n["node"], n["node"]), _kv(n.get("out")))
                 for n in path if n.get("out") and not n["node"].startswith("route.")]
        if facts:
            out.append("      事实: " + " ｜ ".join(facts))

    ret = rec.get("retrieval") or {}
    if ret:
        bits = []
        dom = ret.get("domain") or {}
        if dom:
            bits.append("域=%s（路由 %s）" % (_fmt(dom.get("value")) if dom.get("value") else "全库",
                                             dom.get("via") or "-"))
        rw = ret.get("rewrite") or {}
        if rw.get("via") not in (None, "none"):
            bits.append("改写=%s→「%s」" % (rw.get("via"), rw.get("effective_query")))
        if bits:
            out.append("      检索: " + "  ".join(bits))
        for p in ret.get("passes") or []:
            rk = p.get("rerank") or {}
            out.append("            pass%s: 稠密%s/稀疏%s → 融合%s → 精排%s→%s%s" % (
                p.get("pass"), p.get("dense"), p.get("sparse"), p.get("fused"),
                rk.get("in"), rk.get("out"), "（降级，未精排）" if rk.get("degraded") else ""))
        twice = ret.get("twice_retrieval") or {}
        if twice.get("active"):
            trig = twice.get("trigger") or {}
            out.append("            二次召回: 域内 top1=%s < 阈值%s → 撤域全库，兜底池进 %s 条" % (
                trig.get("top1"), trig.get("threshold"), twice.get("kept")))
        mix = ret.get("merge") or {}
        if mix:
            tail = ""
            if ret.get("evidence"):
                tail = "  证据=%s" % (ret["evidence"] or {}).get("file")
            if ret.get("source") == "retry":
                tail += "  （重检：%s）" % _kv(ret.get("retry"))
            out.append("            最终: %s 条（两路都有 %s / 只稠密 %s / 只 BM25 %s）%s" % (
                mix.get("final"), mix.get("both"), mix.get("dense_only"), mix.get("sparse_only"), tail))

    sop = rec.get("sop") or {}
    if sop:
        flow = " → ".join("%s%s=%s" % (f.get("type"), "(%s)" % f["slot"] if f.get("slot") else "",
                                       f.get("outcome")) for f in (sop.get("flow") or []))
        out.append("      SOP: %s 第%s步 ｜ 槽位 %s ｜ %s%s" % (
            sop.get("id"), sop.get("step"), sop.get("n_slots"), flow or "-",
            " ｜ 已收尾" if sop.get("done") else ""))
    if rec.get("sop_gate"):
        out.append("      SOP 门: %s" % json.dumps(rec["sop_gate"], ensure_ascii=False))

    ans = rec.get("answer") or {}
    if ans:
        out.append("      出口: %s ｜ 模型 %s ｜ %s 字" % (
            ans.get("kind") or "-", ans.get("model") or "-", ans.get("chars")))
    shadow = rec.get("shadow") or {}
    if shadow:
        out.append("      影子: chain=%s ｜ 证据=%s" % (
            shadow.get("chain") or "-", (shadow.get("evidence") or {}).get("file") or "-"))
    flags = _flags(rec)
    if flags:
        out.append("      ⚠ " + "、".join(flags))
    if rec.get("tags") and len(rec["tags"]) > 1:
        out.append("      行为序列: %s" % "→".join(rec["tags"]))
    return out


def _render_turn(rec: dict, flat: bool = False) -> list:
    """默认走分层视图；没有分层块或 `--flat` 时走平铺。"""
    if flat or not (rec.get("branch") or rec.get("path")):
        return _turn_lines(rec)
    from config.word_dict_config import TRACE_NODE_LABELS
    return _turn_lines_nested(rec, TRACE_NODE_LABELS)


def _print_session(session_id: str, paths: list, flat: bool = False) -> None:
    """打印该会话的 trace；`paths` 按日期升序，一天一段。"""
    total = 0
    for path in paths:
        rows = _load(path)
        total += len(rows)
        print("=" * 72)
        print("会话 %s  |  %s  |  %d 轮" % (
            session_id, os.path.basename(path)[: -len(".jsonl")], len(rows)))
        if rows:
            print("版本 v%s  |  %s ~ %s" % (rows[0].get("v"), rows[0].get("ts"),
                                          rows[-1].get("ts")))
        print("=" * 72)
        for rec in rows:
            print("\n".join(_render_turn(rec, flat)))
        flagged = [r.get("turn_index") for r in rows if _flags(r)]
        if flagged:
            print("\n值得先看的轮次：%s" % "、".join("#%s" % t for t in flagged))
        _hint_sides(session_id, os.path.dirname(path))
        print()
    if len(paths) > 1:
        print("（该会话共 %d 天、%d 轮）" % (len(paths), total))


def _hint_sides(session_id: str, session_dir: str) -> None:
    """提示会话目录下的旁路证据（存在才提）。"""
    sides = []
    for kind, flag in ((trace_store.CHUNK_KIND, "--chunks"), (trace_store.SHADOW_KIND, "--shadow"),
                       (trace_store.FEEDBACK_KIND, "--feedback")):
        try:
            hit = any(fn.endswith(".%s.jsonl" % kind) for fn in os.listdir(session_dir))
        except OSError:
            hit = False
        if hit:
            sides.append("%s（%s）" % (kind, flag))
    if sides:
        print("\n另有旁路证据：%s —— 在 --session %s 后加对应开关查看" % ("、".join(sides), session_id))


def _print_list(root: str, day: str) -> None:
    """列出该日期有 trace 的会话（日期目录下的每个会话目录）。"""
    if not os.path.isdir(root):
        print("没有该目录：%s" % root)
        return
    found = []
    for full in trace_store.session_dirs(root):
        name = os.path.basename(full)
        paths = _session_files(root, name, date=day)
        if not paths:
            continue
        rows = _load(paths[-1])
        last = rows[-1] if rows else {}
        found.append((name, sum(len(_load(p)) for p in paths), last))
    print("%s 下 %d 个会话（%s）" % (root, len(found), day))
    for name, n, last in found:
        print("  %-42s %3d 轮   末轮 tag=%-20s %s" % (
            name, n, last.get("tag") or "-", last.get("ts") or "-"))


def _print_chunks(session_id: str, path: str) -> None:
    rows = _load(path)
    print("=" * 72)
    print("会话 %s  |  %d 轮召回片段  |  %s" % (session_id, len(rows), path))
    print("=" * 72)
    for rec in rows:
        head = "#%s  %s" % (rec.get("turn_index"), rec.get("query") or "")
        if rec.get("effective_query") and rec["effective_query"] != rec.get("query"):
            head += "   （检索用：%s）" % rec["effective_query"]
        print("\n%s" % head)
        print("   domain=%s  |  %d 片" % (rec.get("domain") or "全库", rec.get("n_chunks") or 0))
        for c in rec.get("chunks") or []:
            print("   [%s] score=%s (rerank=%s rrf=%s)  稠密#%s 稀疏#%s  %s  %s字" % (
                c.get("rank"), c.get("score"), c.get("rerank"), c.get("rrf"),
                c.get("dense_rank"), c.get("sparse_rank"), c.get("file"), c.get("chars")))
            for line in (c.get("text") or "").split("\n"):
                print("        %s" % line)


def _print_shadow(session_id: str, path: str) -> None:
    rows = _load(path)
    print("=" * 72)
    print("会话 %s  |  %d 条影子决策  |  %s" % (session_id, len(rows), path))
    print("=" * 72)
    for rec in rows:
        print("\n#%s  %s" % (rec.get("turn_index"), rec.get("query") or ""))
        print("   chain=%s(%s)  fc=%s  agree=%s  fc_ms=%s%s" % (
            rec.get("chain"), rec.get("chain_detail") or "-", rec.get("tool") or "-",
            rec.get("agree"), rec.get("ms"),
            "  error=%s" % rec["error"] if rec.get("error") else ""))
        if rec.get("args"):
            print("   fc_args=%s" % json.dumps(rec["args"], ensure_ascii=False))
        if rec.get("extra"):
            print("   fc_extra=%s" % rec.get("extra"))
        if rec.get("notes"):
            print("   notes=%s" % rec.get("notes"))


def _print_feedback(session_id: str, path: str) -> None:
    from collections import Counter

    from config.word_dict_config import FEEDBACK_REASON_LABELS, FEEDBACK_REASON_LAYERS

    rows = _load(path)
    # 同一轮可能有多条（用户改过标注）：生效的是最后一条
    effective, times = {}, Counter()
    for rec in rows:
        effective[rec.get("turn_index")] = rec
        times[rec.get("turn_index")] += 1
    print("=" * 72)
    print("会话 %s  |  %d 条评价 / %d 轮  |  %s" % (session_id, len(rows), len(effective), path))
    print("=" * 72)
    useless = Counter(r.get("reason") or "-" for r in effective.values()
                      if r.get("rating") == "useless")
    if useless:
        print("\n无用原因分布：%s" % "、".join(
            "%s×%d" % (FEEDBACK_REASON_LABELS.get(k, k), n) for k, n in useless.most_common()))
    for rec in effective.values():
        mark = "👍有用" if rec.get("rating") == "useful" else "👎无用"
        print("\n#%s  %s  %s" % (rec.get("turn_index"), mark, rec.get("query") or ""))
        bits = ["tag=%s" % (rec.get("tag") or "-")]
        if rec.get("reason"):
            key = rec["reason"]
            label = FEEDBACK_REASON_LABELS.get(key, key)      # 旧记录可能是自由文本，原样显示
            layer = FEEDBACK_REASON_LAYERS.get(key, "")
            bits.append("原因=%s%s%s" % (label, "（→%s）" % layer if layer else "",
                                       "（事后回填）" if rec.get("reason_backfill") else ""))
        if rec.get("note"):
            bits.append("补充=%s" % rec["note"])
        if times.get(rec.get("turn_index"), 1) > 1:
            bits.append("（该轮共 %d 条，按最后一条）" % times[rec.get("turn_index")])
        print("   %s  %s" % ("  ".join(bits), rec.get("ts")))


# ──────────────────────────────────────────────────────────────
# HTML 报告（`--html`）：把某天渲染成一份自包含页面，打开即看、不引外部 CDN
# ──────────────────────────────────────────────────────────────

BRANCH_COLORS = {
    "safety_alert": "#c0392b", "safety_carry": "#d35400", "block_inject": "#8e44ad",
    "refuse": "#7f8c8d", "no_answer_fallback": "#e67e22",
    "structured": "#2980b9", "answer": "#27ae60", "retrieve_answer": "#27ae60",
    "chitchat": "#9b59b6", "clarify": "#16a085",
    "start_sop": "#2c3e50", "sop_step": "#34495e", "exit_sop": "#95a5a6",
}


def _hhmmss(ts: str) -> str:
    return (ts or "")[11:19]


def _fmt_val(v) -> str:
    """表格里的一格：布尔说人话、空值给短横、嵌套结构压成一行 JSON。"""
    if isinstance(v, bool):
        return "是" if v else "否"
    if v is None or v == "":
        return "-"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _fmt_pairs(d) -> str:
    """逐节点各不同的小 dict → 「键=值 键=值」一行（path 的 out 用它）。"""
    if not isinstance(d, dict) or not d:
        return ""
    return " ".join("%s=%s" % (k, _fmt_val(v)) for k, v in d.items())


def turn_view(rec: dict, chunks: list, feedback: dict, max_chunk: int) -> dict:
    """一轮 → 模板要用的字段（全部取自 trace 现成字段，不另造口径）。"""
    rt = rec.get("route") or {}
    intent = rt.get("intent") or {}
    flags = [k for k in ("inject", "danger", "emotion") if (rt.get(k) or {}).get("hit")]
    ret = rec.get("retrieval") or {}
    rewrite = ret.get("rewrite") or {}
    domain = ret.get("domain") or {}
    shadow = rec.get("shadow") or {}
    ms = rec.get("ms") or {}
    branch = rec.get("branch") or "未记"
    views = []
    for c in chunks:
        text = c.get("text") or ""
        views.append({"rank": c.get("rank"), "score": c.get("score"), "file": c.get("file"),
                      "entry": c.get("entry"), "chars": c.get("chars"), "dense": c.get("dense_rank"),
                      "sparse": c.get("sparse_rank"), "rrf": c.get("rrf"), "rerank": c.get("rerank"),
                      "text": text[:max_chunk], "truncated": len(text) > max_chunk})
    return {
        "i": rec.get("turn_index"), "ts": _hhmmss(rec.get("ts")), "query": rec.get("query") or "",
        "retry": bool(rec.get("retry")), "aborted": bool(rec.get("aborted")), "error": rec.get("error"),
        "detail": rec.get("detail") or {},
        "hits": (rec.get("detail") or {}).get("hits"),
        "branch": branch, "tag": rec.get("tag"), "color": BRANCH_COLORS.get(branch, "#555"),
        "intent": intent.get("label"), "margin": intent.get("margin"), "intent_retry": intent.get("retry"),
        "flags": flags, "domain": domain.get("value"), "domain_via": domain.get("via"),
        "rewrite": rewrite, "passes": ret.get("passes") or [], "merge": ret.get("merge") or {},
        "twice": ret.get("twice_retrieval") or {},
        "sop": rec.get("sop") or {}, "sop_gate": rec.get("sop_gate") or {}, "shadow": shadow,
        "path": rec.get("path") or [], "answer": rec.get("answer") or {}, "ms": ms,
        "chunks": views, "feedback": feedback, "slow": (ms.get("total") or 0) >= _SLOW_MS,
        "norm": bool((rewrite.get("normalize") or {}).get("changed")),
        "shadow_mismatch": shadow.get("agree") is False,
        "no_recall": branch == "no_answer_fallback",
        "total_ms": ms.get("total"),
    }


def session_view(root: str, name: str, date: str, max_chunk: int) -> dict:
    """一个会话 → 模板要用的字段。"""
    paths = trace_reader.session_files(root, name, date=date)
    rows = []
    for p in paths:
        rows += trace_reader.load(p)
    sid = name.rsplit("_", 2)[0]
    fb_paths = trace_reader.session_files(root, name, kind="feedback", date=date)
    fb = trace_reader.effective_feedback(trace_reader.load(fb_paths[0]) if fb_paths else [])
    chunk_rows = []
    for p in trace_reader.session_files(root, name, kind="chunks", date=date):
        chunk_rows += trace_reader.load(p)
    chunks = trace_reader.chunks_by_turn(chunk_rows)
    turns = [turn_view(r, chunks.get(r.get("turn_index"), []), fb.get(r.get("turn_index")) or {}, max_chunk)
             for r in rows]
    return {"name": name, "sid": sid, "short": sid[:8], "date": date, "turns": turns,
            "n_turns": len(turns), "n_useless": sum(1 for t in turns if (t["feedback"] or {}).get("rating") == "useless")}


def latest_date(root: str) -> str:
    """最近有 trace 的日期（目录名 YYYYMMDD）。"""
    days = []
    try:
        for d in os.listdir(root):
            full = os.path.join(root, d)
            if os.path.isdir(full) and re.fullmatch(r"\d{8}", d):
                days.append(d)
    except OSError:
        pass
    if days:
        return max(days)
    # 旧布局（会话目录直接挂根下）：从目录名里取日期
    for name in os.listdir(root):
        m = re.search(r"_(\d{8})_\d{4}$", name)
        if m:
            days.append(m.group(1))
    return max(days) if days else datetime.now().strftime("%Y%m%d")


TEMPLATE = """{% macro kvtable(title, d) -%}
<table class="kv"><caption>{{ title }}</caption>
  {% for k, v in d.items() %}<tr><th>{{ k }}</th><td>{{ v|fmt }}</td></tr>{% endfor %}
</table>
{%- endmacro %}<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>trace 报告 · {{ date }}</title>
<style>
 :root{--fg:#1f2328;--dim:#6b7280;--line:#e5e7eb;--bg:#f6f7f9;--card:#fff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.6 -apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif}
 header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;z-index:5}
 h1{font-size:15px;margin:0 0 6px}
 .meta{color:var(--dim);font-size:12px}
 .filters{margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
 input[type=search]{padding:3px 8px;border:1px solid var(--line);border-radius:4px;min-width:200px}
 .chip{border:1px solid var(--line);background:#fff;border-radius:12px;padding:2px 9px;cursor:pointer;font-size:12px;user-select:none}
 .chip.on{background:#1f2328;color:#fff;border-color:#1f2328}
 main{padding:14px 16px 40px;max-width:1180px;margin:0 auto}
 section{background:var(--card);border:1px solid var(--line);border-radius:8px;margin-bottom:14px;overflow:hidden}
 section>h2{font-size:13px;margin:0;padding:9px 12px;border-bottom:1px solid var(--line);background:#fafbfc;font-weight:600}
 section>h2 small{color:var(--dim);font-weight:400;margin-left:8px}
 .turn{border-bottom:1px solid var(--line);padding:9px 12px}
 .turn:last-child{border-bottom:none}
 .turn.hide{display:none}
 .row1{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline}
 .badge{color:#fff;border-radius:4px;padding:1px 7px;font-size:11px;white-space:nowrap}
 .tag{border:1px solid var(--line);border-radius:4px;padding:0 6px;font-size:11px;color:var(--dim)}
 .tag.warn{border-color:#e67e22;color:#e67e22}
 .tag.bad{border-color:#c0392b;color:#c0392b}
 .q{margin:4px 0 0;font-weight:600;word-break:break-word}
 .facts{color:var(--dim);font-size:12px;margin-top:3px;display:flex;flex-wrap:wrap;gap:10px}
 details{margin-top:6px}
 summary{cursor:pointer;color:#2563eb;font-size:12px}
 table{border-collapse:collapse;font-size:12px;margin:6px 0}
 th,td{border:1px solid var(--line);padding:2px 7px;text-align:left;vertical-align:top}
 th{background:#fafbfc;color:var(--dim);font-weight:500}
 caption{caption-side:top;text-align:left;color:var(--dim);font-size:12px;padding:2px 0}
 .kv th{width:110px}
 pre{background:#fafbfc;border:1px solid var(--line);border-radius:4px;padding:7px;margin:4px 0;white-space:pre-wrap;word-break:break-word;font-size:12px}
 .chunk{border-left:3px solid var(--line);padding-left:8px;margin:7px 0}
 .k{color:var(--dim)}
 button.copy{font-size:11px;border:1px solid var(--line);background:#fff;border-radius:4px;padding:1px 7px;cursor:pointer}
 code{background:#f3f4f6;border-radius:3px;padding:0 3px}
</style></head><body>
<header>
  <h1>扫地机器人客服 · trace 报告 <span class="meta">{{ date }}</span></h1>
  <div class="meta">根目录 {{ root }} · {{ sessions|length }} 个会话 / {{ total_turns }} 轮 · 出口分布
    {% for b, n in branch_counts %}<code>{{ b }} {{ n }}</code> {% endfor %}
    · 人工👎 {{ total_useless }} 条</div>
  <div class="filters">
    <input type="search" id="kw" placeholder="搜 query 关键词…">
    <span class="chip" data-f="useless">有👎</span>
    <span class="chip" data-f="no_recall">0 召回</span>
    <span class="chip" data-f="norm">归一化改过</span>
    <span class="chip" data-f="shadow">影子不一致</span>
    <span class="chip" data-f="bad">error / aborted</span>
    <span class="chip" data-f="slow">慢（≥30s）</span>
    <span class="meta">·</span>
    {% for b in branches %}<span class="chip" data-b="{{ b }}">{{ b }}</span>{% endfor %}
    <span class="chip" id="reset">重置</span>
  </div>
</header>
<main>
{% for s in sessions %}
  <section data-sid="{{ s.name }}">
    <h2>{{ s.short }} <small>{{ s.name }} · {{ s.n_turns }} 轮{% if s.n_useless %} · 👎 {{ s.n_useless }}{% endif %}</small></h2>
    {% for t in s.turns %}
    <article class="turn" data-branch="{{ t.branch }}" data-useless="{{ 1 if (t.feedback or {}).get('rating')=='useless' else 0 }}"
             data-no_recall="{{ 1 if t.no_recall else 0 }}" data-norm="{{ 1 if t.norm else 0 }}"
             data-shadow="{{ 1 if t.shadow_mismatch else 0 }}" data-bad="{{ 1 if (t.error or t.aborted) else 0 }}"
             data-slow="{{ 1 if t.slow else 0 }}" data-q="{{ t.query }}">
      <div class="row1">
        <span class="k">#{{ t.i }}</span><span class="k">{{ t.ts }}</span>
        <span class="badge" style="background:{{ t.color }}">{{ t.branch }}</span>
        {% if t.intent %}<span class="tag">意图 {{ t.intent }}{% if t.margin is not none %} {{ '%.3f'|format(t.margin) }}{% endif %}</span>{% endif %}
        {% if t.intent_retry %}<span class="tag">两遍分类→{{ t.intent_retry.label }}</span>{% endif %}
        {% if t.flags %}<span class="tag warn">进门命中 {{ t.flags|join(' / ') }}</span>{% endif %}
        {% if t.domain %}<span class="tag">域 {{ t.domain }}{% if t.domain_via %}（{{ t.domain_via }}）{% endif %}</span>{% endif %}
        {% if t.twice.active %}<span class="tag warn">撤域二次召回 kept={{ t.twice.kept }}</span>{% endif %}
        {% if t.norm %}<span class="tag">归一化改过</span>{% endif %}
        {% if t.retry %}<span class="tag">重发</span>{% endif %}
        {% if t.aborted %}<span class="tag bad">中止</span>{% endif %}
        {% if t.error %}<span class="tag bad">error</span>{% endif %}
        {% if t.slow %}<span class="tag warn">慢</span>{% endif %}
        {% if (t.feedback or {}).get('rating') == 'useless' %}<span class="tag bad">👎 {{ t.feedback.reason or '-' }}{% if t.feedback.note %}：{{ t.feedback.note }}{% endif %}</span>{% endif %}
        {% if (t.feedback or {}).get('rating') == 'useful' %}<span class="tag">👍</span>{% endif %}
        <span class="k">{{ t.total_ms and '%.0fms'|format(t.total_ms) or '' }}</span>
        <button class="copy" data-copy="{{ t.query }}">复制 query</button>
      </div>
      <p class="q">{{ t.query }}</p>
      <div class="facts">
        {% if t.answer %}<span>出口 {{ t.answer.kind }}{% if t.answer.model %} · {{ t.answer.model }}{% endif %}{% if t.answer.chars %} · {{ t.answer.chars }} 字{% endif %}</span>{% endif %}
        {% if t.hits is not none %}<span>召回 {{ t.hits }} 片</span>
        {% elif t.chunks %}<span>召回 {{ t.chunks|length }} 片</span>{% endif %}
        {% if t.sop %}<span>SOP {{ t.sop.id }} · step {{ t.sop.step }}{% if t.sop.slots %} · {{ t.sop.slots }}{% endif %}</span>{% endif %}
        {% if t.ms %}<span>耗时 retrieve {{ t.ms.get('retrieve') }}ms / 合计 {{ t.ms.get('total') }}ms</span>{% endif %}
      </div>
      <details><summary>链路与检索（path · 检索各遍 · rewrite / merge / 影子）</summary>
        {% if t.path %}
        <table><caption>有序路径</caption><tr><th>节点</th><th>状态</th><th>by</th><th>ms</th><th>out</th></tr>
        {% for n in t.path %}<tr><td>{{ n.node }}</td><td>{{ n.status }}</td><td>{{ n.by or '' }}</td>
          <td>{{ n.ms if n.ms is not none else '' }}</td><td>{{ n.out|pairs }}</td></tr>{% endfor %}</table>
        {% endif %}
        {% if t.detail %}{{ kvtable('detail · 标签细节', t.detail) }}{% endif %}
        {% if t.passes %}
        <table><caption>检索各遍</caption>
          <tr><th>遍</th><th>域过滤</th><th>稠密</th><th>稀疏</th><th>融合</th><th>精排 in→out</th><th>模型</th></tr>
          {% for p in t.passes %}{% set rk = p.rerank or {} %}
          <tr><td>{{ p.pass|fmt }}</td>
            <td>{% if p.filter %}{{ p.filter.file_name if p.filter.file_name else p.filter }}{% else %}全库{% endif %}</td>
            <td>{{ p.dense|fmt }}</td><td>{{ p.sparse|fmt }}</td><td>{{ p.fused|fmt }}</td>
            <td>{{ rk.get('in')|fmt }} → {{ rk.get('out')|fmt }}{% if rk.degraded %}（降级，未精排）{% endif %}</td>
            <td>{{ rk.get('model')|fmt }}</td></tr>
          {% endfor %}
        </table>
        {% endif %}
        {% if t.rewrite %}
        <table class="kv"><caption>rewrite · 检索口径</caption>
          {% for k, v in t.rewrite.items() if k != 'normalize' %}<tr><th>{{ k }}</th><td>{{ v|fmt }}</td></tr>{% endfor %}
          {% set nz = t.rewrite.normalize %}{% if nz %}
          <tr><th>normalize</th><td>{% if nz.changed %}改了 → {{ nz.text }}{% else %}未改（{{ nz.text }}）{% endif %}</td></tr>
          {% endif %}
        </table>
        {% endif %}
        {% if t.merge %}{{ kvtable('merge · 两路合并', t.merge) }}{% endif %}
        {% if t.twice and t.twice.active %}{{ kvtable('twice_retrieval · 撤域二次召回', t.twice) }}{% endif %}
        {% if t.shadow %}{{ kvtable('shadow · 影子决策', t.shadow) }}{% endif %}
        {% if t.sop_gate %}{{ kvtable('sop_gate · SOP 门', t.sop_gate) }}{% endif %}
      </details>
      {% if t.chunks %}
      <details><summary>召回片段 {{ t.chunks|length }} 片（正文）</summary>
        {% for c in t.chunks %}
        <div class="chunk">
          <div class="facts"><span class="k">rank {{ c.rank }} · score {{ c.score }}</span>
            <span class="k">{{ c.file }}{% if c.entry %} · {{ c.entry }}{% endif %}</span><span class="k">{{ c.chars }} 字</span>
            <span class="k">稠密#{{ c.dense }} 稀疏#{{ c.sparse }} rrf{{ c.rrf }} 精排{{ c.rerank }}</span></div>
          <pre>{{ c.text }}{% if c.truncated %}…（已截断）{% endif %}</pre>
        </div>
        {% endfor %}
      </details>
      {% endif %}
    </article>
    {% endfor %}
  </section>
{% endfor %}
{% if not sessions %}<p>这一天没有 trace。</p>{% endif %}
</main>
<script>
const kw = document.getElementById('kw');
let on = new Set();
document.querySelectorAll('.chip[data-f]').forEach(c => c.onclick = () => {
  const k = c.dataset.f; on.has(k) ? on.delete(k) : on.add(k); c.classList.toggle('on'); apply();
});
document.querySelectorAll('.chip[data-b]').forEach(c => c.onclick = () => {
  const k = 'b:' + c.dataset.b; on.has(k) ? on.delete(k) : on.add(k); c.classList.toggle('on'); apply();
});
document.getElementById('reset').onclick = () => { on.clear(); kw.value = '';
  document.querySelectorAll('.chip.on').forEach(c => c.classList.remove('on')); apply(); };
kw.oninput = apply;
function apply() {
  const text = (kw.value || '').trim().toLowerCase();
  const bs = [...on].filter(k => k.startsWith('b:')).map(k => k.slice(2));
  document.querySelectorAll('.turn').forEach(t => {
    const d = t.dataset;
    let ok = true;
    if (on.has('useless') && d.useless !== '1') ok = false;
    if (on.has('no_recall') && d.no_recall !== '1') ok = false;
    if (on.has('norm') && d.norm !== '1') ok = false;
    if (on.has('shadow') && d.shadow !== '1') ok = false;
    if (on.has('bad') && d.bad !== '1') ok = false;
    if (on.has('slow') && d.slow !== '1') ok = false;
    if (bs.length && !bs.includes(d.branch)) ok = false;
    if (text && !(d.q || '').toLowerCase().includes(text)) ok = false;
    t.classList.toggle('hide', !ok);
  });
  document.querySelectorAll('section').forEach(s => {
    const vis = [...s.querySelectorAll('.turn')].some(t => !t.classList.contains('hide'));
    s.style.display = vis ? '' : 'none';
  });
}
document.querySelectorAll('button.copy').forEach(b => b.onclick = () => {
  const v = b.dataset.copy || '';
  (navigator.clipboard ? navigator.clipboard.writeText(v) : Promise.reject())
    .then(() => { b.textContent = '已复制'; setTimeout(() => b.textContent = '复制 query', 1200); })
    .catch(() => { const a = document.createElement('textarea'); a.value = v; document.body.appendChild(a);
                   a.select(); document.execCommand('copy'); a.remove(); b.textContent = '已复制'; });
});
</script></body></html>
"""


def render(sessions: list, date: str, root: str) -> str:
    import jinja2          # 只 HTML 路径需要，文本视图不因此多一个依赖

    branch_counts = {}
    branches = []
    for s in sessions:
        for t in s["turns"]:
            b = t["branch"]
            branch_counts[b] = branch_counts.get(b, 0) + 1
            if b not in branches:
                branches.append(b)
    env = jinja2.Environment(autoescape=True)
    env.filters["fmt"] = _fmt_val
    env.filters["pairs"] = _fmt_pairs
    return env.from_string(TEMPLATE).render(
        sessions=sessions, date=date, root=root, branches=branches,
        branch_counts=sorted(branch_counts.items(), key=lambda kv: -kv[1]),
        total_turns=sum(s["n_turns"] for s in sessions),
        total_useless=sum(s["n_useless"] for s in sessions))


def out_dir(arg: str = "") -> str:
    """报告输出根目录：--out-dir 优先，默认项目下 reports/。"""
    return arg or os.path.join(_ROOT, "reports")


def resolve_day(root: str, date_arg: str = "") -> str:
    """YYYYMMDD：给了就用它，否则取最近有 trace 的一天。"""
    return (date_arg or "").replace("-", "") or latest_date(root)


def default_out_path(base: str, day: str) -> str:
    """默认落点：<base>/<YYYYMMDD>/trace_<YYYYMMDD>.html（与 trace 一样按日期一层目录）。"""
    return os.path.join(base, day, "trace_%s.html" % day)


def write_report(root: str, day: str, out_path: str, max_chunk: int = 1500) -> tuple:
    """把某天渲染成自包含 HTML 写到 out_path（父目录自动建），返回 (路径, 会话数, 轮数)。"""
    date = "%s-%s-%s" % (day[:4], day[4:6], day[6:8])
    sessions = [session_view(root, name, date, max_chunk) for name, _ in trace_reader.sessions_on(root, date)]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render(sessions, date, root))
    return out_path, len(sessions), sum(s["n_turns"] for s in sessions)


def main() -> None:
    ap = argparse.ArgumentParser(description="读主链路 trace，看某个会话每一轮的走向")
    ap.add_argument("--session", help="会话 id（= logs/trace 下的目录名，如 abc12345-……）")
    ap.add_argument("--list", action="store_true", help="列出该日期下有 trace 的会话")
    ap.add_argument("--chunks", action="store_true", help="看该会话每轮召回的片段全文（.<kind>.chunks.jsonl）")
    ap.add_argument("--shadow", action="store_true", help="看该会话的影子决策对照（.shadow.jsonl）")
    ap.add_argument("--feedback", action="store_true", help="看该会话的用户评价（.feedback.jsonl）")
    ap.add_argument("--date", default=None,
                    help="只看某天（YYYY-MM-DD）；默认：--list 看今天，--session 看全部")
    ap.add_argument("--dir", default=None, help="trace 根目录，默认 logs/trace（或 TRACE_DIR）")
    ap.add_argument("--flat", action="store_true", help="按平铺形态打印")
    ap.add_argument("--html", action="store_true", help="产出那天的 HTML 报告（只出报告，不打印 trace 文本）")
    ap.add_argument("--out-dir", default=None, help="HTML 报告输出根目录，默认 reports/；报告落在其 <YYYYMMDD>/ 下")
    ap.add_argument("--out", default=None, help="HTML 报告的输出文件整路径；给了就不再套日期层")
    ap.add_argument("--max-chunk-chars", type=int, default=1500, help="HTML 里单条召回片段正文的截断上限")
    args = ap.parse_args()

    root = _root_dir(args.dir)
    if args.html:
        day = resolve_day(root, args.date or "")
        out = args.out or default_out_path(out_dir(args.out_dir), day)
        path, n_sessions, n_turns = write_report(root, day, out, args.max_chunk_chars)
        print("已生成 %s（%d 会话 / %d 轮，%.0f KB）"
              % (path, n_sessions, n_turns, os.path.getsize(path) / 1024))
        return
    if args.chunks or args.shadow or args.feedback:
        if not args.session:
            print("--chunks / --shadow / --feedback 需要配合 --session")
            sys.exit(1)
        kind = trace_store.CHUNK_KIND if args.chunks else (
            trace_store.SHADOW_KIND if args.shadow else trace_store.FEEDBACK_KIND)
        paths = _session_files(root, args.session, kind=kind, date=args.date or "")
        if not paths:
            print("没有该文件：%s/%s 下无 .%s.jsonl" % (root, args.session, kind))
            sys.exit(1)
        _printers = {trace_store.CHUNK_KIND: _print_chunks,
                     trace_store.SHADOW_KIND: _print_shadow,
                     trace_store.FEEDBACK_KIND: _print_feedback}
        for path in paths:
            _printers[kind](args.session, path)
        return
    if args.list or not args.session:
        _print_list(root, args.date or datetime.now().strftime("%Y-%m-%d"))
        if not args.session:
            return
    paths = _session_files(root, args.session, date=args.date or "")
    if not paths:
        print("没有该会话的 trace：%s/%s" % (root, args.session))
        sys.exit(1)
    _print_session(args.session, paths, flat=args.flat)


if __name__ == "__main__":
    main()
