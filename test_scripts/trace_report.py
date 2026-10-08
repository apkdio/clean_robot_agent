"""读主链路 trace，按会话列出每一轮的走向——「从哪一轮开始跑偏」的一眼视图。

trace 由 `tools/trace_store.py` 落盘（默认 `logs/trace/<session_id>/<session_id>_<YYYYMMDD>_<HHMM>.jsonl`，
一个会话一个目录、当天一份文件；`TRACE_DIR` 可覆盖）。本脚本**只读**，不改任何数据、不调模型。

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
"""
import argparse
import json
import os
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 刻意不 import `_runner`：它会设 TRACE_DIR 等测试隔离变量，而这里读的是真实 trace（--dir 可覆盖）。
from tools import trace_store  # noqa: E402

_SLOW_MS = 30000   # 单轮超过这个耗时就在摘要里标「慢」


def _root_dir(arg_dir: str) -> str:
    return arg_dir or os.environ.get("TRACE_DIR") or os.path.join(trace_store.log_path, "trace")


def _load(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue          # 半截行（写到一半被杀）跳过即可
    return rows


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


def _session_dir(root: str, session_id: str) -> str:
    """解析会话目录：会话名本身或 `<会话名>_*`（目录名带创建时间）都认；找不到返回按会话名的路径。"""
    exact = os.path.join(root, session_id)
    if os.path.isdir(exact):
        return exact
    try:
        for name in sorted(os.listdir(root)):
            if name.startswith(session_id + "_") and os.path.isdir(os.path.join(root, name)):
                return os.path.join(root, name)
    except OSError:
        pass
    return exact


def _session_files(root: str, session_id: str, kind: str = "", date: str = "") -> list:
    """该会话在 root 下的 trace 文件（`kind` 空 = 主文件，非空 = 旁路证据），按文件名升序。

    `date` 非空时只留那一天（按文件名里的 YYYYMMDD）；一个会话可能跨多天。
    """
    suffix = (".%s.jsonl" % kind) if kind else ".jsonl"
    want_day = date.replace("-", "")
    out = []
    try:
        names = sorted(os.listdir(_session_dir(root, session_id)))
    except OSError:
        return out
    for fn in names:
        if not fn.endswith(suffix):
            continue
        stem = fn[: -len(suffix)]
        if "." in stem:                       # 查主文件时排除旁路文件
            continue
        parts = stem.rsplit("_", 2)
        if want_day and (len(parts) < 3 or parts[-2] != want_day):
            continue                     # 名字里没带日期的（旧命名 / 测试用短名）只在不按日期筛时才看
        out.append(os.path.join(_session_dir(root, session_id), fn))
    return out


def _print_session(session_id: str, paths: list) -> None:
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
            print("\n".join(_turn_lines(rec)))
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
    """列出该日期有 trace 的会话（每个会话一个目录）。"""
    if not os.path.isdir(root):
        print("没有该目录：%s" % root)
        return
    found = []
    for name in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, name)):
            continue
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
    print("=" * 72)
    print("会话 %s  |  %d 条评价  |  %s" % (session_id, len(rows), path))
    print("=" * 72)
    useless = Counter(r.get("reason") or "-" for r in rows if r.get("rating") == "useless")
    if useless:
        print("\n无用原因分布：%s" % "、".join(
            "%s×%d" % (FEEDBACK_REASON_LABELS.get(k, k), n) for k, n in useless.most_common()))
    for rec in rows:
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
        print("   %s  %s" % ("  ".join(bits), rec.get("ts")))


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
    args = ap.parse_args()

    root = _root_dir(args.dir)
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
    _print_session(args.session, paths)


if __name__ == "__main__":
    main()
