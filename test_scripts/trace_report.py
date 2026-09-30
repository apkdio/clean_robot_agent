"""读主链路 trace，按会话列出每一轮的走向——「从哪一轮开始跑偏」的一眼视图。

trace 由 `tools/trace_store.py` 落盘（默认 `logs/trace/<YYYY-MM-DD>/<session_id>.jsonl`，
`TRACE_DIR` 可覆盖）。本脚本**只读**，不改任何数据、不调模型。

用法：解释器用 .venv\\Scripts\\python.exe

    # 列某天有 trace 的会话（默认今天）
    python test_scripts/trace_report.py --list
    # 看某个会话每一轮
    python test_scripts/trace_report.py --session <session_id>
    # 指定日期 / 目录（评测跑出来的 trace 在 temp/trace_test）
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


def _print_session(session_id: str, path: str) -> None:
    rows = _load(path)
    print("=" * 72)
    print("会话 %s  |  %d 轮  |  %s" % (session_id, len(rows), path))
    if rows:
        print("版本 v%s  |  %s ~ %s" % (rows[0].get("v"), rows[0].get("ts"),
                                      rows[-1].get("ts")))
    print("=" * 72)
    for rec in rows:
        print("\n".join(_turn_lines(rec)))
    flagged = [r.get("turn_index") for r in rows if _flags(r)]
    if flagged:
        print("\n值得先看的轮次：%s" % "、".join("#%s" % t for t in flagged))


def _print_list(root: str, day: str) -> None:
    day_dir = os.path.join(root, day)
    if not os.path.isdir(day_dir):
        print("没有该日期的 trace 目录：%s" % day_dir)
        return
    names = sorted(n for n in os.listdir(day_dir) if n.endswith(".jsonl"))
    print("%s 下 %d 个会话（%s）" % (day_dir, len(names), day))
    for name in names:
        rows = _load(os.path.join(day_dir, name))
        last = rows[-1] if rows else {}
        print("  %-42s %3d 轮   末轮 tag=%-20s %s" % (
            name[:-6], len(rows), last.get("tag") or "-", last.get("ts") or "-"))


def main() -> None:
    ap = argparse.ArgumentParser(description="读主链路 trace，看某个会话每一轮的走向")
    ap.add_argument("--session", help="会话 id（文件名主干，如 abc12345……）")
    ap.add_argument("--list", action="store_true", help="列出该日期下有 trace 的会话")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="日期，默认今天")
    ap.add_argument("--dir", default=None, help="trace 根目录，默认 logs/trace（或 TRACE_DIR）")
    args = ap.parse_args()

    root = _root_dir(args.dir)
    if args.list or not args.session:
        _print_list(root, args.date)
        if not args.session:
            return
    path = os.path.join(root, args.date, "%s.jsonl" % args.session)
    if not os.path.isfile(path):
        print("没有该会话的 trace：%s" % path)
        sys.exit(1)
    _print_session(args.session, path)


if __name__ == "__main__":
    main()
