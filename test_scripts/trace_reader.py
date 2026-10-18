"""trace 读取层：命令行读数器与它自带的 HTML 报告（trace_report.py）共用。

刻意不 import test_scripts._runner——它会设 TRACE_DIR 等测试隔离变量，而这里读的是真实 trace
（--dir 可指到评测隔离目录）。
"""

from __future__ import annotations

import json
import os

from tools import trace_store

SIDE_KINDS = ("chunks", "shadow", "feedback")


def root_dir(arg_dir: str = "") -> str:
    """trace 根目录：--dir 优先，其次 TRACE_DIR，最后 logs/trace。"""
    return arg_dir or os.environ.get("TRACE_DIR") or os.path.join(trace_store.log_path, "trace")


def session_dir(root: str, session_id: str) -> str:
    """解析会话目录（认日期层与旧布局）：会话名本身或 <会话名>_*；找不到返回按会话名的路径。"""
    exact = os.path.join(root, session_id)
    if os.path.isdir(exact):
        return exact
    for full in trace_store.session_dirs(root):
        name = os.path.basename(full)
        if name == session_id or name.startswith(session_id + "_"):
            return full
    return exact


def session_files(root: str, session_id: str, kind: str = "", date: str = "") -> list:
    """该会话在 root 下的 trace 文件（kind 空 = 主文件，非空 = 旁路证据），按文件名升序。

    date 非空时只留那一天（按文件名里的 YYYYMMDD）；一个会话可能跨多天。
    """
    suffix = (".%s.jsonl" % kind) if kind else ".jsonl"
    want_day = date.replace("-", "")
    out = []
    dir_path = session_dir(root, session_id)
    try:
        names = sorted(os.listdir(dir_path))
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
        out.append(os.path.join(dir_path, fn))
    return out


def load(path: str) -> list:
    """读 jsonl；半截行（写到一半被杀）跳过即可。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def sessions_on(root: str, date: str = "") -> list:
    """有 trace 的会话：[(会话目录名, 目录路径)]，按名升序（日期层在前的按日期排）。"""
    out = []
    for full in trace_store.session_dirs(root):
        name = os.path.basename(full)
        if session_files(root, name, date=date):
            out.append((name, full))
    return out


def effective_feedback(rows: list) -> dict:
    """每轮生效的评价：同一轮有多条时取最后一条（重新标注以最后一次为准）。"""
    out = {}
    for rec in rows:
        out[rec.get("turn_index")] = rec
    return out


def chunks_by_turn(rows: list) -> dict:
    """每轮的召回片段（按 rank 升序）。

    旁路文件是一轮一条记录（{"turn_index": n, "chunks": [...]}），片段在嵌套列表里——
    别当成「一行一片」。
    """
    out = {}
    for rec in rows:
        out.setdefault(rec.get("turn_index"), []).extend(rec.get("chunks") or [])
    for items in out.values():
        items.sort(key=lambda c: c.get("rank") or 0)
    return out
