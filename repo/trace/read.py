"""trace 读取侧：最近一轮、按锚点/query 定位轮次。

只读主 trace；旁路证据由 sidecar 读取。
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta

from repo.trace import paths, sidecar


def latest_turn(session_id: str) -> dict:
    """该会话最近一条**已落盘**的轮次（不含正在进行的这一轮），跨天也认。"""
    if not session_id:
        return {}
    best = None                                   # (mtime, path)
    try:
        for d in paths.session_dirs():
            if not os.path.basename(d).startswith(session_id):
                continue
            for name in os.listdir(d):
                if not name.endswith(".jsonl") or re.search(
                        r"\.(chunks|feedback|events)\.jsonl$", name):
                    continue
                p = os.path.join(d, name)
                m = os.path.getmtime(p)
                if best is None or m > best[0]:
                    best = (m, p)
    except OSError:
        return {}
    if not best:
        return {}
    try:
        with open(best[1], encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        rec = json.loads(lines[-1])
    except (OSError, ValueError, IndexError):
        return {}
    return {"session_id": session_id,
            "date": paths._date_of_file(best[1]) or (rec.get("ts") or "")[:10],
            "turn_index": rec.get("turn_index"), "query": rec.get("query"),
            "aborted": bool(rec.get("aborted"))}


def find_turn_by_index(session_id: str, date: str, turn_index) -> dict:
    """按锚点直接取该轮；找不到返回空。"""
    if not session_id or not date or turn_index is None:
        return {}
    try:
        with open(paths._path(session_id, day=date), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("turn_index") == turn_index:
                    return {"session_id": session_id, "date": date, "turn_index": turn_index,
                            "query": rec.get("query"), "tag": rec.get("tag"),
                            "branch": rec.get("branch"), "sop": rec.get("sop") or {},
                            "sop_gate": rec.get("sop_gate") or {}}
    except OSError:
        pass
    return {}


def find_turn(session_id: str, query: str, days: int = sidecar.FEEDBACK_LOOKBACK_DAYS) -> dict:
    """按 query 找该会话最近的一轮（从今天往回 days 天；跳过 aborted）。"""
    for offset in range(days):
        day_s = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
        path = paths._path(session_id, day=day_s)
        rows = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            continue
        for rec in reversed(rows):
            if rec.get("query") == query and not rec.get("aborted"):
                return {"session_id": session_id, "date": day_s,
                        "turn_index": rec.get("turn_index"),
                        "query": rec.get("query"), "tag": rec.get("tag"),
                        "branch": rec.get("branch")}
    return {}
