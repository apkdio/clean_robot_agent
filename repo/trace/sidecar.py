"""trace 旁路证据：事件、召回片段与用户评价。

旁路文件与主 trace 同目录；写入 best-effort，读不到返回空。
`shadow` 仅保留历史文件只读兼容，主链自 2026-11-10 起不再写入。
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from repo.trace import facts as fact_utils
from repo.trace import paths

SIDE_VERSION = 1
# 旁路证据（chunks / feedback，及历史 shadow）共用的 schema 版本，字段名同为 v
CHUNK_KIND = "chunks"
SHADOW_KIND = "shadow"
FEEDBACK_KIND = "feedback"
EVENT_KIND = "events"
CHUNK_MAX_CHARS = 4000       # 单片正文上限（防极端长条目把旁路文件撑爆）
FEEDBACK_LOOKBACK_DAYS = 7   # 标注时按 query 往回找轮次的天数


def _append(path: str, record: dict) -> None:
    """追写一行 JSON（best-effort）。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _event(event: str, **facts) -> None:
    """旁路事件日志：执行中逐条追加，进程崩溃后仍可复原到崩溃前一步。"""
    from repo.trace import write

    turn = getattr(write._local, "turn", None) or getattr(write._local, "event_turn", None)
    if not turn:
        return
    try:
        sid = turn.get("session_id")
        path = paths._path(sid, EVENT_KIND, create=True)
        if not path:
            return
        rec = {
            "v": SIDE_VERSION,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session_id": sid,
            "turn_index": turn.get("turn_index"),
            "event": event,
        }
        for key, value in (facts or {}).items():
            if isinstance(value, dict):
                rec.update(fact_utils.to_facts(value))
            else:
                rec[key] = fact_utils.to_fact(value)
        _append(path, rec)
    except Exception:  # noqa: BLE001
        pass


def event(event: str, **facts) -> None:
    """公开事件入口：业务代码也可向旁路日志追加事件。"""
    _event(event, **facts)


def events_of(session_id: str, day: str = "") -> list:
    """读某会话某天的事件日志；文件不存在返回空列表。"""
    path = paths._path(session_id, EVENT_KIND, day=day)
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def shadow_turn(ctx: dict, record: dict) -> None:
    """历史 shadow 决策落盘；主链已不再调用，仅供旧文件/读侧兼容。"""
    sid = (ctx or {}).get("session_id")
    if not sid:
        return
    rec = {"v": SIDE_VERSION, "ts": datetime.now().isoformat(timespec="seconds"),
           "session_id": sid, "turn_index": (ctx or {}).get("turn_index")}
    rec.update(record or {})
    _append(paths._path(sid, SHADOW_KIND, create=True), rec)


def feedback_turn(ctx: dict, rating: str, reason: str = "", note: str = "") -> None:
    """把「有用/无用」标注落到该轮所在日期目录的 feedback 旁路。"""
    sid = (ctx or {}).get("session_id")
    if not sid or not (ctx or {}).get("date"):
        return
    rec = {"v": SIDE_VERSION, "ts": datetime.now().isoformat(timespec="seconds"),
           "session_id": sid, "turn_index": (ctx or {}).get("turn_index"),
           "query": (ctx or {}).get("query"), "tag": (ctx or {}).get("tag"),
           "rating": rating, "reason": reason or "", "note": note or ""}
    _append(paths._path(sid, FEEDBACK_KIND, day=ctx.get("date"), create=True), rec)


def feedback_of(session_id: str, date: str, turn_index: int) -> dict:
    """该轮已有的标注（同一轮只认一条）；没有返回空。"""
    path = paths._path(session_id, FEEDBACK_KIND, day=date)
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("turn_index") == turn_index:
                    return rec
    except OSError:
        pass
    return {}


def effective_feedbacks(session_id: str, date: str = "") -> dict:
    """该会话（某天）每轮的生效标注：同一轮有多条时取最后一条。"""
    out = {}
    want = date.replace("-", "")
    for day, path in paths.day_files(session_id, FEEDBACK_KIND):
        if want and day != want:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    out[rec.get("turn_index")] = rec
        except OSError:
            continue
    return out
