"""trace 目录与文件路径：日期目录 + 会话目录 + 当天主干。

文件名/目录名只在首次写入时定名，之后由缓存沿用；TRACE_DIR 可覆盖根目录。
"""
from __future__ import annotations

import os
import re
from datetime import datetime

from utils.log_tool import log_path

_DATE_DIR = re.compile(r"^\d{8}$")
_stem_cache = {}             # (session_id, 日期) -> 当天文件名主干
_dir_cache = {}              # session_id -> 会话目录（带时间）


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _root_dir() -> str:
    """trace 根目录（TRACE_DIR 可覆盖）。"""
    return os.environ.get("TRACE_DIR") or os.path.join(log_path, "trace")


def _path(session_id: str, kind: str = "", day: str = "", create: bool = False) -> str:
    """该会话某天的 trace 文件路径（含旁路证据）；当天首次写入时按当前时间定名。"""
    day = day or _today()
    stamp = datetime.now().strftime("%Y%m%d_%H%M") if create else ""
    session_dir = _session_dir(session_id, create=create, stamp=stamp)
    if not session_dir:
        return ""
    stem = _stem_of(session_id, day, create, session_dir, stamp)
    if not stem:
        return ""
    name = "%s.%s.jsonl" % (stem, kind) if kind else "%s.jsonl" % stem
    return os.path.join(session_dir, name)


def session_dirs(root: str = "") -> list:
    """root 下日期目录里的全部会话目录，按路径升序。"""
    base = root or _root_dir()
    out = []
    try:
        for name in sorted(os.listdir(base)):
            full = os.path.join(base, name)
            if not _DATE_DIR.match(name) or not os.path.isdir(full):
                continue
            out += [os.path.join(full, sub) for sub in sorted(os.listdir(full))
                    if os.path.isdir(os.path.join(full, sub))]
    except OSError:
        pass
    return out


def _session_dir(session_id: str, create: bool = False, stamp: str = "") -> str:
    """该会话的 trace 目录：<root>/<YYYYMMDD>/<session_id>_<YYYYMMDD>_<HHMM>/。"""
    sid = session_id or "default"
    cached = _dir_cache.get(sid)
    if cached and os.path.isdir(cached):
        return cached
    root = _root_dir()
    for full in session_dirs(root):
        if os.path.basename(full).startswith(sid + "_"):
            _dir_cache[sid] = full
            return full
    if not create:
        return ""
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M")
    _dir_cache[sid] = os.path.join(root, stamp.split("_")[0], "%s_%s" % (sid, stamp))
    return _dir_cache[sid]


def reset_session_cache(session_id: str) -> None:
    """清掉某会话的目录/文件名缓存（测试重置或目录被删后调用）。"""
    sid = session_id or "default"
    _dir_cache.pop(sid, None)
    for key in [k for k in list(_stem_cache) if k[0] == sid]:
        _stem_cache.pop(key, None)


def _stem_of(session_id: str, day: str, create: bool, session_dir: str, stamp: str = "") -> str:
    """该会话某天的文件名主干；已存在则沿用，否则按当前时间新建。"""
    key = (session_id or "default", day)
    if key in _stem_cache:
        return _stem_cache[key]
    prefix = "%s_%s_" % (key[0], day.replace("-", ""))
    stem = ""
    try:
        for fn in sorted(os.listdir(session_dir)):
            if fn.startswith(prefix) and fn.endswith(".jsonl") and "." not in fn[:-6]:
                stem = fn[:-6]
                break
    except OSError:
        pass
    if not stem and create:
        stem = "%s_%s" % (key[0], stamp or datetime.now().strftime("%Y%m%d_%H%M"))
    if stem:
        _stem_cache[key] = stem
    return stem


def day_files(session_id: str, kind: str = "") -> list:
    """该会话的 (YYYYMMDD, 路径) 列表，按日期升序。"""
    out = []
    session_dir = _session_dir(session_id)
    if not session_dir:
        return out
    suffix = ".%s.jsonl" % kind if kind else ".jsonl"
    try:
        names = sorted(os.listdir(session_dir))
    except OSError:
        return out
    for fn in names:
        if not fn.endswith(suffix):
            continue
        stem = fn[: -len(suffix)]
        if "." in stem:
            continue
        parts = stem.rsplit("_", 2)
        out.append((parts[-2] if len(parts) >= 3 else "", os.path.join(session_dir, fn)))
    return sorted(out)


def _date_of_file(path: str) -> str:
    """trace 文件所属日期目录（YYYY-MM-DD）；无日期层返回空。"""
    day = os.path.basename(os.path.dirname(os.path.dirname(path)))
    if _DATE_DIR.match(day):
        return "%s-%s-%s" % (day[:4], day[4:6], day[6:])
    return ""
