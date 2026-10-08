"""会话级 trace：每轮问答落一条结构化记录。

同目录下另有三份**旁路证据**（放在该会话的目录里，文件名同主干、后缀区分）：
  - `<sid>_<YYYYMMDD>_<HHMM>.chunks.jsonl`：本轮召回片段全文（含排序与分数），由 `chunks()` 触发、`end_turn()` 落盘；
  - `<sid>_<YYYYMMDD>_<HHMM>.shadow.jsonl`：影子决策对照（跑在 daemon 线程，晚于本轮结束，故独立追加）；
  - `<sid>_<YYYYMMDD>_<HHMM>.feedback.jsonl`：前端「有用/无用」标注，落在**被标注轮次那天**的文件里。
布局：`<TRACE_DIR>/<session_id>_<YYYYMMDD>_<HHMM>/<session_id>_<YYYYMMDD>_<HHMM>[.<kind>].jsonl`——
一个会话一个目录，目录名与文件名都带时间（与 `data/context/` 命名一致；目录时间 = 该会话首条 trace，
文件名时间 = 写入当天）；一天一份，跨天换新文件。
trace 行里只留指路字段（`steps.retrieve.chunks_log` / `steps.shadow.log`），正文不进 trace。
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta

from tools.log_tool import log_path

VERSION = 2
# 本轮记录形如 {v, ts, session_id, turn_index, query, retry, tag, tags, detail, steps, ms, aborted, error}
SIDE_VERSION = 1
# 旁路证据形如 {v, ts, session_id, turn_index, query, ...}
CHUNK_KIND = "chunks"
SHADOW_KIND = "shadow"
FEEDBACK_KIND = "feedback"
CHUNK_MAX_CHARS = 4000       # 单片正文上限（防极端长条目把旁路文件撑爆）
FEEDBACK_LOOKBACK_DAYS = 7   # 标注时按 query 往回找轮次的天数

_local = threading.local()   # 当前线程正在处理的那一轮（ask_stream 的 wrapper 开/收）
_t0 = threading.local()      # 本轮起点，只用于算 ms.total
_seq = {}                    # (session_id, 日期) -> 已写轮数，进程内首次从文件行数种入
_last = {}                   # session_id -> 上一条记录（判「重新回答」用，首次从文件末行种入）
_lock = threading.Lock()
_stem_cache = {}             # (session_id, 日期) -> 当天文件名主干（首次写入时定名，之后沿用）
_dir_cache = {}              # session_id -> 会话目录（带时间，首次写入时定名，之后沿用）


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _path(session_id: str, kind: str = "", day: str = "", create: bool = False) -> str:
    """该会话某天的 trace 文件路径（含旁路证据）；当天首次写入时按当前时间定名，之后沿用。
    """
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


def _root_dir() -> str:
    """trace 根目录（`TRACE_DIR` 可覆盖）。"""
    return os.environ.get("TRACE_DIR") or os.path.join(log_path, "trace")


def _session_dir(session_id: str, create: bool = False, stamp: str = "") -> str:
    """该会话的 trace 目录：`<root>/<session_id>_<YYYYMMDD>_<HHMM>/`（时间 = 该会话首条 trace 的时间）。
    """
    sid = session_id or "default"
    cached = _dir_cache.get(sid)
    if cached and os.path.isdir(cached):
        # 命中缓存也要保住「目录名带时间」：旧命名目录在写入时补上时间
        if create and os.path.basename(cached) == sid:
            _dir_cache[sid] = _rename_legacy_dir(cached, sid)
        return _dir_cache[sid]
    root = _root_dir()
    legacy = ""
    try:
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if not os.path.isdir(full):
                continue
            if name.startswith(sid + "_"):      # 正式命名：<sid>_<时间>
                _dir_cache[sid] = full
                return full
            if name == sid:                     # 旧命名（只有会话 ID）→ 写入时补上时间
                legacy = full
    except OSError:
        pass
    if legacy:
        _dir_cache[sid] = _rename_legacy_dir(legacy, sid) if create else legacy
        return _dir_cache[sid]
    if not create:
        return ""
    _dir_cache[sid] = os.path.join(root, "%s_%s" % (sid, stamp or datetime.now().strftime("%Y%m%d_%H%M")))
    return _dir_cache[sid]


def _rename_legacy_dir(path: str, sid: str) -> str:
    """把「目录名只有会话 ID」的旧目录补上时间（取最早一条 trace 的 ts）；失败就沿用原名。"""
    stamp = ""
    try:
        names = sorted(n for n in os.listdir(path)
                       if n.endswith(".jsonl") and "." not in n[:-6])
        if names:
            with open(os.path.join(path, names[0]), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ts = json.loads(line).get("ts") or ""
                        if len(ts) >= 16:
                            stamp = ts[:4] + ts[5:7] + ts[8:10] + "_" + ts[11:13] + ts[14:16]
                        break
    except Exception:  # noqa: BLE001
        pass
    new = "%s_%s" % (path, stamp or datetime.now().strftime("%Y%m%d_%H%M"))
    try:
        os.rename(path, new)
        return new
    except OSError:
        return path


def _stem_of(session_id: str, day: str, create: bool, session_dir: str, stamp: str = "") -> str:
    """该会话某天的文件名主干 `<sid>_<YYYYMMDD>_<HHMM>`：已存在则沿用，否则按当前时间新建。"""
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
    """该会话的 `(YYYYMMDD, 路径)` 列表，按日期升序（一个会话可能跨多天）。
    """
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


def _append(path: str, record: dict) -> None:
    """追写一行 JSON（best-effort）。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _seed(session_id: str, key) -> None:
    """首次用到某会话（当天）时读一次文件：拿行数（轮次序号）与最后一条记录（判「重新回答」）。"""
    n, last = 0, None
    try:
        with open(_path(session_id), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    last = json.loads(line)
                except ValueError:
                    continue
    except OSError:
        pass
    _seq[key] = n
    _last[session_id] = last or {}


def begin_turn(session_id: str, query: str) -> None:
    """开一轮并补写上一轮；同一句 query 连续出现＝「重新回答」，记 retry=True。"""
    try:
        end_turn()
        with _lock:
            key = (session_id, _today())
            if key not in _seq:
                _seed(session_id, key)
            _seq[key] += 1
            turn_index = _seq[key]
            prev = _last.get(session_id) or {}
        _t0.start = time.perf_counter()
        retry = bool(prev.get("session_id") == session_id and prev.get("query") == query)
        _local.turn = {
            "v": VERSION,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "turn_index": turn_index,
            "query": query,
            "retry": retry,
            "tag": None,
            "tags": [],
            "detail": {},
            "steps": {},
            "ms": {},
            "aborted": False,
            "error": None,
        }
    except Exception:  # noqa: BLE001
        _local.turn = None


def step(name: str, **fields) -> None:
    """记一个环节的输入输出（加环节＝加一个 key）。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        turn["steps"].setdefault(name, {}).update(fields)
    except Exception:  # noqa: BLE001
        pass


def note_behavior(tag: str, detail: dict = None) -> None:
    """记一个行为标签：`tag` 取最后一次（主行为），`tags` 保留全序列。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        turn["tag"] = tag
        turn["tags"].append(tag)
        if detail:
            turn["detail"].update({k: v for k, v in detail.items() if isinstance(v, (str, int, float, bool))})
    except Exception:  # noqa: BLE001
        pass


def add_ms(name: str, started_at: float) -> None:
    """记某个环节耗时（传入 start 时的 time.perf_counter()）。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        turn["ms"][name] = int((time.perf_counter() - started_at) * 1000)
    except Exception:  # noqa: BLE001
        pass


def note_error(exc: BaseException) -> None:
    """记本轮异常（异常仍照常向上抛，这里只留痕）。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        turn["error"] = "%s: %s" % (type(exc).__name__, exc)
    except Exception:  # noqa: BLE001
        pass


def brief(obj, max_str: int = 40):
    """把值压成看得懂又不炸日志的摘要：标量原样、长字符串截断、对象列表只留条数。"""
    if isinstance(obj, dict):
        return {k: brief(v, max_str) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if all(isinstance(x, (str, int, float, bool, type(None))) for x in obj):
            return [brief(x, max_str) for x in obj]
        return "%d 项" % len(obj)
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[:max_str] + "…"
    return obj


def _as_triple(item):
    """兼容 `(doc, score, meta)` 与裸 doc 两种输入。"""
    if isinstance(item, tuple) and len(item) == 3:
        return item[0], item[1], item[2]
    return item, None, None


def _num(value, ndigits: int = 4):
    try:
        return round(float(value), ndigits) if value is not None else None
    except (TypeError, ValueError):
        return None


def chunk_brief(ranked, n: int = 3) -> list:
    """取前 n 个召回片段的 {file, score} 摘要，供 trace 行用（任何异常都吞掉）。
    """
    out = []
    try:
        for item in list(ranked)[:n]:
            doc, score, _meta = _as_triple(item)
            meta = getattr(doc, "metadata", None) or {}
            out.append({"file": meta.get("file_name"), "score": _num(score, 3)})
    except Exception:  # noqa: BLE001
        return []
    return out


def chunk_records(ranked, max_chars: int = CHUNK_MAX_CHARS) -> list:
    """`(doc, score, meta)` 列表 → 旁路文件里的片段记录（全文 + 排序 + 分数）。"""
    out = []
    for rank, item in enumerate(list(ranked), 1):
        doc, score, meta = _as_triple(item)
        meta = meta or {}
        text = getattr(doc, "page_content", "") or ""
        md = getattr(doc, "metadata", None) or {}
        entry = next((ln.strip() for ln in text.split("\n") if ln.strip()), "")
        out.append({
            "rank": rank,
            "file": md.get("file_name"),
            "entry": entry[:80],
            "score": _num(score),
            "rerank": _num(meta.get("rerank_score")),
            "rrf": _num(meta.get("rrf_score")),
            "dense_rank": meta.get("dense_rank"),
            "sparse_rank": meta.get("sparse_rank"),
            "chars": len(text),
            "text": text if len(text) <= max_chars else text[:max_chars] + "…",
        })
    return out


def chunks(ranked) -> None:
    """记本轮召回的片段（含全文），由 `end_turn()` 落到 `<sid>.chunks.jsonl`。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        turn["_chunks"] = chunk_records(ranked)
    except Exception:  # noqa: BLE001
        pass


def current_turn() -> dict:
    """当前轮的最小定位信息，给异步任务（影子决策）对齐 trace 用。"""
    turn = getattr(_local, "turn", None) or {}
    return {"session_id": turn.get("session_id"), "turn_index": turn.get("turn_index")}


def shadow_turn(ctx: dict, record: dict) -> None:
    """影子决策落盘。
    """
    sid = (ctx or {}).get("session_id")
    if not sid:
        return
    rec = {"v": SIDE_VERSION, "ts": datetime.now().isoformat(timespec="seconds"),
           "session_id": sid, "turn_index": (ctx or {}).get("turn_index")}
    rec.update(record or {})
    _append(_path(sid, SHADOW_KIND, create=True), rec)


def find_turn(session_id: str, query: str, days: int = FEEDBACK_LOOKBACK_DAYS) -> dict:
    """按 query 找该会话最近的一轮（从今天往回 `days` 天；跳过 aborted）。

    前端标注只有会话与原话，没有轮次号，靠它定位；返回
    `{session_id, date, turn_index, query, tag}`，找不到返回 `{}`（可直接当
    `feedback_turn` 的 `ctx` 用）。
    """
    for offset in range(days):
        day_s = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
        path = _path(session_id, day=day_s)
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
                        "query": rec.get("query"), "tag": rec.get("tag")}
    return {}


def feedback_turn(ctx: dict, rating: str, reason: str = "", note: str = "") -> None:
    """把「有用/无用」标注落到该轮所在日期目录的 `<sid>.feedback.jsonl`。

    `ctx` 来自 `find_turn`；写入 best-effort，失败不影响接口。
    """
    sid = (ctx or {}).get("session_id")
    if not sid or not (ctx or {}).get("date"):
        return
    rec = {"v": SIDE_VERSION, "ts": datetime.now().isoformat(timespec="seconds"),
           "session_id": sid, "turn_index": (ctx or {}).get("turn_index"),
           "query": (ctx or {}).get("query"), "tag": (ctx or {}).get("tag"),
           "rating": rating, "reason": reason or "", "note": note or ""}
    _append(_path(sid, FEEDBACK_KIND, day=ctx.get("date"), create=True), rec)


def end_turn(aborted: bool = False) -> None:
    """收尾并落盘（幂等）；写失败只吞掉。

    召回片段正文写旁路文件，trace 行只留指路字段（`steps.retrieve.chunks_log`）。
    """
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    _local.turn = None
    try:
        turn["aborted"] = bool(aborted)
        started = getattr(_t0, "start", None)
        if started:
            turn["ms"]["total"] = int((time.perf_counter() - started) * 1000)
        sid = turn.get("session_id")
        chunk_list = turn.pop("_chunks", None)
        if chunk_list:
            retrieve = turn["steps"].setdefault("retrieve", {})
            chunks_path = _path(sid, CHUNK_KIND, create=True)
            retrieve["chunks_log"] = os.path.basename(chunks_path) if chunks_path else ""
            _append(chunks_path, {
                "v": SIDE_VERSION,
                "ts": turn["ts"],
                "session_id": sid,
                "turn_index": turn.get("turn_index"),
                "query": turn.get("query"),
                "effective_query": (turn["steps"].get("rewrite") or {}).get("effective_query"),
                "domain": retrieve.get("domain"),
                "n_chunks": len(chunk_list),
                "chunks": chunk_list,
            })
        _append(_path(sid, create=True), turn)
        _last[sid] = turn
    except Exception:  # noqa: BLE001
        pass
