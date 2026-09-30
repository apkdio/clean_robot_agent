"""会话级 trace：每轮问答落一条结构化记录。
"""

import json
import os
import threading
import time
from datetime import datetime

from tools.log_tool import log_path

VERSION = 2
# 本轮记录形如 {v, ts, session_id, turn_index, query, retry, tag, tags, detail, steps, ms, aborted, error}

_local = threading.local()   # 当前线程正在处理的那一轮（ask_stream 的 wrapper 开/收）
_t0 = threading.local()      # 本轮起点，只用于算 ms.total
_seq = {}                    # (session_id, 日期) -> 已写轮数，进程内首次从文件行数种入
_last = {}                   # session_id -> 上一条记录（判「重新回答」用，首次从文件末行种入）
_lock = threading.Lock()


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _path(session_id: str) -> str:
    root = os.environ.get("TRACE_DIR") or os.path.join(log_path, "trace")
    return os.path.join(root, _today(), "%s.jsonl" % (session_id or "default"))


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


def chunk_brief(docs, n: int = 3) -> list:
    """取前 n 个召回片段的 {file, score} 摘要，供 trace 用（任何异常都吞掉）。
    """
    out = []
    try:
        for doc in list(docs)[:n]:
            meta = getattr(doc, "metadata", None) or {}
            score = getattr(doc, "score", None)
            out.append({"file": meta.get("file_name"),
                        "score": round(float(score), 3) if score is not None else None})
    except Exception:  # noqa: BLE001
        return []
    return out


def end_turn(aborted: bool = False) -> None:
    """收尾并落盘（幂等）；写失败只吞掉。
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
        path = _path(turn.get("session_id"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
        _last[turn.get("session_id")] = turn
    except Exception:  # noqa: BLE001
        pass
