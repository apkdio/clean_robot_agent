"""trace 写入侧：一轮的生命周期、分层块与旁路挂接。

同一线程的当前轮只保存在本模块的 `_local`；其它模块只通过薄壳读取同一实例。
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime

from config.word_dict_config import BEHAVIOR_BY_TAG, TRACE_NODE_LABELS, TRACE_NODE_STATUSES
from utils.log_tool import get_logger

from mapper.trace import facts as fact_utils
from mapper.trace import paths, sidecar

logger = get_logger(name="trace_store")

TRACE_VERSION = 4
# 轮次记录（主干）的 schema 版本；落盘字段名仍是 v，靠所在文件区分是谁的版本。结构见 notes/TRACE.md。
# 本轮记录 = 稳定核 {v, ts, session_id, turn_index, query, retry, tag, tags, detail, aborted, error}
#   + 事实块 {route, branch, path, retrieval, sop, sop_gate, answer} + ms

_local = threading.local()   # 当前线程正在处理的那一轮（ask_stream 的 wrapper 开/收）
_t0 = threading.local()      # 本轮起点，只用于算 ms.total
_seq = {}                    # (session_id, 日期) -> 已写轮数，进程内首次从文件行数种入
_last = {}                   # session_id -> 上一条记录（判「重新回答」用，首次从文件末行种入）
_lock = threading.Lock()
_SEQ_TTL = 7 * 24 * 3600   # trace:seq 只保留一周，避免日 key 无限堆积
_SEQ_LUA = """
local n = tonumber(redis.call('GET', KEYS[1]) or '0')
local seed = tonumber(ARGV[1])
if n < seed then
  redis.call('SET', KEYS[1], seed)
  n = seed
end
return redis.call('INCR', KEYS[1])
"""


def _scalars(obj) -> dict:
    """只留标量（数字 / 布尔 / 字符串 / 空）。"""
    if not isinstance(obj, dict):
        return {}
    return {k: v for k, v in obj.items()
            if v is None or isinstance(v, (str, int, float, bool))}


def _seed(session_id: str, key) -> None:
    """首次用到某会话（当天）时读一次文件：拿行数（轮次序号）与最后一条记录。"""
    n, last = 0, None
    try:
        with open(paths._path(session_id), encoding="utf-8") as f:
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


def _next_turn_index(session_id: str, day: str, file_lines: int) -> int:
    """跨进程轮次号：Redis INCR，首次用文件行数做种子；Redis 不可用回退本地计数。"""
    from gateway import redis_store

    r = redis_store.get_redis()
    if r is not None:
        key = "trace:seq:%s:%s" % (session_id, day)
        try:
            value = int(r.eval(_SEQ_LUA, 1, key, file_lines))
            try:
                r.expire(key, _SEQ_TTL)
            except Exception:  # noqa: BLE001
                pass
            return value
        except Exception as e:  # noqa: BLE001
            logger.warning("[Trace] Redis 轮次号失败，回退本地计数：%s", e)
            redis_store.mark_unavailable()
    key = (session_id, day)
    _seq[key] = max(int(_seq.get(key) or 0), int(file_lines or 0)) + 1
    return _seq[key]


def reset_sequence(session_id: str, day: str = "") -> None:
    """清掉某会话的轮次号（测试/重置用）；不清理则同 sid 会接着数。"""
    from gateway import redis_store

    day = day or paths._today()
    r = redis_store.get_redis()
    if r is not None:
        try:
            r.delete("trace:seq:%s:%s" % (session_id, day))
        except Exception as e:  # noqa: BLE001
            logger.warning("[Trace] 清 Redis 轮次号失败：%s", e)
    for key in [k for k in list(_seq) if k[0] == session_id]:
        _seq.pop(key, None)


def begin_turn(session_id: str, query: str, retry: bool = None) -> None:
    """开一轮并补写上一轮；同一句 query 连续出现＝「重新回答」，记 retry=True。"""
    try:
        end_turn()
        with _lock:
            key = (session_id, paths._today())
            if key not in _seq:
                _seed(session_id, key)
            turn_index = _next_turn_index(session_id, key[1], _seq[key])
            _seq[key] = max(int(_seq.get(key) or 0), int(turn_index))
            prev = _last.get(session_id) or {}
            try:
                from mapper.trace import read
                prev = read.latest_turn(session_id) or prev
            except Exception:  # noqa: BLE001
                pass
        _t0.start = time.perf_counter()
        retry = bool(retry) if retry is not None else bool(
            prev.get("session_id") == session_id and prev.get("query") == query)
        _local.turn = {
            "v": TRACE_VERSION,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "turn_index": turn_index,
            "query": query,
            "retry": retry,
            "tag": None,
            "tags": [],
            "detail": {},
            "route": {},
            "branch": None,
            "path": [],
            "retrieval": {},
            "sop": {},
            "sop_gate": {},
            "answer": {},
            "span_seq": 0,
            "span_stack": [],
            "_day": key[1],
            "_retrieved": False,
            "ms": {},
            "aborted": False,
            "error": None,
        }
        sidecar._event("begin", query=query[:120], retry=retry)
    except Exception:  # noqa: BLE001
        _local.turn = None
    _local.event_turn = _local.turn


# step() 的已知环节；新增 key 必须在这里登记，否则只能留下散字段。
_STEP_NAMES = {"intent", "rewrite", "retrieve", "generate", "sop", "sop_gate"}


def _apply_step(turn: dict, name: str, fields: dict) -> None:
    """把 step 的字段直接写进对应事实块，不再维护 steps 投影。"""
    if name == "intent":
        intent = dict(turn["route"].get("intent") or {})
        intent.update(label=fields.get("intent"),
                      margin=_num(fields.get("margin"), 6))
        if fields.get("degraded"):
            intent["degraded"] = True
        retry = fields.get("intent_retry") or {}
        if retry:
            intent["retry"] = retry
        turn["route"]["intent"] = intent
        node("route.intent", status="degraded" if fields.get("degraded") else "ok",
             out={"label": fields.get("intent"),
                  "margin": _num(fields.get("margin"), 6),
                  "retry": retry.get("label"),
                  "degraded": bool(fields.get("degraded"))})
    elif name == "rewrite":
        rw = turn["retrieval"].setdefault("rewrite", {})
        if fields.get("normalize") is not None:
            rw["normalize"] = fields["normalize"]
        if fields.get("via") is not None:
            rw["via"] = fields.get("via")
            rw["effective_query"] = fields.get("effective_query")
            node("rewrite", out={"via": rw.get("via"),
                                 "normalized": bool((rw.get("normalize") or {}).get("changed"))})
    elif name == "retrieve":
        turn["_retrieved"] = True
        if fields.get("domain") is not None or fields.get("via") is not None:
            turn["retrieval"]["domain"] = {"value": fields.get("domain"),
                                           "via": fields.get("via")}
            node("retrieve", out={"domain": fields.get("domain"),
                                  "via": fields.get("via"),
                                  "n": fields.get("n_chunks")})
    elif name == "generate":
        turn["answer"]["model"] = fields.get("model")
        node("generate", out={"model": fields.get("model")})
    elif name == "sop":
        turn["sop"].update(_scalars(fields))
        for key in ("slots", "result"):
            if fields.get(key) is not None:
                turn["sop"][key] = fact_utils.to_facts(fields[key])
        if fields.get("flow"):
            turn["sop"]["flow"] = fields["flow"]
    elif name == "sop_gate":
        turn["sop_gate"].update(_scalars(fields))
        node("sop.gate", status="skipped", by=fields.get("skipped"),
             in_={"sop": fields.get("sop")})
    else:
        logger.warning("[Trace] step '%s' 未登记：只记事件，不写事实块", name)


def step(name: str, **fields) -> None:
    """记一个环节的事实，并同步到对应分层块。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        _apply_step(turn, name, fields)
        sidecar._event("step", name=name, facts=fact_utils.to_facts(fields))
    except Exception:  # noqa: BLE001
        pass


# 环节 → 节点名（耗时同步到对应的路径节点用）
_NODE_OF_STEP = {"intent": "route.intent", "rewrite": "rewrite",
                 "retrieve": "retrieve", "generate": "generate"}

# route() 不落节点的键：intent 的节点由 step 落，ops 只有字段
_ROUTE_MIRRORED = {"intent", "ops"}


def _next_span_id(turn: dict) -> int:
    turn["span_seq"] = int(turn.get("span_seq") or 0) + 1
    return turn["span_seq"]


def _current_parent(turn: dict):
    stack = turn.get("span_stack") or []
    return stack[-1] if stack else None


def begin_span(name: str, kind: str = "step", parent: int | None = None,
               in_: dict = None):
    """开一个 span；返回 id。父节点默认取当前 span 栈顶，in_ 记输入事实。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return None
    if name not in TRACE_NODE_LABELS:
        raise ValueError("未登记的 trace 节点：%s" % name)
    if kind not in ("stage", "step", "action", "decision"):
        raise ValueError("未知 span kind：%s" % kind)
    parent = _current_parent(turn) if parent is None else parent
    span_id = _next_span_id(turn)
    in_facts = fact_utils.to_facts(in_)
    rec = {"id": span_id, "parent": parent, "node": name,
           "kind": kind, "status": "running"}
    if in_facts:
        rec["in"] = in_facts
    turn["path"].append(rec)
    turn["span_stack"].append(span_id)
    sidecar._event("span.begin", id=span_id, parent=parent, name=name, kind=kind,
                   facts=in_facts)
    return span_id


def end_span(span_id: int | None, status: str = "ok", by: str = None,
             out: dict = None, ms: int = None) -> None:
    """关闭 span；主行只保留状态/耗时/短事实，具体事件已在 events 中。"""
    turn = getattr(_local, "turn", None)
    if not turn or span_id is None:
        return
    if status not in TRACE_NODE_STATUSES:
        raise ValueError("未知 span status：%s" % status)
    rec = next((n for n in turn["path"] if n.get("id") == span_id), None)
    if rec is None:
        logger.warning("[Trace] end_span 找不到 span：%s", span_id)
        return
    rec["status"] = status
    if by:
        rec["by"] = by
    out_facts = fact_utils.to_facts(out)
    if out_facts:
        rec["out"] = out_facts
    if ms is not None:
        rec["ms"] = int(ms)
    stack = turn.get("span_stack") or []
    if span_id in stack:
        stack[:] = stack[:stack.index(span_id)]
    sidecar._event("span.end", id=span_id, status=status, by=by, ms=rec.get("ms"),
                   facts=out_facts)


def route(name: str, **facts) -> None:
    """记一条「进门判断」的结论，并顺手落一个路径节点。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        route_facts = fact_utils.to_facts(facts)
        turn["route"].setdefault(name, {}).update(route_facts)
        if name not in _ROUTE_MIRRORED:
            node("route.%s" % name, status="hit" if facts.get("hit") else "ok",
                 out=route_facts, kind="decision")
        sidecar._event("route", name=name, facts=route_facts)
    except Exception:  # noqa: BLE001
        pass


def node(name: str, status: str = "ok", by: str = None, in_: dict = None,
         out: dict = None, kind: str = "step") -> None:
    """按执行顺序记一个节点；自动挂到当前 span 下。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        rec = {"id": _next_span_id(turn), "parent": _current_parent(turn),
               "node": name, "kind": kind, "status": status}
        if by:
            rec["by"] = by
        in_facts = fact_utils.to_facts(in_)
        out_facts = fact_utils.to_facts(out)
        if in_facts:
            rec["in"] = in_facts
        if out_facts:
            rec["out"] = out_facts
        if name in turn["ms"]:
            rec["ms"] = turn["ms"][name]
        turn["path"].append(rec)
        node_facts = {**in_facts, **out_facts}
        sidecar._event("node", id=rec["id"], parent=rec["parent"], name=name,
                       kind=kind, status=status, by=by,
                       facts=node_facts, ms=rec.get("ms"))
    except Exception as e:  # noqa: BLE001
        logger.warning("[Trace] node event failed: %s", e)


def stage(name: str, status: str = "ok", ms: int = None, out: dict = None) -> None:
    """兼容入口：开/关一个 stage span。主链已改用 begin_span/end_span。"""
    span_id = begin_span(name, kind="stage")
    end_span(span_id, status=status, out=out, ms=ms)


def verify(checked: list, missing: list, action: str = "log", exempt: list = None,
           skipped: list = None, unchecked: list = None) -> None:
    """记答案校验结果：完整清单进 answer 块，路径节点只放计数和短摘要。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        checked = list(checked or [])
        missing = list(missing or [])
        exempt = list(exempt or [])
        skipped = list(skipped or [])
        unchecked = list(unchecked or [])
        facts = {"checked": checked, "missing": missing,
                 "exempt": exempt, "action": action}
        if skipped:
            facts["skipped"] = skipped
        if unchecked:
            facts["unchecked"] = unchecked
        turn["answer"]["verify"] = facts
        node("answer.verify", status="degraded" if skipped else ("miss" if missing else "ok"),
             out={"checked": len(checked), "missing_n": len(missing),
                  "missing": "/".join(missing)[:120] or "-",
                  "exempt_n": len(exempt), "unchecked_n": len(unchecked),
                  "unchecked": "/".join(unchecked)[:120] or "-",
                  "skipped": "/".join(skipped) or "-",
                  "action": action})
    except Exception:  # noqa: BLE001
        pass


def retrieval(**facts) -> None:
    """记检索这一段的控制流事实。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        turn["retrieval"].update(facts)
        turn["_retrieved"] = True
        sidecar._event("retrieval", facts=fact_utils.to_facts(facts))
    except Exception:  # noqa: BLE001
        pass


def note_behavior(tag: str, detail: dict = None) -> None:
    """记一个行为标签：tag 取最后一次（主行为），tags 保留全序列。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        turn["tag"] = tag
        turn["tags"].append(tag)
        detail_facts = fact_utils.to_facts(detail or {})
        if detail_facts:
            turn["detail"].update(detail_facts)
        sidecar._event("behavior", tag=tag, facts=detail_facts)
    except Exception:  # noqa: BLE001
        pass


def add_ms(name: str, started_at: float) -> None:
    """记某个环节耗时（传入 start 时的 time.perf_counter()）。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        spent = int((time.perf_counter() - started_at) * 1000)
        turn["ms"][name] = spent
        target = _NODE_OF_STEP.get(name, name)
        for rec in reversed(turn["path"]):
            if rec.get("node") == target:
                rec["ms"] = spent
                break
        sidecar._event("ms", name=name, ms=spent)
    except Exception:  # noqa: BLE001
        pass


def note_error(exc: BaseException) -> None:
    """记本轮异常（异常仍照常向上抛，这里只留痕）。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    try:
        turn["error"] = "%s: %s" % (type(exc).__name__, exc)
        sidecar._event("error", type=type(exc).__name__, message=str(exc)[:200])
    except Exception:  # noqa: BLE001
        pass


def brief(obj, max_str: int = 40):
    """把值压成看得懂又不炸日志的摘要。"""
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
    """兼容 (doc, score, meta) 与裸 doc 两种输入。"""
    if isinstance(item, tuple) and len(item) == 3:
        return item[0], item[1], item[2]
    return item, None, None


def _num(value, ndigits: int = 4):
    try:
        return round(float(value), ndigits) if value is not None else None
    except (TypeError, ValueError):
        return None


def chunk_brief(ranked, n: int = 3) -> list:
    """取前 n 个召回片段的 {file, score} 摘要。"""
    out = []
    try:
        for item in list(ranked)[:n]:
            doc, score, _meta = _as_triple(item)
            meta = getattr(doc, "metadata", None) or {}
            out.append({"file": meta.get("file_name"), "score": _num(score, 3)})
    except Exception:  # noqa: BLE001
        return []
    return out


def chunk_records(ranked, max_chars: int = sidecar.CHUNK_MAX_CHARS) -> list:
    """(doc, score, meta) 列表 → 旁路文件里的片段记录（全文 + 排序 + 分数）。"""
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
    """记本轮召回的片段（含全文），由 end_turn() 落到 chunks 旁路。"""
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


def turn_anchor() -> dict:
    """当前轮的锚点 {date, turn_index}；不在轮次里时为空。"""
    turn = getattr(_local, "turn", None) or {}
    if turn.get("turn_index") is None:
        return {}
    return {"date": turn.get("_day") or (turn.get("ts") or "")[:10],
            "turn_index": turn.get("turn_index")}


def end_turn(aborted: bool = False, reply: str = "") -> None:
    """收尾并落盘（幂等）；写失败只吞掉。"""
    turn = getattr(_local, "turn", None)
    if not turn:
        return
    _local.turn = None
    _local.event_turn = turn
    try:
        turn["aborted"] = bool(aborted)
        started = getattr(_t0, "start", None)
        if started:
            turn["ms"]["total"] = int((time.perf_counter() - started) * 1000)
        sid = turn.get("session_id")
        sidecar._event("end", aborted=bool(aborted),
                       branch=BEHAVIOR_BY_TAG.get(turn.get("tag")),
                       tag=turn.get("tag"), error=turn.get("error"))
        chunk_list = turn.pop("_chunks", None)
        if chunk_list:
            chunks_path = paths._path(sid, sidecar.CHUNK_KIND, create=True,
                                      day=turn.get("_day") or "")
            if chunks_path:
                turn["retrieval"]["evidence"] = {"file": os.path.basename(chunks_path)}
            rewrite = turn["retrieval"].get("rewrite") or {}
            domain = (turn["retrieval"].get("domain") or {}).get("value")
            sidecar._append(chunks_path, {
                "v": sidecar.SIDE_VERSION,
                "ts": turn["ts"],
                "session_id": sid,
                "turn_index": turn.get("turn_index"),
                "query": turn.get("query"),
                "effective_query": rewrite.get("effective_query"),
                "domain": domain,
                "n_chunks": len(chunk_list),
                "chunks": chunk_list,
            })
        _finalize(turn, reply)
        turn.pop("_retrieved", None)
        turn.pop("span_stack", None)
        day = turn.get("_day") or ""
        turn.pop("_day", None)
        sidecar._append(paths._path(sid, create=True, day=day), turn)
        _last[sid] = turn
    except Exception:  # noqa: BLE001
        pass


def _finalize(turn: dict, reply: str) -> None:
    """收尾补齐出口信息：折算唯一出口、补 answer；本轮没检索就补一条 skipped 节点。"""
    tag = turn.get("tag")
    if tag:
        turn["branch"] = BEHAVIOR_BY_TAG.get(tag)
    if turn["branch"]:
        prev = turn.get("answer") or {}
        answer = {"kind": turn["branch"], "model": prev.get("model"),
                  "chars": len(reply or "")}
        if prev.get("verify") is not None:
            answer["verify"] = prev["verify"]
        turn["answer"] = answer
        retrieved = bool(turn.get("_retrieved"))
        if not retrieved:
            turn["path"].append({"id": _next_span_id(turn),
                                 "parent": _current_parent(turn),
                                 "node": "retrieve", "kind": "step",
                                 "status": "skipped", "by": turn["branch"]})
