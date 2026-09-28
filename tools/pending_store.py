"""统一的「待确认」状态存储：SOP 退出确认、网点问城市、网点重名选序号。

Redis 存 `sop:pending:{session_id}`（带 TTL）；无 Redis 时降级内存并做时间戳校验。
`kind` 取值：confirm_exit（答是/不是）、ask_city（答城市名）、pick_city（答序号）；
`confirm_enter` 用于「进入 SOP 的确认门」（弱触发才问）。
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from tools.log_tool import get_logger
from tools.redis_store import get_redis, json_dumps, json_loads, pending_ttl

logger = get_logger(name="pending_store")

# 无 Redis 时的降级存储（带 ts，读时校验是否过期）
_sessions: Dict[str, dict] = {}


def _key(session_id: str) -> str:
    return "sop:pending:%s" % session_id


def _write(session_id: str, state: dict) -> dict:
    state["ts"] = time.time()
    r = get_redis()
    if r is not None:
        r.set(_key(session_id), json_dumps(state), ex=pending_ttl())
    else:
        _sessions[session_id] = state
    return state


def get(session_id: str) -> dict | None:
    """读待确认状态；不存在或已过期返回 None。"""
    r = get_redis()
    if r is not None:
        raw = r.get(_key(session_id))
        return json_loads(raw) if raw else None
    item = _sessions.get(session_id)
    if item is None:
        return None
    if time.time() - float(item.get("ts") or 0) > pending_ttl():
        _sessions.pop(session_id, None)
        logger.info("[Pending] expired (memory fallback) session=%s", session_id)
        return None
    return item


def set(session_id: str, kind: str, **data) -> dict:
    """写待确认状态（带 TTL）；`retry` 缺省为 0。"""
    return _write(session_id, dict(data, kind=kind, retry=int(data.get("retry") or 0)))


def bump_retry(session_id: str) -> Optional[dict]:
    """`retry` +1（用于"再问一次"）；没有状态时返回 None。"""
    state = get(session_id)
    if not state:
        return None
    state["retry"] = int(state.get("retry") or 0) + 1
    return _write(session_id, state)


def clear(session_id: str) -> None:
    """清除待确认状态。"""
    r = get_redis()
    if r is not None:
        r.delete(_key(session_id))
    _sessions.pop(session_id, None)


def has(session_id: str) -> bool:
    return get(session_id) is not None
