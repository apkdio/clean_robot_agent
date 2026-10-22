"""统一的澄清（二次确认）状态存储：所有"问了要用户再回一句"的环节共用一份状态。

边界：这里只做**存取、校验与过期**，不解析「是/不是」、不决定业务怎么走。业务方按
`biz` 认领自己的状态，再执行本业务逻辑：

    sop.exit     SOP 退出确认门
    sop.enter    SOP 进入确认门（所有 SOP 默认先问）
    ops.clarify  操作判别器澄清（多候选分差过近，选 1/2）

业务 id 隔离：`take` / `bump` / `close` 都要求 biz 匹配——A 业务的回复不会被 B 业务消费，
也不会误清别人的状态（不匹配时打 WARNING 并放弃本次操作）。

记录形状：{"v": 1, "biz": ..., "ts": ..., "retry": 0, "state": {...}, "data": {...}}
  - `state`：业务方给的当前状态（SOP 的 step / slots、原问题等），本模块原样存取、不做解释；
  - `data`：澄清展示用的载荷（选项、标签等）。
Redis 存 clarify:{session_id}（TTL 见 redis_store.clarify_ttl）；无 Redis 降级内存并校验时间戳。
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from tools.log_tool import get_logger
from tools.redis_store import clarify_ttl, get_redis, json_dumps, json_loads

logger = get_logger(name="pending_store")

BIZ_SOP_EXIT = "sop.exit"
BIZ_SOP_ENTER = "sop.enter"
BIZ_OPS_CLARIFY = "ops.clarify"

# 无 Redis 时的降级存储（带 ts，读时校验是否过期）
_sessions: Dict[str, dict] = {}


def _key(session_id: str) -> str:
    return "clarify:%s" % session_id


def _write(session_id: str, state: dict) -> dict:
    state["v"] = 1
    state["ts"] = time.time()
    r = get_redis()
    if r is not None:
        r.set(_key(session_id), json_dumps(state), ex=clarify_ttl())
    else:
        _sessions[session_id] = state
    return state


def peek(session_id: str) -> dict | None:
    """读当前澄清状态；不存在或已过期返回 None。"""
    r = get_redis()
    if r is not None:
        raw = r.get(_key(session_id))
        return json_loads(raw) if raw else None
    item = _sessions.get(session_id)
    if item is None:
        return None
    if time.time() - float(item.get("ts") or 0) > clarify_ttl():
        _sessions.pop(session_id, None)
        logger.info("[Clarify] expired (memory fallback) session=%s", session_id)
        return None
    return item


def ask(session_id: str, biz: str, state: dict = None, **data) -> dict:
    """开一个澄清：先清掉该会话已有的澄清标记，再写 {biz, retry=0, state, data}。

    业务方要把自己的当前状态放进 `state`（如 SOP 的 step/slots、原问题），二次进入时
    `take` 拿回来自己判断是否还对得上；本模块不理解它。
    """
    if not isinstance(biz, str) or not biz.strip():
        raise ValueError("clarify biz 不能为空")
    prev = peek(session_id)
    if prev:
        if prev.get("biz") == biz:
            logger.info("[Clarify] 重建 %s 的澄清标记 session=%s", biz, session_id)
        else:
            logger.warning("[Clarify] 覆盖未完成的澄清：%s → %s session=%s",
                           prev.get("biz"), biz, session_id)
        close(session_id)
    rec = {"biz": biz, "retry": 0, "state": state or {}, "data": dict(data)}
    json_dumps(rec)                        # 状态要能序列化（存不进去比存错了更难查）
    return _write(session_id, rec)


def take(session_id: str, biz: str) -> dict | None:
    """业务认领：只有 biz 匹配才返回状态（不清除，由业务决定 close）。"""
    state = peek(session_id)
    if not state:
        return None
    if state.get("biz") != biz:
        logger.warning("[Clarify] biz 不匹配：期望 %s，当前 %s（状态保留）", biz, state.get("biz"))
        return None
    return state


def bump(session_id: str, biz: str) -> Optional[dict]:
    """retry +1（用于"再问一次"）；状态不存在或不属于该业务时返回 None。"""
    state = take(session_id, biz)
    if not state:
        return None
    state["retry"] = int(state.get("retry") or 0) + 1
    return _write(session_id, state)


def close(session_id: str, biz: str = None) -> bool:
    """关闭澄清状态。给了 biz 时只清自己的；不匹配则不动并返回 False。"""
    if biz is not None:
        state = peek(session_id)
        if state and state.get("biz") != biz:
            logger.warning("[Clarify] 拒绝关闭别人的澄清：期望 %s，当前 %s", biz, state.get("biz"))
            return False
    r = get_redis()
    if r is not None:
        r.delete(_key(session_id))
    _sessions.pop(session_id, None)
    return True


def has(session_id: str, biz: str = None) -> bool:
    state = peek(session_id)
    if not state:
        return False
    return biz is None or state.get("biz") == biz
