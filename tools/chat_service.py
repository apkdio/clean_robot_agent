"""一轮对话的生命周期：锁、停止、上下文收尾、标题与 meta。

webapp 只负责 HTTP/SSE 序列化；本模块产出的 TurnEvent 是统一交付事件。
"""
from __future__ import annotations

import time
from typing import Callable, Iterator

from tools.dto.transport import TurnEvent
from tools.log_tool import get_logger

logger = get_logger(name="chat_service")

_STOP_TTL = 120
_stop_flags = {}


def set_stop(session_id: str) -> None:
    """设置停止标志（Redis + 本地降级，均带 TTL）。"""
    from tools.redis_store import get_redis

    r = get_redis()
    if r is not None:
        r.set(f"stop:{session_id}", "1", ex=_STOP_TTL)
    _stop_flags[session_id] = time.time() + _STOP_TTL


def is_stopped(session_id: str) -> bool:
    """查询停止标志；过期即清理，不误杀下一轮。"""
    from tools.redis_store import get_redis

    r = get_redis()
    if r is not None:
        return bool(r.exists(f"stop:{session_id}"))
    expire = _stop_flags.get(session_id)
    if expire is None:
        return False
    if expire <= time.time():
        _stop_flags.pop(session_id, None)
        return False
    return True


def clear_stop(session_id: str) -> None:
    from tools.redis_store import get_redis

    r = get_redis()
    if r is not None:
        r.delete(f"stop:{session_id}")
    _stop_flags.pop(session_id, None)


def make_stop_probe(session_id: str, min_interval: float = 0.15) -> Callable[[], bool]:
    """节流的停止探测：流式循环每个 chunk 查一次，节流到至多 ~7 次/秒。"""
    last = [0.0]

    def probe() -> bool:
        now = time.monotonic()
        if now - last[0] >= min_interval:
            last[0] = now
            return is_stopped(session_id)
        return False

    return probe


def acquire_turn_lock(session_id: str) -> bool:
    from tools.redis_store import acquire_lock

    return acquire_lock(f"lock:session:{session_id}")


def release_turn_lock(session_id: str) -> None:
    from tools.redis_store import release_lock

    release_lock(f"lock:session:{session_id}")


def _rollback_user(session_id: str) -> None:
    from tools.context_store import rollback_last_user_message

    try:
        rollback_last_user_message(session_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("[ChatService] rollback user failed: %s", e)


def _finish_turn(session_id: str, query: str, answer: str) -> dict:
    """正常收尾：assistant 落盘、首轮生成标题、返回 meta。"""
    from tools.context_store import (
        append_message, generate_session_title, get_session_title, set_session_title,
    )
    from sops.base import get_last_recommend
    from sops import has_active_sop
    from tools import trace_store

    append_message(session_id, "assistant", answer,
                   models=get_last_recommend(session_id))
    meta = {"branch": "", "sop_active": bool(has_active_sop(session_id)),
            "anchor": None}
    try:
        if not get_session_title(session_id):
            set_session_title(session_id, generate_session_title(query, answer))
    except Exception as e:  # noqa: BLE001
        logger.warning("[ChatService] generate title failed: %s", e)
    try:
        turn = trace_store.find_turn(session_id, query) or {}
        meta = {"branch": turn.get("branch") or "",
                "sop_active": bool(has_active_sop(session_id)),
                "anchor": {"date": turn.get("date"),
                           "turn_index": turn.get("turn_index")}}
    except Exception as e:  # noqa: BLE001
        logger.warning("[ChatService] build meta failed: %s", e)
    return meta


def stream_turn(query: str, session_id: str, lng=None, lat=None,
                retry: bool = False, ask_stream_func=None) -> Iterator[TurnEvent]:
    """跑完一轮问答并交付统一事件。

    调用方需先 acquire_turn_lock；停止 / 异常 / 断连的收尾都在这里完成。
    """
    if ask_stream_func is None:
        from tools.agent import ask_stream as ask_stream_func

    probe = make_stop_probe(session_id)
    parts: list[str] = []
    completed = False
    stopped = False
    try:
        try:
            for chunk in ask_stream_func(query, session_id, lng=lng, lat=lat,
                                         retry=retry, stop_check=probe):
                if probe():
                    stopped = True
                    break
                parts.append(chunk)
                yield TurnEvent(kind="delta", text=chunk)
            if not stopped and is_stopped(session_id):
                stopped = True
        except Exception as e:  # noqa: BLE001
            logger.error("[ChatService] turn failed: %s", e)
            _rollback_user(session_id)
            yield TurnEvent(kind="error", text="生成失败，请重试")
            yield TurnEvent(kind="done")
            completed = True
            return
        if stopped:
            _rollback_user(session_id)
            yield TurnEvent(kind="stopped")
            completed = True
            return
        meta = _finish_turn(session_id, query, "".join(parts))
        yield TurnEvent(kind="meta", meta=meta)
        yield TurnEvent(kind="done")
        completed = True
    finally:
        if not completed:
            _rollback_user(session_id)
