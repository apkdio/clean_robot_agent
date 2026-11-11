"""Redis + 本地双写的状态存储：统一 SOP、澄清、停止标志的状态生命周期。

本地副本保留原始值，Redis 存带版本的 envelope。读取时两侧都看，按
(updated_at, rev) 取较新者；删除写 tombstone，避免 Redis 恢复后旧值复活。
Redis 不可用或运行中异常时降级本地，并记 WARNING。

本地 tombstone 保留 2 倍业务 TTL 后清理。这个折中成立的前提是：单进程、
删除后没有其他 worker 刷新旧状态、旧 Redis key 带 TTL、进程期间不重启；
如果 Redis tombstone 写入失败、旧 key 无 TTL 或进程重启，仍可能在该窗口
内读到旧 active 状态，需要持久化 tombstone 或启动对账才能彻底消除。
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from tools.log_tool import get_logger
from gateway import redis_store

logger = get_logger(name="state_store")

_SENTINEL = object()
_TOMBSTONE_TTL_FACTOR = 2.0
_TOMBSTONE_CLEAN_INTERVAL = 60.0


class StateStore:
    """一个命名空间的状态存储；local_provider 返回当前本地 dict（便于测试重置）。"""

    def __init__(self, prefix: str, local_provider: Callable[[], dict],
                 ttl_provider: Callable[[], int], redis_getter: Callable | None = None):
        self.prefix = prefix
        self._local_provider = local_provider
        self._ttl = ttl_provider
        self._redis_getter = redis_getter or (lambda: redis_store.get_redis())
        self._meta = {}
        self._last_tombstone_cleanup = 0.0
        self._lock = threading.RLock()

    def _key(self, local_id: str) -> str:
        return f"{self.prefix}{local_id}"

    def _local(self) -> dict:
        return self._local_provider()

    @staticmethod
    def _record(value, rev=0, updated_at=0.0, deleted=False):
        # rev用于单例进程内的该条记录版本维护（即该会话是第几次写）
        return {"value": value, "rev": rev, "updated_at": updated_at, "deleted": deleted}

    def _read_remote(self, local_id: str):
        r = self._redis_getter()
        if r is None:
            return None
        try:
            raw = r.get(self._key(local_id))
            if not raw:
                return None
            obj = redis_store.json_loads(raw)
            if isinstance(obj, dict) and obj.get("__state__") == 1:
                return self._record(obj.get("value"), int(obj.get("rev") or 0),
                                    float(obj.get("updated_at") or 0),
                                    bool(obj.get("deleted")))
            # 兼容旧版直接存原始 dict 的 Redis 值；无流量时会在下一次写入被 envelope 替换。
            updated = float((obj or {}).get("ts") or 0) if isinstance(obj, dict) else 0.0
            return self._record(obj, 0, updated, False)
        except Exception as e:  # noqa: BLE001
            logger.warning("[StateStore] redis get failed (%s): %s", self._key(local_id), e)
            redis_store.mark_unavailable()
            return None

    def _read_local(self, local_id: str):
        local = self._local()
        # 处理存在、不存在、存在但是value为None的三态情况
        # _SENTINEL用于区分第三种情况
        value = local.get(local_id, _SENTINEL)
        meta = self._meta.get(local_id)
        if value is _SENTINEL:
            if meta is None:
                return None
            # 本地原件被外部重置（测试/清理）时，不把旧 meta 当成仍有效的值。
            return meta if meta.get("deleted") else None
        # 会话存在但是meta不存在
        if meta is None:
            updated = float((value or {}).get("ts") or 0) if isinstance(value, dict) else 0.0
            return self._record(value, 0, updated, False)
        meta = dict(meta)
        meta["value"] = value
        # 兼容旧测试/旧本地态直接改 ts 的过期方式：显式 ts 优先于内部 updated_at。
        if isinstance(value, dict) and value.get("ts"):
            meta["updated_at"] = float(value["ts"])
        return meta

    def _write_remote(self, local_id: str, record: dict, ttl: int | None = None) -> None:
        r = self._redis_getter()
        if r is None:
            return
        try:
            r.set(self._key(local_id), redis_store.json_dumps({
                "__state__": 1,
                "value": record.get("value"),
                "rev": record.get("rev", 0),
                "updated_at": record.get("updated_at", 0),
                "deleted": bool(record.get("deleted")),
            }), ex=max(1, int(ttl if ttl is not None else self._ttl())))
        except Exception as e:  # noqa: BLE001
            logger.warning("[StateStore] redis set failed (%s): %s", self._key(local_id), e)
            redis_store.mark_unavailable()

    def _store_local(self, local_id: str, record: dict) -> None:
        local = self._local()
        if record.get("deleted"):
            local.pop(local_id, None)
        else:
            local[local_id] = record.get("value")
        self._meta[local_id] = dict(record)

    def _cleanup_tombstones(self, now: float | None = None) -> None:
        """清理超过 2x TTL 的本地 tombstone。

        仅作为当前单例决策。局限： Redis tombstone 写失败、进程重启或并发旧写者
        仍可能让旧 active 在窗口内复活；这里只控制本地内存增长。
        """
        now = time.time() if now is None else now
        ttl = max(0, int(self._ttl()))
        horizon = ttl * _TOMBSTONE_TTL_FACTOR
        for local_id, record in list(self._meta.items()):
            if not record.get("deleted"):
                continue
            if now - float(record.get("updated_at") or 0) > horizon:
                self._meta.pop(local_id, None)

    def _maybe_cleanup_tombstones(self) -> None:
        now = time.time()
        if now - self._last_tombstone_cleanup < _TOMBSTONE_CLEAN_INTERVAL:
            return
        self._last_tombstone_cleanup = now
        self._cleanup_tombstones(now)

    def get(self, local_id: str, validate: Callable | None = None):
        """读取有效状态；过期/校验失败/已删除返回 None。"""
        with self._lock:
            self._maybe_cleanup_tombstones()
            local = self._read_local(local_id)
            remote = self._read_remote(local_id)
            candidates = [x for x in (local, remote) if x is not None]
            if not candidates:
                return None
            record = max(candidates, key=lambda x: (float(x.get("updated_at") or 0),
                                                    int(x.get("rev") or 0)))
            if record.get("deleted"):
                self._meta[local_id] = dict(record)
                self._local().pop(local_id, None)
                return None
            updated = float(record.get("updated_at") or 0)
            if updated and time.time() - updated > max(0, int(self._ttl())):
                self.delete(local_id)
                logger.info("[StateStore] expired: %s", self._key(local_id))
                return None
            value = record.get("value")
            if validate is not None and not validate(value):
                self.delete(local_id)
                return None
            # 回填时先写远端：进程若在本地回填前崩掉，Redis 仍保有较新的记录。
            if remote is None or (remote.get("updated_at", 0), remote.get("rev", 0)) < (
                    record.get("updated_at", 0), record.get("rev", 0)):
                self._write_remote(local_id, record)
            self._store_local(local_id, record)
            return value

    def set(self, local_id: str, value, ttl: int | None = None) -> None:
        """写入新状态；rev 递增，双写 Redis/本地。"""
        with self._lock:
            self._maybe_cleanup_tombstones()
            old = self._read_local(local_id)
            remote = self._read_remote(local_id)
            rev = max([int((old or {}).get("rev") or 0),
                       int((remote or {}).get("rev") or 0)]) + 1
            record = self._record(value, rev, time.time(), False)
            # Redis 优先：进程在本地写入前崩溃时，已写入共享存储的新状态不会丢。
            self._write_remote(local_id, record, ttl)
            self._store_local(local_id, record)

    def delete(self, local_id: str) -> None:
        """删除状态；写 tombstone，防止另一侧旧副本复活。"""
        with self._lock:
            old = self._read_local(local_id)
            remote = self._read_remote(local_id)
            rev = max([int((old or {}).get("rev") or 0),
                       int((remote or {}).get("rev") or 0)]) + 1
            record = self._record(None, rev, time.time(), True)
            # tombstone 先落 Redis：删除后进程崩溃也不会让旧 active 状态复活。
            self._write_remote(local_id, record)
            self._store_local(local_id, record)

    def pop(self, local_id: str, validate: Callable | None = None):
        """取走并删除；无有效状态返回 None。"""
        with self._lock:
            value = self.get(local_id, validate=validate)
            if value is None:
                return None
            self.delete(local_id)
            return value

    def reset(self) -> None:
        """清空当前进程本地副本与版本元数据；Redis 由调用方/测试自行清理。"""
        with self._lock:
            self._local().clear()
            self._meta.clear()
            self._last_tombstone_cleanup = 0.0
