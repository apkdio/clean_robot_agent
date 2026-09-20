"""Redis 连接与操作封装（降级可回退）。

统一管理 Redis 连接：懒加载单例 + 连接失败降级为不可用。业务方在
``get_redis()`` 返回 None 时回退到本地内存/文件，保证无 Redis 也能正常运行。

key 约定（统一前缀）：
  - SOP 会话状态：sop:session:{session_id}    (JSON string + TTL)
  - 会话并发锁：  lock:session:{session_id}   (SETNX + EXPIRE)
  - 摄入锁：      lock:ingest                 (SETNX + EXPIRE)
"""

import json

from log_tool import get_logger

logger = get_logger(name="redis_store")

_redis = None
_enabled = None  # None=未探测，True/False=是否可用


def _load_config() -> dict:
    from config_tool import load_config
    try:
        return load_config("redis")
    except Exception:
        return {}


def get_redis():
    """懒加载 Redis 连接（单例）。连接失败返回 None（降级）。"""
    global _redis, _enabled
    if _enabled is False:
        return None
    if _redis is None:
        import redis
        cfg = _load_config()
        try:
            client = redis.Redis(
                host=cfg.get("host", "localhost"),
                port=cfg.get("port", 6379),
                password=cfg.get("password", ""),
                db=cfg.get("db", 0),
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            client.ping()
            _redis = client
            _enabled = True
            logger.info("[Redis] connected %s:%s", cfg.get("host", "localhost"), cfg.get("port", 6379))
        except Exception as e:
            _redis = None
            _enabled = False
            logger.warning("[Redis] unavailable, fallback to local: %s", e)
    return _redis


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(_load_config().get(key, default))
    except Exception:
        return default


def sop_ttl() -> int:
    """SOP 会话状态过期秒数。"""
    return _cfg_int("sop_ttl", 1800)


def lock_ttl() -> int:
    """锁过期秒数（防死锁）。"""
    return _cfg_int("lock_ttl", 60)


def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


# ── 互斥锁（Redis SETNX + 本地降级）──

_local_locks = set()  # Redis 不可用时的本地降级锁（单进程语义）


def acquire_lock(key: str, ttl: int = None) -> bool:
    """获取互斥锁。返回 True=获取成功，False=已被占用。

    Redis 可用走 SETNX + EXPIRE（跨进程/多实例）；不可用回退本地 set。
    """
    r = get_redis()
    if r is not None:
        try:
            return bool(r.set(key, "1", nx=True, ex=ttl or lock_ttl()))
        except Exception:
            pass
    if key in _local_locks:
        return False
    _local_locks.add(key)
    return True


def release_lock(key: str):
    """释放互斥锁。"""
    r = get_redis()
    if r is not None:
        try:
            r.delete(key)
        except Exception:
            pass
    _local_locks.discard(key)
