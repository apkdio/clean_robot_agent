"""Redis 连接与操作封装（懒加载单例；连接失败降级为不可用，业务方回退本地内存/文件）。

key 约定：
  - SOP 会话状态：sop:session:{session_id}    (JSON string + TTL)
  - 会话并发锁：  lock:session:{session_id}   (SETNX + EXPIRE)
  - 摄入锁：      lock:ingest                 (SETNX + EXPIRE)
"""

import json
import time

from tools.log_tool import get_logger

logger = get_logger(name="redis_store")

_redis = None
_enabled = None  # None=未探测，True/False=是否可用


def _load_config() -> dict:
    from tools.config_tool import load_config
    try:
        return load_config("redis")
    except Exception:
        return {}


def get_redis():
    """懒加载 Redis 连接（单例）。包未安装或连接失败都返回 None（降级本地内存）。"""
    global _redis, _enabled
    if _enabled is False:
        return None
    if _redis is None:
        cfg = _load_config()
        try:
            import redis
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
        except ImportError as e:
            _redis = None
            _enabled = False
            logger.warning("[Redis] package not installed, fallback to local: %s", e)
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
    """SOP 会话状态过期秒数（滑动：每轮重盖）。"""
    return _cfg_int("sop_ttl", 900)


def lock_ttl() -> int:
    """锁过期秒数（防死锁）。"""
    return _cfg_int("lock_ttl", 60)


def pending_ttl() -> int:
    """待确认状态（SOP 退出确认 / 进入确认门）的过期秒数。

    默认**跟随 sop_ttl**：pending 先过期而 SOP 还活着时，「是」会被当成槽位输入。
    取同一个数就不必靠人记得同步（配置里显式给了才覆盖）。
    """
    return _cfg_int("pending_ttl", sop_ttl())


def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


# ── 互斥锁（Redis SETNX + 本地降级）──

# key → 过期时间戳。Redis 不可用时的本地降级锁（单进程语义）；
# 必须带过期：Redis 路径靠 EX 自动释放，本地态没有 TTL 就会永久占住（进程内死锁）
_local_locks = {}


def acquire_lock(key: str, ttl: int = None) -> bool:
    """获取互斥锁。返回 True=获取成功，False=已被占用。

    Redis 可用走 SETNX + EXPIRE（跨进程/多实例）；不可用回退本地 set。
    """
    r = get_redis()
    if r is not None:
        try:
            return bool(r.set(key, "1", nx=True, ex=ttl or lock_ttl()))
        except Exception as e:
            logger.warning("[Redis] acquire_lock failed, fallback to local lock: %s", e)
    now = time.time()
    expire = _local_locks.get(key)
    if expire is not None and expire > now:
        return False
    _local_locks[key] = now + (ttl or lock_ttl())
    return True


def release_lock(key: str):
    """释放互斥锁。"""
    r = get_redis()
    if r is not None:
        try:
            r.delete(key)
        except Exception as e:
            logger.warning("[Redis] release_lock failed (本地也会释放): %s", e)
    _local_locks.pop(key, None)
