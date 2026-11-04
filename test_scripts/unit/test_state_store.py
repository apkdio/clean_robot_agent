"""StateStore 契约：双写、取新者、tombstone、本地降级。"""
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from tools.state_store import StateStore


class _FakeRedis:
    def __init__(self):
        self.data = {}
        self.on_set = None

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        if self.on_set:
            self.on_set()
        self.data[key] = value


def _store(local, fake, ttl=900):
    return StateStore("test:", lambda: local, lambda: ttl, redis_getter=lambda: fake)


def test_set_get_and_newer_local_wins():
    local, fake = {}, _FakeRedis()
    store = _store(local, fake)
    store.set("a", {"x": 1})
    _assert_eq(store.get("a"), {"x": 1}, "双写后读回")

    remote = json.loads(fake.data["test:a"])
    remote["updated_at"] = time.time() - 100
    remote["rev"] = 1
    fake.data["test:a"] = json.dumps(remote)
    local["a"] = {"x": 2}
    _assert_eq(store.get("a"), {"x": 2}, "本地新于 Redis 时取本地")
    _assert_eq(json.loads(fake.data["test:a"])["value"], {"x": 2}, "本地新值回填 Redis")


def test_delete_tombstone_does_not_resurrect():
    local, fake = {}, _FakeRedis()
    store = _store(local, fake)
    store.set("a", {"x": 1})
    store.delete("a")
    _assert_eq(store.get("a"), None, "删除后本地不再返回")
    _assert_eq(json.loads(fake.data["test:a"])["deleted"], True, "Redis 写 tombstone")

    local2 = {}
    store2 = _store(local2, fake)
    _assert_eq(store2.get("a"), None, "新进程读到 tombstone 也不会复活")


def test_ttl_expiry():
    local, fake = {}, _FakeRedis()
    store = _store(local, fake, ttl=0)
    store.set("a", {"x": 1})
    _assert_eq(store.get("a"), None, "TTL 0 立即过期")


def test_redis_error_falls_back_local():
    local = {}
    store = _store(local, None)

    class Broken:
        def get(self, key):
            raise RuntimeError("redis down")

        def set(self, key, value, ex=None):
            raise RuntimeError("redis down")

    store._redis_getter = lambda: Broken()
    store.set("a", {"x": 1})
    _assert_eq(store.get("a"), {"x": 1}, "Redis 异常仍可读本地")


def test_delete_writes_remote_before_local():
    """删除时先落 Redis tombstone，再清本地；缩小进程崩溃窗口。"""
    local, fake = {}, _FakeRedis()
    store = _store(local, fake)
    store.set("a", {"x": 1})
    seen = []
    fake.on_set = lambda: seen.append("a" in local)
    store.delete("a")
    _assert_eq(seen[-1], True, "写远端 tombstone 时本地旧值尚未删除")
    _assert_eq(store.get("a"), None, "删除后不再返回")


def test_tombstone_cleanup_after_2x_ttl():
    """本地 tombstone 超过 2x TTL 后清理，active 记录不受影响。"""
    local, fake = {}, _FakeRedis()
    store = _store(local, fake, ttl=10)
    store.set("active", {"x": 1})
    store.set("deleted", {"x": 1})
    store.delete("deleted")
    store._meta["deleted"]["updated_at"] = time.time() - 21
    store._cleanup_tombstones()
    _assert("deleted" not in store._meta, "过期 tombstone 被清理")
    _assert("active" in store._meta, "active 记录不被误清")


TESTS = [test_set_get_and_newer_local_wins, test_delete_tombstone_does_not_resurrect,
         test_ttl_expiry, test_redis_error_falls_back_local,
         test_delete_writes_remote_before_local, test_tombstone_cleanup_after_2x_ttl]


def run():
    reset()
    return run_tests("StateStore 契约（双写/版本/tombstone/降级）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
