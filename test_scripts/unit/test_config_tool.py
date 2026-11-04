"""配置入口契约：统一缓存、可选缺省与显式重载。"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from tools.config_tool import get_config, load_config, reload_configs


def test_same_config_is_cached():
    first = get_config("agent")
    second = get_config("agent")
    _assert(first is second, "同一配置只解析一次")


def test_reload_replaces_cache():
    before = get_config("agent")
    reload_configs()
    after = get_config("agent")
    _assert(before is not after, "reload_configs 后重新解析")


def test_optional_missing_returns_default():
    name = "missing-config-for-test"
    _assert_eq(get_config(name, required=False, default={"ok": True}),
               {"ok": True}, "非必需配置缺失返回 default")


def test_required_missing_raises():
    name = "missing-config-for-test"
    _assert_raises(FileNotFoundError, lambda: get_config(name, required=True),
                   "必需配置缺失给出 FileNotFoundError")


def test_load_config_uses_cache():
    first = load_config("rag")
    second = load_config("rag")
    _assert(first is second, "load_config 也走统一缓存")


TESTS = [test_same_config_is_cached, test_reload_replaces_cache,
         test_optional_missing_returns_default, test_required_missing_raises,
         test_load_config_uses_cache]


def run():
    reset()
    try:
        return run_tests("配置入口契约（缓存/缺省/重载）", TESTS)
    finally:
        reload_configs()


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
