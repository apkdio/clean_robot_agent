"""澄清注册、A10 与安全决策契约。"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *


def _clean(sid):
    from test_scripts.test_hooks import reset as reset_session

    reset_session(sid)


def test_safety_decision_levels():
    from service.safety import (
        SAFETY_CARRY, SAFETY_HINT, SAFETY_NONE, SAFETY_STOP,
        SafetyDecision, decide_safety,
    )
    from test_scripts.test_hooks import set_danger_window

    sid = "t-s5-safety"
    _clean(sid)
    try:
        set_danger_window(sid, "漏电")
        carry = decide_safety(sid, "那这个电流现象影响大吗", mode=SAFETY_CARRY)
        _assert_eq(carry.level, SAFETY_CARRY, "窗口告警 → 硬承接")
        _assert_in("安全风险", carry.reply, "承接话术来自配置")
        hint = decide_safety(sid, mode=SAFETY_HINT)
        _assert_eq(hint.level, SAFETY_HINT, "窗口告警 → 软提醒")
        _assert_in("安全提醒", hint.reply, "软提醒话术来自配置")
        stop = decide_safety(sid, "机器人冒烟了", mode=SAFETY_STOP)
        _assert_eq(stop.level, SAFETY_STOP, "当前句危险词 → 停机")
        _assert_eq(stop.word, "冒烟", "带命中的危险词")
        _assert_eq(decide_safety(sid, "边刷多久换一次", mode=SAFETY_STOP).level,
                   SAFETY_NONE, "普通问题不停机")
        _assert_raises(ValueError, lambda: SafetyDecision("bogus"), "级别闭集")
        _assert_raises(ValueError, lambda: decide_safety(sid, mode="bogus"), "模式闭集")
    finally:
        _clean(sid)


def test_unknown_clarify_biz_is_consumed():
    from tools import pending_store, trace_store
    from service.clarify_service import dispatch_clarify
    from test_scripts import trace_reader

    sid = "t-s5-clarify-unknown"
    _clean(sid)
    pending_store.ask(sid, "unknown.biz", state={"x": 1})
    trace_store.begin_turn(sid, "随便看看")
    try:
        result = dispatch_clarify(sid, "unknown.biz", "随便看看", "随便看看")
        trace_store.end_turn(reply="")
        row = trace_reader.load(trace_store._path(sid))[-1]
    finally:
        _clean(sid)
    _assert_eq(result.code, "drop", "未知 biz 显式 drop")
    _assert_eq(pending_store.peek(sid), None, "未知 biz 状态被消费")
    node = next((n for n in (row.get("path") or []) if n.get("node") == "clarify"), None)
    _assert(node is not None, "落了 clarify 节点")
    _assert_eq(node.get("status"), "skipped", "标 skipped")
    _assert_eq((node.get("out") or {}).get("by"), "no_handler", "标出原因")


def test_sop_enter_missing_sop_drops():
    from service.sops.sop_gate import _clarify_sop_enter
    from tools import pending_store

    sid = "t-s5-enter-missing"
    _clean(sid)
    pending_store.ask(sid, pending_store.BIZ_SOP_ENTER, state={})
    try:
        result = _clarify_sop_enter(sid, "是", "是")
        _assert_eq(result.code, "drop", "缺 sop 不再默认 model_query")
        _assert_eq(pending_store.peek(sid), None, "缺 sop 状态被清掉")
    finally:
        _clean(sid)


TESTS = [
    test_safety_decision_levels,
    test_unknown_clarify_biz_is_consumed,
    test_sop_enter_missing_sop_drops,
]


def run():
    reset()
    return run_tests("澄清注册 / A10 / 安全决策契约", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
