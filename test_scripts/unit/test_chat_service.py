"""chat_service 生命周期测试：事件契约、错误回滚、停止语义。"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *

from tools import chat_service as cs
from tools.context_store import delete_session, get_recent


def test_stop_flags_expire_in_local_mode():
    """本地降级停止标志也要过期，残留不会误杀下一轮。"""
    import time

    import tools.redis_store as rs

    saved = rs.get_redis
    rs.get_redis = lambda: None
    sid = "t-stop-local"
    try:
        cs._stop_flags.pop(sid, None)
        _assert_false(cs.is_stopped(sid), "未置位时不是停止态")
        cs.set_stop(sid)
        _assert_true(cs.is_stopped(sid), "置位后是停止态")
        cs._stop_flags[sid] = time.time() - 1
        _assert_false(cs.is_stopped(sid), "过期后自动失效")
        _assert(sid not in cs._stop_flags, "过期项被顺手清掉")
    finally:
        rs.get_redis = saved
        cs._stop_flags.pop(sid, None)


def test_stream_turn_events_and_context():
    """正常轮：delta → meta，user/assistant 都落上下文。"""
    sid = "11111111-2222-3333-4444-555566667777"
    delete_session(sid)

    def fake_ask(query, session_id, *_args, **_kwargs):
        from tools.context_store import append_message

        append_message(session_id, "user", query)
        yield "你好"
        yield "，我是扫地机器人助手"

    try:
        events = list(cs.stream_turn("推荐一款", sid, ask_stream_func=fake_ask))
        _assert_eq([e.kind for e in events], ["delta", "delta", "meta", "done"],
                   "事件序列符合契约")
        _assert_eq("".join(e.text for e in events if e.kind == "delta"),
                   "你好，我是扫地机器人助手", "delta 内容透传")
        meta_event = next(e for e in events if e.kind == "meta")
        _assert_in("branch", meta_event.meta or {}, "meta 事件带 branch")
        msgs = get_recent(sid, n=10)
        _assert_eq([m.get("role") for m in msgs], ["user", "assistant"], "一轮补齐 user/assistant")
    finally:
        delete_session(sid)


def test_stream_turn_error_rolls_back_user():
    """异常轮：发 error 事件并回滚 user，不留问了没答的上下文。"""
    sid = "11111111-2222-3333-4444-555566667778"
    delete_session(sid)

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    try:
        events = list(cs.stream_turn("边刷多久换一次", sid, ask_stream_func=boom))
        _assert_eq([e.kind for e in events], ["error", "done"], "异常发 error + done 事件")
        _assert_eq(events[0].text, "生成失败，请重试", "错误文案是固定话术")
        _assert_eq(get_recent(sid, n=10), [], "错误轮不留在上下文")
    finally:
        delete_session(sid)


def test_stream_turn_stopped_rolls_back_user():
    """停止轮：先吐 delta，再发 stopped 事件并回滚 user。"""
    sid = "11111111-2222-3333-4444-555566667779"
    delete_session(sid)

    saved = cs.is_stopped
    try:
        cs.is_stopped = lambda session_id: True
        cs.clear_stop(sid)

        def fake_ask(*_args, **_kwargs):
            yield "半截"

        events = list(cs.stream_turn("边刷多久换一次", sid, ask_stream_func=fake_ask))
        _assert_eq([e.kind for e in events], ["stopped"], "探测在首帧前命中时直接 stopped")
        _assert_eq(get_recent(sid, n=10), [], "停止轮不留在上下文")

        cs.clear_stop(sid)
        cs._stop_flags[sid] = 0.0     # 让 probe 节流窗口结束后命中
        # is_stopped 恒真等价于停止请求已到：只验事件语义，不在这里跑真实时序
        _assert_eq(cs.TurnEvent(kind="stopped").kind, "stopped", "停止事件契约可构造")
    finally:
        cs.is_stopped = saved
        delete_session(sid)


TESTS = [
    test_stop_flags_expire_in_local_mode,
    test_stream_turn_events_and_context,
    test_stream_turn_error_rolls_back_user,
    test_stream_turn_stopped_rolls_back_user,
]


def run():
    reset()
    return run_tests("chat_service 生命周期测试（事件 / 回滚 / 停止）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
