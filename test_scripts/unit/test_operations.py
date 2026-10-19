"""操作判别器测试：注册表完整性 + 入口判据 + 三态出口。

判据本身由现成实现保证，这里只测接线与三态是否符合设计。"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 与 test_scripts._runner 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from tools import operations as ops


def test_registry_shape():
    """注册表与表序一一对应，每条都有 predicate、allow 是 tuple 或 None。"""
    for name, (pred, allow, order) in ops.OPERATIONS.items():
        _assert(callable(pred), "%s 有 predicate" % name)
        _assert(allow is None or isinstance(allow, tuple), "%s 的 allow 是 tuple 或 None" % name)
        _assert_eq(order, ops.ORDER.index(name), "%s 的 order 与 ORDER 对齐" % name)
    _assert_eq(sorted(ops.OPERATIONS), sorted(ops.ORDER), "注册表与表序一一对应")


def test_probe_hits():
    """典型问法命中哪些操作；真判据比指代词表窄，断言以真判据为准。"""
    hit = ops.probe("离我最近的维修点")
    _assert_in("service_point", hit, "网点词 → service_point")
    _assert_in("sop", hit, "「维修」同时撞 SOP 弱触发（真实竞争，靠表序裁决）")
    _assert_in("date", hit, "「最近」会误撞时间词表——已知假阳性，靠 allow/判据收窄")
    casual = ops.probe("那你可以陪我玩过家家吗，班里这个游戏很火")
    _assert("model" not in casual, "闲聊句不命中 model（真判据比指代词表窄）")


def test_resolve_three_states():
    _assert_eq(ops.resolve(["service_point"], intent="robot"), ("win", "service_point"), "单命中 → win")
    _assert_eq(ops.resolve(["service_point", "sop"], intent="robot", margin=0.9),
               ("win", "service_point"), "多命中、未过阈值门 → 按表序裁决（与现链路一致）")
    _assert_eq(ops.resolve(["service_point", "sop"], intent="robot", margin=0.05),
               ("ambiguous", "service_point"), "多命中且低置信 → ambiguous（该问用户）")
    _assert_eq(ops.resolve(["model", "sop"], intent="casual", margin=0.9), ("none", None),
               "意图不允许 → none（假阳性被 allow 挡住）")
    _assert_eq(ops.resolve([], intent="robot"), ("none", None), "没有命中 → none")


def test_note_ops_why_none():
    """state=none 时 why 必须是 none（S-16）：intent=other/casual 的常态轮，
    被标成 order 会把「无可用操作」读成「按表序裁决过」。"""
    note = ops.note_ops("谢谢你的回答", intent="casual", margin=0.9)
    _assert_eq(note.get("state"), "none", "无可用操作 → state=none")
    _assert_eq(note.get("why"), "none", "why=none（不再误标 order）")
    _assert_eq(ops.note_ops("离我最近的维修点", intent="robot", margin=0.9).get("why"), "order",
               "多命中且高置信 → why=order（不受影响）")


def test_note_ops_writes_route_and_returns_usable():
    """判定结果落 trace 的 route.ops，并把过 allow 门的候选交给主链路派发。"""
    from tools import trace_store
    from test_scripts import trace_reader
    reset_trace("ops1")
    trace_store.begin_turn("ops1", "离我最近的维修点")
    note = ops.note_ops("离我最近的维修点", intent="robot", margin=0.9)
    trace_store.end_turn()
    rows = trace_reader.load(trace_store._path("ops1"))
    got = ((rows[0].get("route") or {}).get("ops") if rows else None) or {}
    _assert_eq(got.get("state"), "win", "落 state")
    _assert_eq(got.get("ops"), "service_point/sop/date/domain", "落候选（按表序、/ 连接）")
    _assert_eq(got.get("n"), 4, "落候选数")
    _assert_eq(got.get("winner"), "service_point", "落表序裁决结果")
    _assert_eq(note.get("usable"), ["service_point", "sop", "date", "domain"],
               "usable 按表序给出（主链路照它派发）")
    # 意图不在 allow 门内 → usable 为空（假阳性被挡住）
    _assert_eq(ops.note_ops("这两个有什么区别", intent="casual", margin=0.9).get("usable"), [],
               "casual 时没有可用操作")


TESTS = [test_registry_shape, test_probe_hits, test_resolve_three_states, test_note_ops_why_none,
         test_note_ops_writes_route_and_returns_usable]


def run():
    reset()
    return run_tests("操作判别器测试（注册表 / 入口判据 / 三态）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
