"""trace 拆分契约：薄壳只能 re-export，不能复制线程状态/路径缓存。"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from mapper import trace_store
from mapper.trace import paths, read, sidecar, write


def test_facade_reexports_public_api():
    _assert_true(trace_store._path is paths._path, "_path 来自 paths")
    _assert_true(trace_store.begin_turn is write.begin_turn, "begin_turn 来自 write")
    _assert_true(trace_store.latest_turn is read.latest_turn, "latest_turn 来自 read")
    _assert_true(trace_store.feedback_turn is sidecar.feedback_turn, "feedback_turn 来自 sidecar")


def test_sop_span_contains_internal_node():
    """SOP 内部节点必须挂到 stage.sop，并且 span.begin 早于节点事件。"""
    sid = "tracetest-sop-span"
    trace_store.begin_turn(sid, "机器人不动了")
    stage_id = trace_store.begin_span("stage.sop", kind="stage")
    trace_store.node("sop.symptom", in_={"text": "不动"}, out={"via": "rule"})
    trace_store.end_span(stage_id, status="hit")
    trace_store.note_behavior("sop_step")
    trace_store.end_turn(reply="排查步骤")

    events = trace_store.events_of(sid)
    begin = next(i for i, e in enumerate(events)
                 if e.get("event") == "span.begin" and e.get("name") == "stage.sop")
    symptom = next(i for i, e in enumerate(events)
                   if e.get("event") == "node" and e.get("name") == "sop.symptom")
    _assert_true(begin < symptom, "stage.sop span.begin 早于 sop.symptom")
    row = read_last_turn(sid)
    path = row.get("path") or []
    stage = next(n for n in path if n.get("node") == "stage.sop")
    child = next(n for n in path if n.get("node") == "sop.symptom")
    _assert_eq(stage.get("id"), stage_id, "stage span id 固定")
    _assert_eq(child.get("parent"), stage_id, "内部节点 parent = stage.sop")


def test_sop_slots_and_intent_degraded_survive():
    """结构化 SOP 槽位不能被 _scalars 丢弃；intent degraded 不能被覆盖。"""
    sid = "tracetest-structured-facts"
    trace_store.begin_turn(sid, "测试")
    trace_store.route("intent", degraded=True)
    trace_store.step("intent", intent="unknown", margin=0.0, degraded=True)
    trace_store.step("sop", id="model_query", n_slots=1,
                     slots={"budget": [None, 1000]},
                     result={"count": 1})
    trace_store.end_turn(reply="ok")
    row = read_last_turn(sid)
    _assert_eq(row["route"]["intent"].get("degraded"), True, "intent degraded 保留")
    _assert_true(row["sop"].get("slots"), "SOP slots 不丢")
    _assert_true(row["sop"].get("result"), "SOP result 不丢")
    route_intents = [n for n in (row.get("path") or [])
                     if n.get("node") == "route.intent"]
    _assert_eq(len(route_intents), 1, "degraded 不重复写 route.intent 节点")


def test_sop_step_spans_match_flow():
    """每个 SOP 循环迭代一条 sop.step span，outcome 与 sop.flow 1:1。"""
    import service.sops
    import service.sops.base
    import service.sops.repair_sop

    sid = "tracetest-sop-steps"
    steps = service.sops.repair_sop.REPAIR_SOP["steps"]
    old_action = steps[1]["action"]
    steps[1]["action"] = lambda slots: {"query": "", "domain": "repair"}
    try:
        trace_store.begin_turn(sid, "机器人不动了")
        stage_id = trace_store.begin_span("stage.sop", kind="stage")
        result = service.sops.base.start_sop(sid, "repair", "机器人不动了")
        trace_store.end_span(stage_id, status="hit")
        trace_store.note_behavior("sop_step")
        trace_store.end_turn(reply=result.reply)
    finally:
        steps[1]["action"] = old_action
        service.sops.base.end_sop(sid)

    row = read_last_turn(sid)
    flow = (row.get("sop") or {}).get("flow") or []
    steps = [n for n in (row.get("path") or []) if n.get("node") == "sop.step"]
    _assert_eq([s.get("out", {}).get("outcome") for s in steps],
               [f.get("outcome") for f in flow], "sop.step 与 flow 的 outcome 1:1")
    _assert_true(all(s.get("parent") == stage_id for s in steps),
                 "SOP step span 都挂在 stage.sop 下")
    _assert_eq(steps[0].get("in", {}).get("type"), "ask", "sop.step.in 记步骤类型")
    _assert_eq(steps[0].get("out", {}).get("outcome"), "filled", "sop.step.out 记结果")
    symptom = next(n for n in (row.get("path") or []) if n.get("node") == "sop.symptom")
    _assert_eq(symptom.get("parent"), steps[0].get("id"), "sop.symptom 挂在首个 sop.step 下")


def test_turn_index_increments_and_resets():
    """轮次号由共享序列推进；reset_trace 后从 1 重新开始。"""
    sid = "tracetest-seq"
    reset_trace(sid)
    seen = []
    try:
        for i in range(20):
            trace_store.begin_turn(sid, "第%d轮" % i)
            trace_store.note_behavior("chitchat")
            trace_store.end_turn(reply="ok")
            seen.append(write._last[sid].get("turn_index"))
        _assert_eq(seen, list(range(1, 21)), "同一会话轮次号连续 1..20")
    finally:
        reset_trace(sid)
    trace_store.begin_turn(sid, "重置后")
    try:
        _assert_eq(write._last.get(sid, {}).get("turn_index"), None,
                   "重置后还没落盘")
        trace_store.end_turn(reply="ok")
        _assert_eq(write._last[sid].get("turn_index"), 1, "重置后从 1 开始")
    finally:
        trace_store.end_turn()
        reset_trace(sid)


def test_trace_dir_isolation():
    root = trace_store._root_dir()
    _assert_true(os.path.abspath(root).startswith(os.path.abspath("temp/trace_test")),
                 "TRACE_DIR 测试隔离仍生效")
    dirs = trace_store.session_dirs()
    _assert(isinstance(dirs, list), "session_dirs 仍公开且返回列表")


TESTS = [test_facade_reexports_public_api,
         test_sop_span_contains_internal_node,
         test_sop_slots_and_intent_degraded_survive,
         test_sop_step_spans_match_flow,
         test_turn_index_increments_and_resets, test_trace_dir_isolation]


def run():
    reset()
    return run_tests("trace 拆分契约测试", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
