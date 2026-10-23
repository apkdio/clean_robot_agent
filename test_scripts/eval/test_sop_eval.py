"""第四轨：SOP 行为轨（确定性状态机）。

只验证「进入 SOP 之后」的槽位与流程，不判路由、不判检索质量、不调 LLM。
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
import sops
import sops.base
from test_scripts import trace_reader
from tools import pending_store, trace_store
from tools.context_store import delete_session


def _reset(sid):
    sops.end_sop(sid)
    pending_store.close(sid)
    delete_session(sid)


def _models_ready():
    try:
        from tools.metadata_extractor import get_all_models
        return len(get_all_models()) > 0
    except Exception:
        return False


def _turn(sid, fn, query):
    """跑一轮并返回 (reply, flow)；flow 取自 trace 的 steps.sop.flow。"""
    trace_store.begin_turn(sid, query)
    try:
        reply = fn()
    finally:
        trace_store.end_turn()
    row = trace_reader.load(trace_store._path(sid))[-1]
    flow = ((row.get("steps") or {}).get("sop") or {}).get("flow") or []
    return reply, flow


def _start(sid, sop_id, query, slots=None):
    reply, _done = sops.start_sop(sid, sop_id, query, slots=slots)
    return reply


def _cont(sid, query):
    got = sops.continue_sop(sid, query)
    return got[0] if got else ""


def test_purchase_slots_and_finish():
    """选购 SOP：预算重问 → 预算 → 宠物 → 完成，槽位与 flow 可断言。"""
    sid = "sopeval-purchase"
    _reset(sid)
    try:
        reply, flow = _turn(sid, lambda: _start(sid, "purchase", "帮我推荐一款扫地机器人"),
                            "帮我推荐一款扫地机器人")
        _assert_in("预算", reply, "首问预算")
        _assert_eq(sops.get_active_sop_id(sid), "purchase", "SOP 活跃")
        _assert_eq((sops.get_sop_state(sid).get("step")), 0, "停在预算步")
        _assert_eq(sops.get_sop_state(sid).get("retry_count"), 1, "入口 query 未提取到预算")
        _assert_eq(flow[-1].get("outcome"), "asked", "首轮 flow=asked")

        reply, flow = _turn(sid, lambda: _cont(sid, "不知道"), "不知道")
        _assert_in("预算", reply, "没听清预算 → 重问")
        _assert_eq(sops.get_sop_state(sid).get("retry_count"), 2, "retry_count=2")

        if not _models_ready():
            _skip("模型数据不可用，跳过选购 action 轮")
            return
        reply, flow = _turn(sid, lambda: _cont(sid, "3000以内"), "3000以内")
        _assert_in("宠物", reply, "预算填完后问宠物")
        _assert_eq(list((sops.get_sop_state(sid).get("slots") or {}).get("budget") or []),
                   [None, 3000], "预算槽位写入")
        _assert("filled" in [f.get("outcome") for f in flow], "flow 记录 filled")

        reply, flow = _turn(sid, lambda: _cont(sid, "没有"), "没有")
        _assert(reply, "宠物填完后有回复")
        _assert_eq(sops.has_active_sop(sid), False, "SOP 已完成")
        _assert("done" in [f.get("outcome") for f in flow], "flow 记录 done")
    finally:
        _reset(sid)


def test_repair_flow():
    """故障排查 SOP：明确症状可直接进入并产出回复。"""
    sid = "sopeval-repair"
    _reset(sid)
    try:
        if not _models_ready():
            _skip("模型/检索数据不可用，跳过 repair")
            return
        reply, flow = _turn(sid, lambda: _start(sid, "repair", "机器人不动了"),
                            "机器人不动了")
        _assert(reply, "故障 SOP 有回复")
        _assert(any(f.get("type") in ("ask", "action", "reply") for f in flow),
                "flow 有动作链")
        _assert_eq(sops.has_active_sop(sid), False, "明确症状流程完成")
    finally:
        _reset(sid)


def test_model_query_tags_and_skip():
    """型号查询 SOP：空参可跳过，标签序号可写槽并完成。"""
    sid = "sopeval-model-query"
    _reset(sid)
    try:
        reply, flow = _turn(sid, lambda: _start(sid, "model_query", ""), "有哪些型号")
        _assert_in("预算", reply, "先问预算")
        reply, flow = _turn(sid, lambda: _cont(sid, "跳过"), "跳过")
        _assert_in("系列", reply, "预算跳过 → 问系列")
        reply, flow = _turn(sid, lambda: _cont(sid, "跳过"), "跳过")
        _assert_in("发布时间", reply, "系列跳过 → 问时间")
        reply, flow = _turn(sid, lambda: _cont(sid, "跳过"), "跳过")
        _assert_in("特点", reply, "时间跳过 → 问标签")
        _assert("1 " in reply and "养宠友好" in reply, "标签按序号展示")
        reply, flow = _turn(sid, lambda: _cont(sid, "1,3"), "1,3")
        _assert(reply, "标签选择后有回复")
        _assert_eq(sops.has_active_sop(sid), False, "型号查询 SOP 完成")
        _assert("skipped" in [f.get("outcome") for f in flow] or
                "filled" in [f.get("outcome") for f in flow], "flow 记录跳过/填充")
    finally:
        _reset(sid)


def test_model_query_prefill():
    """预填槽：不再返回 ask，直接执行 action/reply。"""
    sid = "sopeval-model-prefill"
    _reset(sid)
    try:
        if not _models_ready():
            _skip("模型数据不可用，跳过 prefill")
            return
        slots = {"budget": (None, 3000), "series": "净界 P",
                 "publish_range": ("2025-01-01", "2026-12-31"),
                 "tags": ["auto_dust"]}
        reply, flow = _turn(sid, lambda: _start(sid, "model_query", "", slots=slots),
                            "已有约束")
        _assert_in("符合您条件", reply, "全预填直接出结果")
        _assert_eq(sops.has_active_sop(sid), False, "预填后直接完成")
        _assert("skipped" in [f.get("outcome") for f in flow], "flow 记录预填跳过")
    finally:
        _reset(sid)


TESTS = [
    test_purchase_slots_and_finish,
    test_repair_flow,
    test_model_query_tags_and_skip,
    test_model_query_prefill,
]


def run():
    return run_tests("SOP 行为轨（确定性状态机 / 槽位 / flow）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
