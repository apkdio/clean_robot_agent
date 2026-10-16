"""选购 SOP 测试：SOP 状态机基础设施 + 选购推荐 SOP。

覆盖：
  - sops.base：register / match_sop（trigger + guards）/ start_sop /
    continue_sop / end_sop / has_active_sop / get_active_sop_id
  - 槽位提取：预算（extract_price_constraint，真实；＋纯数字短答按上限）、宠物（purchase._extract_has_pet，＋裸否定短答）
  - reply 步：函数模板（按结果分支：0 款换 PURCHASE_EMPTY_REPLIES、有结果保持原格式）
  - 重试循环 / 槽位放弃（max_retry）/ 模板 fallback / 异常防御
  - purchase SOP 触发与 guard（选购咨询/售后/品牌不触发选购）
  - [数据可用时] 选购 SOP 全流程（真实 enumerate_models）"""
import sys
import re
import os
import time
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 与 test_scripts._runner 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
import sops  # 触发 purchase/repair SOP 注册
import sops.base
from test_scripts import trace_reader
from tools import trace_store


def _models_ready():
    try:
        from tools.metadata_extractor import get_all_models
        return len(get_all_models()) > 0
    except Exception:
        return False


def _cleanup():
    sops.base._sessions = {}
    # 会话状态已迁到 Redis，同时清空 Redis 里的 SOP 会话，避免测试间残留
    try:
        from tools.redis_store import get_redis
        r = get_redis()
        if r is not None:
            for k in r.keys("sop:session:*"):
                r.delete(k)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# 测试用自包含 SOP（不依赖 Chroma/LLM）
# ──────────────────────────────────────────────────────────────

def _extract_budget(text, slots):
    from tools.metadata_extractor import extract_price_constraint
    return extract_price_constraint(text)


def _extract_has_pet(text, slots):
    if re.search(r"没|无|不养|没有|没养", text):
        return False
    if re.search(r"有|养|猫|狗|宠物", text):
        return True
    return None


def _mock_search(slots):
    budget = slots.get("budget")
    models = [
        {"name": "型号A", "price": 699},
        {"name": "型号B", "price": 999},
        {"name": "型号C", "price": 1299},
    ]
    if not budget:
        matched = models
    else:
        lo, hi = budget
        matched = [m for m in models if (lo is None or m["price"] >= lo) and (hi is None or m["price"] <= hi)]
    return {"count": len(matched), "list": "\n".join(m["name"] for m in matched), "models": matched}


_TEST_SOP = {
    "id": "sop_unit_test",
    "trigger": ["单测触发"],
    "max_retry": 2,
    "steps": [
        {"id": "ask_budget", "type": "ask", "slot": "budget",
         "ask": "预算多少？", "retry": "预算没听清，说个数字。", "extract": _extract_budget},
        {"id": "ask_pet", "type": "ask", "slot": "has_pet",
         "ask": "有宠物吗？", "retry": "有或没有？", "extract": _extract_has_pet},
        {"id": "do_search", "type": "action", "action": _mock_search},
        {"id": "reply", "type": "reply",
         "template": "共 {count} 款：{list}", "fallback": "抱歉，出了点小问题。"},
    ],
}

_BAD_SOP = {
    "id": "sop_bad_template",
    "trigger": ["坏模板"],
    "steps": [
        {"id": "ask_name", "type": "ask", "slot": "name",
         "ask": "叫什么？", "retry": "说名字。", "extract": lambda text, slots: text.strip() or None},
        {"id": "reply", "type": "reply",
         "template": "你好{name}{missing_field}", "fallback": "抱歉，出了点小问题。"},
    ],
}


def _boom_action(slots):
    raise RuntimeError("step boom")


_BOOM_SOP = {
    "id": "sop_boom_action",
    "trigger": ["炸一下"],
    "steps": [
        {"id": "boom", "type": "action", "action": _boom_action},
        {"id": "reply", "type": "reply", "template": "不会走到这里"},
    ],
}


# ──────────────────────────────────────────────────────────────
# 1. 注册与匹配
# ──────────────────────────────────────────────────────────────

def test_register_and_match():
    sops.base.register(_TEST_SOP)
    _assert("sop_unit_test" in sops.base.SOPS, "注册成功")
    _assert_eq(sops.base.match_sop("单测触发一下"), "sop_unit_test", "trigger 命中")
    _assert_eq(sops.base.match_sop("你好呀"), None, "闲聊不触发")
    _assert_eq(sops.base.match_sop("边刷多久换一次"), None, "维护不触发")


def test_purchase_trigger_and_guards():
    # 真实 purchase SOP
    _assert_eq(sops.base.match_sop("帮我推荐一款扫地机器人"), "purchase", "推荐→选购 SOP")
    _assert_eq(sops.base.match_sop("预算1000以内买一个"), "purchase", "预算/买→选购 SOP")
    # guard：选购咨询不触发选购 SOP（走 RAG）
    _assert(sops.base.match_sop("选购扫地机器人要注意什么") is None, "选购咨询 guard 拦截")
    _assert(sops.base.match_sop("扫地机器人保修多久") is None, "售后 guard 拦截")
    _assert(sops.base.match_sop("不染一尘有什么优势") is None, "品牌 guard 拦截")


# ──────────────────────────────────────────────────────────────
# 2. 会话生命周期
# ──────────────────────────────────────────────────────────────

def test_start_and_continue():
    _cleanup()
    reply, done = sops.base.start_sop("s1", "sop_unit_test", "帮我看看")
    _assert_eq(reply, "预算多少？", "首轮返回 ask")
    _assert(not done, "未结束")
    _assert(sops.base.has_active_sop("s1"), "有活跃会话")
    reply, done = sops.base.continue_sop("s1", "1000以内")
    _assert_eq(reply, "有宠物吗？", "推进到 ask_pet")
    _cleanup()


def test_continue_no_session():
    _cleanup()
    _assert(sops.base.continue_sop("s1", "随便") is None, "无会话→None")


def test_end_sop():
    _cleanup()
    sops.base.start_sop("s1", "sop_unit_test", "x")
    _assert(sops.base.has_active_sop("s1"), "启动后有会话")
    sops.base.end_sop("s1")
    _assert(not sops.base.has_active_sop("s1"), "结束后无会话")
    _assert_eq(sops.base.get_active_sop_id("s1"), None, "结束后 id=None")


def test_full_flow():
    _cleanup()
    sops.base.start_sop("s1", "sop_unit_test", "x")
    sops.base.continue_sop("s1", "1000以内")
    reply, done = sops.base.continue_sop("s1", "没有")
    _assert(done, "SOP 结束")
    _assert_in("共 2 款", reply, "1000以内命中 2 款")
    _assert_in("型号A", reply, "含型号A")
    _assert_in("型号B", reply, "含型号B")
    _assert_not_in("型号C", reply, "不含超预算型号C")
    _assert(not sops.base.has_active_sop("s1"), "会话已结束")
    _cleanup()


# ──────────────────────────────────────────────────────────────
# 3. 提取 / 重试 / 放弃 / fallback
# ──────────────────────────────────────────────────────────────

def test_extract_has_pet():
    from sops.purchase import _extract_has_pet
    _assert(_extract_has_pet("有猫", {}) is True, "有猫→True")
    _assert(_extract_has_pet("养狗", {}) is True, "养狗→True")
    _assert(_extract_has_pet("没有", {}) is False, "没有→False")
    _assert(_extract_has_pet("没养", {}) is False, "没养→False")
    _assert(_extract_has_pet("无", {}) is False, "无→False")
    _assert(_extract_has_pet("不", {}) is False, "裸「不」→False（槽位短答，别再问一遍）")
    _assert(_extract_has_pet("不养", {}) is False, "不养→False")
    _assert(_extract_has_pet("不知道", {}) is None, "不知道→None")


def test_extract_budget_bare_number():
    from sops.purchase import _extract_budget
    _assert_eq(_extract_budget("800块以内", {}), (None, 800), "带量词照旧走通用解析")
    _assert(_extract_budget("2000左右", {}) is not None, "区间照旧走通用解析")
    _assert_eq(_extract_budget("800", {}), (None, 800), "纯数字短答按上限收（BC-20261015-03）")
    _assert_eq(_extract_budget("预算 5000 元", {}), (None, 5000), "带「预算」照旧")
    _assert_eq(_extract_budget("不知道", {}), None, "非预算表达→None")


def test_result_reply():
    from sops.purchase import _render_result
    full = _render_result({"count": 2, "list": "- A\n- B"})
    _assert("共找到 2 款" in full and "对比其中某两款" in full, "有结果时保持原格式")
    empty = _render_result({"count": 0, "list": "", "models": []})
    _assert("预算" in empty, "0 款时说明是预算内没有")
    _assert(not re.search(r"\d", empty), "0 款时不给价格数字（没有依据的推荐）")
    _assert("对比" not in empty and "型号" not in empty, "0 款时不追问对比、不提型号")


def test_retry_then_abandon():
    _cleanup()
    # start_sop 的触发词会先尝试一次提取（失败 → retry=1 → ask 话术）
    r, _ = sops.base.start_sop("s1", "sop_unit_test", "x")
    _assert_eq(r, "预算多少？", "触发词提取失败→ask")
    # 第 2 次失败（retry=2）→ retry 话术
    r, _ = sops.base.continue_sop("s1", "不知道")
    _assert_eq(r, "预算没听清，说个数字。", "重试失败→retry")
    # 第 3 次失败（retry=3 > max_retry=2）→ 放弃预算槽位推进到下一问
    r, _ = sops.base.continue_sop("s1", "还是不知道")
    _assert_eq(r, "有宠物吗？", "超重试→放弃预算槽位")
    _cleanup()


def test_template_fallback():
    _cleanup()
    sops.base.register(_BAD_SOP)
    sops.base.start_sop("s1", "sop_bad_template", "")
    reply, done = sops.base.continue_sop("s1", "张三")
    _assert(done, "SOP 结束")
    _assert_eq(reply, "抱歉，出了点小问题。", "缺字段触发 fallback")
    _cleanup()


def test_unknown_sop_id_dropped():
    """会话里记的 sop_id 已不存在（SOP 被改名/删掉）→ 状态作废、按「没有活跃 SOP」处理，不再抛 KeyError。"""
    _cleanup()
    sops.base.register(_TEST_SOP)
    sops.base._start("s1", "nonexistent_sop")
    _assert(not sops.base.has_active_sop("s1"), "旧状态被作废")
    _assert_eq(sops.base.continue_sop("s1", None), None, "continue_sop 返回 None，本轮回落普通作答")
    _cleanup()


def test_stale_def_state_dropped():
    """代码改过 SOP 定义（步骤/槽位变了）→ Redis 里的旧状态作废，不拿旧下标续跑。"""
    _cleanup()
    sops.base.register(_TEST_SOP)
    sops.base.start_sop("s1", "sop_unit_test", "")
    _assert(sops.base.has_active_sop("s1"), "先有活跃状态")
    # 模拟一次发版：同一 sop_id 的定义多出一步 → 指纹变
    changed = dict(_TEST_SOP, steps=list(_TEST_SOP["steps"]) +
                   [{"id": "extra", "type": "reply", "template": "补充说明"}])
    sops.base.register(changed)
    _assert(not sops.base.has_active_sop("s1"), "定义变了 → 旧状态作废")
    _assert(sops.base._load_session("s1") is None, "Redis 里那份也已被删掉")
    _assert_eq(sops.base.continue_sop("s1", "1000以内"), None, "不再拿旧下标续跑")
    sops.base.register(_TEST_SOP)          # 还原注册表，别影响后面的用例
    _cleanup()


def test_injected_state_keeps_fingerprint():
    """测试注入的状态（tools.test_hooks.set_active_sop）带指纹，能正常续跑——行为轨的 stateful 用例靠它。"""
    _cleanup()
    sops.base.register(_TEST_SOP)
    from tools.test_hooks import set_active_sop
    set_active_sop("s1", "sop_unit_test", step=1, slots={"budget": [None, 1000]})
    _assert(sops.base.has_active_sop("s1"), "注入的状态没被当成过期")
    reply, done = sops.base.continue_sop("s1", "没有")
    _assert(done, "答完剩下槽位后一轮走完 action + reply")
    _assert_in("型号A", reply, "回复里带推荐结果")
    _cleanup()


def test_stale_def_state_recorded_in_trace():
    """作废要留在 trace 里（AGENTS：降级必须 WARNING + trace）：sop.active / skipped / by=stale_def。"""
    _cleanup()
    sops.base.register(_TEST_SOP)
    sops.base.start_sop("s1", "sop_unit_test", "")
    reset_trace("s1")
    trace_store.begin_turn("s1", "1000以内")
    # 发版：定义多一步 → 读状态时作废（这一步落节点）
    sops.base.register(dict(_TEST_SOP, steps=list(_TEST_SOP["steps"]) +
                            [{"id": "extra", "type": "reply", "template": "补充说明"}]))
    _assert(not sops.base.has_active_sop("s1"), "状态被作废")
    trace_store.end_turn(reply="（单测）")
    rows = trace_reader.load(trace_reader.session_files(trace_reader.root_dir(), "s1")[0])
    path = (rows[0].get("path") or []) if rows else []
    _assert(any(n.get("node") == "sop.active" and n.get("status") == "skipped"
                and n.get("by") == "stale_def" for n in path),
            "trace 里有 sop.active / skipped / by=stale_def")
    sops.base.register(_TEST_SOP)
    _cleanup()


def test_local_session_expires_without_redis():
    """无 Redis 的降级路径按 sop_ttl 滑动过期（Redis 路径由 Redis 自己过期，不走这段）。"""
    _cleanup()
    from tools import redis_store
    sops.base.register(_TEST_SOP)
    real = redis_store.get_redis
    redis_store.get_redis = lambda: None          # 强制走降级分支
    try:
        sops.base.start_sop("s1", "sop_unit_test", "")
        _assert(sops.base.has_active_sop("s1"), "刚写的降级态可用")
        _assert("ts" in sops.base._sessions["s1"], "降级态带 ts（Redis 路径不带）")
        sops.base._sessions["s1"]["ts"] = time.time() - 100000      # 手动拨旧（sop_ttl=1800）
        _assert(not sops.base.has_active_sop("s1"), "超过 sop_ttl → 作废")
        _assert_eq(sops.base._load_session("s1"), None, "降级态那份也删了")
    finally:
        redis_store.get_redis = real
        sops.base._sessions.pop("s1", None)
    _cleanup()


def test_step_exception_ends_sop():
    """步骤函数抛异常：向上抛（外层兜底）+ 结束会话（不留半截状态、下轮不重跑同一步）。"""
    _cleanup()
    sops.base.register(_BOOM_SOP)
    _assert_raises(RuntimeError, lambda: sops.base.start_sop("s1", "sop_boom_action", "随便"),
                   "步骤异常照常向上抛")
    _assert(not sops.base.has_active_sop("s1"), "异常后会话已结束（不会被 finally 写回去）")
    _assert_eq(sops.base.continue_sop("s1", "再说一遍"), None, "下一轮不再停在同一步重复失败")
    _cleanup()


# ──────────────────────────────────────────────────────────────
# 4. 选购 SOP 全流程（真实 enumerate_models，数据可用时）
# ──────────────────────────────────────────────────────────────

def test_purchase_sop_full_flow():
    if not _models_ready():
        _skip("无型号数据（Chroma/pkl 缓存缺失）")
        return
    _cleanup()
    sid = "purchase_flow_test"
    reply, done = sops.base.start_sop(sid, "purchase", "帮我推荐一款")
    _assert(sops.base.has_active_sop(sid), "purchase SOP 启动")
    _assert_in("预算", reply, "开场问预算")

    reply, done = sops.base.continue_sop(sid, "1000以内")
    _assert(not done, "预算后未结束")
    _assert_in("宠物", reply, "下一步问宠物")

    reply, done = sops.base.continue_sop(sid, "没有")
    _assert(done, "SOP 结束")
    _assert_in("共找到", reply, "推荐含统计")
    _assert_in("净白 S1", reply, "1000以内含净白 S1(899)")
    _assert_not_in("净界 P1", reply, "1000以内不含净界 P1(1799)")
    _assert(not sops.base.has_active_sop(sid), "会话已结束")
    _cleanup()


TESTS = [
    test_register_and_match,
    test_purchase_trigger_and_guards,
    test_start_and_continue,
    test_continue_no_session,
    test_end_sop,
    test_full_flow,
    test_extract_has_pet,
    test_extract_budget_bare_number,
    test_result_reply,
    test_retry_then_abandon,
    test_template_fallback,
    test_unknown_sop_id_dropped,
    test_stale_def_state_dropped,
    test_injected_state_keeps_fingerprint,
    test_stale_def_state_recorded_in_trace,
    test_step_exception_ends_sop,
    test_local_session_expires_without_redis,
    test_purchase_sop_full_flow,
]


def run():
    reset()
    return run_tests("选购 SOP 测试（sops.base + purchase）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
