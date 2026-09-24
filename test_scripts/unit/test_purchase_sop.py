"""选购 SOP 测试：SOP 状态机基础设施 + 选购推荐 SOP。

覆盖：
  - sops.base：register / match_sop（trigger + guards）/ start_sop /
    continue_sop / end_sop / has_active_sop / get_active_sop_id
  - 槽位提取：预算（extract_price_constraint，真实）、宠物（purchase._extract_has_pet）
  - 重试循环 / 槽位放弃（max_retry）/ 模板 fallback / 异常防御
  - purchase SOP 触发与 guard（选购咨询/售后/品牌不触发选购）
  - [数据可用时] 选购 SOP 全流程（真实 enumerate_models）"""
import sys
import re
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_SCRIPT_DIR) not in sys.path:   # test_scripts/：供 `_runner` 导入
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

from _runner import *
import sops  # 触发 purchase/repair SOP 注册
import sops.base


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
    _assert(_extract_has_pet("不知道", {}) is None, "不知道→None")


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


def test_unknown_sop_id_raises():
    _cleanup()
    sops.base.register(_TEST_SOP)
    sops.base._start("s1", "nonexistent_sop")

    def _run():
        sops.base._run("s1", None)

    _assert_raises(KeyError, _run, "不存在的 sop_id 抛 KeyError")
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
    test_retry_then_abandon,
    test_template_fallback,
    test_unknown_sop_id_raises,
    test_purchase_sop_full_flow,
]


def run():
    reset()
    return run_tests("选购 SOP 测试（sops.base + purchase）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
