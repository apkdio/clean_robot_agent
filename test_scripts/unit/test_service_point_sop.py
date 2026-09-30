"""网点 SOP 测试：入口判定 / guard / 唯一城市一句到位 / 重名选序号 / 乱答重问 / 引擎的 when 与话术占位。

geocode 与 LLM 兜底都打桩（不依赖 geonamescache 与 Ollama）；网点数据文件缺失时跳过全流程用例。
"""
import os
import re
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 与 test_scripts._runner 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
import sops  # 触发各 SOP 注册
import sops.base
import function_tools.service_point_tool as spt
from sops import service_point as sp

_CAND = [
    {"name": "Chaoyang", "cn_name": "朝阳（辽宁）", "lng": 120.45, "lat": 41.57, "population": 400000},
    {"name": "Chaoyang", "cn_name": "朝阳（北京）", "lng": 116.48, "lat": 39.92, "population": 350000},
]
_SHANGHAI = [{"name": "Shanghai", "cn_name": "上海市", "lng": 121.47, "lat": 31.23, "population": 20000000}]


def _cleanup():
    sops.base._sessions = {}
    try:
        from tools.redis_store import get_redis
        r = get_redis()
        if r is not None:
            for k in r.keys("sop:session:*"):
                r.delete(k)
    except Exception:
        pass


def _points_ready():
    try:
        return len(spt._load_points()) > 0
    except Exception:
        return False


def _stub_geo(mapping, llm=None):
    """geocode 与 LLM 兜底换打桩，返回还原函数。"""
    old_geo, old_llm = spt.geocode_city, sp._extract_city_by_llm
    spt.geocode_city = lambda name: list(mapping.get((name or "").strip(), []))
    sp._extract_city_by_llm = llm or (lambda text: [])

    def restore():
        spt.geocode_city, sp._extract_city_by_llm = old_geo, old_llm

    return restore


# ──────────────────────────────────────────────────────────────
# 入口判定与 guard
# ──────────────────────────────────────────────────────────────

def test_entry_judgement():
    hit = lambda q: sops.base.match_sop(q, "service_point") is not None
    _assert(hit("上海有售后网点吗"), "网点词命中")
    _assert(hit("附近的服务站"), "门店/服务站也算")
    _assert(not hit("网点怎么查询"), "政策咨询不算")
    _assert(not hit("机器人吸力多大"), "无网点词不算")


def test_match_and_guard():
    _assert_eq(sops.base.match_sop("上海有售后网点吗"), "service_point", "match_sop 能命中网点")
    _assert_eq(sops.base.match_sop("网点怎么查询"), None, "政策咨询被 guard 挡掉")


# ──────────────────────────────────────────────────────────────
# 流程：唯一城市 / 重名 / 乱答
# ──────────────────────────────────────────────────────────────

def test_unique_city_one_turn():
    if not _points_ready():
        _skip("无网点数据（data/service_point/service_points.json 缺失）")
        return
    _cleanup()
    restore = _stub_geo({"上海有售后网点吗": _SHANGHAI})
    try:
        sid = "sp_unique"
        reply, done = sops.base.start_sop(sid, "service_point", "上海有售后网点吗")
        _assert(done, "唯一城市一句问完")
        _assert_in("网点", reply, "给出网点列表")
        _assert(not sops.base.has_active_sop(sid), "已结束，不留状态")
    finally:
        restore()
        _cleanup()


def test_ambiguous_then_pick():
    if not _points_ready():
        _skip("无网点数据（data/service_point/service_points.json 缺失）")
        return
    _cleanup()
    restore = _stub_geo({"朝阳有维修点吗": _CAND})
    try:
        sid = "sp_ambiguous"
        reply, done = sops.base.start_sop(sid, "service_point", "朝阳有维修点吗")
        _assert(not done, "重名时先问不结束")
        _assert_in("查到多个同名地点", reply, "列出同名候选")
        _assert_in("1.", reply, "候选带序号")
        _assert(sops.base.has_active_sop(sid), "停在选序号那步")

        reply, done = sops.base.continue_sop(sid, "2")
        _assert(done, "选完即结束")
        _assert_in("朝阳（北京）", reply, "按所选候选作答")
        _assert(not sops.base.has_active_sop(sid), "已结束")
    finally:
        restore()
        _cleanup()


def test_garbage_then_retry_then_city():
    if not _points_ready():
        _skip("无网点数据（data/service_point/service_points.json 缺失）")
        return
    _cleanup()
    restore = _stub_geo({"杭州": [{"name": "Hangzhou", "cn_name": "杭州市", "lng": 120.15, "lat": 30.28, "population": 9000000}]})
    try:
        sid = "sp_retry"
        reply, done = sops.base.start_sop(sid, "service_point", "有没有门店")
        _assert(not done, "没识出城市 → 反问")
        _assert_in("请问您所在的城市", reply, "首次反问用 ask 话术")

        reply, done = sops.base.continue_sop(sid, "嗯嗯那个")
        _assert(not done, "还是没识出 → 再问")
        _assert_in("没太听清", reply, "第二次用 retry 话术")

        reply, done = sops.base.continue_sop(sid, "杭州")
        _assert(done and "网点" in reply, "答出城市后给结果")
    finally:
        restore()
        _cleanup()


# ──────────────────────────────────────────────────────────────
# 引擎新增能力：条件步骤（when）+ 话术占位
# ──────────────────────────────────────────────────────────────

def test_when_skips_step_and_formats_ask():
    _cleanup()
    sops.base.register({
        "id": "when_probe",
        "trigger": ["zzz-when-probe"],
        "steps": [
            {"id": "ask_n", "type": "ask", "slot": "n", "ask": "给个数字",
             "extract": lambda text, slots: int(re.search(r"\d", text).group()) if re.search(r"\d", text) else None},
            {"id": "ask_pick", "type": "ask", "slot": "pick", "when": lambda ctx: ctx.get("n", 0) > 1,
             "ask": "{n} 大于 1，请再给个序号",
             "extract": lambda text, slots: int(re.search(r"\d", text).group()) if re.search(r"\d", text) else None},
            {"id": "reply", "type": "reply", "template": "{n}"},
        ],
    })
    sid = "when_probe_big"
    reply, done = sops.base.start_sop(sid, "when_probe", "3")
    _assert(not done, "n=3 → 该步不跳过")
    _assert_in("3 大于 1", reply, "ask 话术按槽位填充")

    reply, done = sops.base.continue_sop(sid, "1")
    _assert(done and reply.strip() == "3", "reply 模板按槽位填充")

    sid2 = "when_probe_small"
    reply, done = sops.base.start_sop(sid2, "when_probe", "1")
    _assert(done and reply.strip() == "1", "n=1 → 该步被跳过")
    _cleanup()


def test_guard_only_vetoes_its_own_sop():
    """guard 命中只否掉自己那条，后面的 SOP 照样命中（2026-09-30 改）。"""
    _cleanup()
    sops.base.register({
        "id": "guard_probe_a", "trigger": ["zzz-guard"],
        "guards": [{"check": lambda q: True}],
        "steps": [{"id": "r", "type": "reply", "template": "A"}],
    })
    sops.base.register({
        "id": "guard_probe_b", "trigger": ["zzz-guard"],
        "steps": [{"id": "r", "type": "reply", "template": "B"}],
    })
    _assert_eq(sops.base.match_sop("zzz-guard 测试"), "guard_probe_b", "A 被 guard 否掉后轮到 B")
    reply, done = sops.base.start_sop("guard_probe", "guard_probe_a", "zzz-guard 测试")
    _assert_eq(reply.strip(), "A", "显式指定进哪条就进哪条（guards 只在匹配时评估）")
    _cleanup()


TESTS = [
    test_entry_judgement,
    test_match_and_guard,
    test_unique_city_one_turn,
    test_ambiguous_then_pick,
    test_garbage_then_retry_then_city,
    test_when_skips_step_and_formats_ask,
    test_guard_only_vetoes_its_own_sop,
]


def run():
    reset()
    return run_tests("网点 SOP 测试（sops.base + service_point）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
