"""故障排查 SOP 测试：症状映射 + repair SOP 流程。

覆盖：
  - repair._extract_symptom 关键词规则分支（口语故障 → 标准检索 query）
  - repair SOP 触发（故障词 / 维修词）
  - repair SOP 流程：触发词含症状 → 直接出排查（stub 掉 LLM/Chroma 的 action）
  - repair SOP 提取失败 → 反问故障现象（stub 掉 extract）"""
import sys
from _runner import *
import sops  # 触发注册
import sops.base
import sops.repair


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
# 1. 症状关键词映射（规则分支，不触发 LLM）
# ──────────────────────────────────────────────────────────────

def test_extract_symptom_rule():
    from sops.repair import _extract_symptom
    _assert_eq(_extract_symptom("机器人不动了", {}), "机器人不移动怎么办", "不动")
    _assert_eq(_extract_symptom("水箱漏水", {}), "水箱漏水怎么办", "漏水")
    _assert_eq(_extract_symptom("有异响", {}), "扫地机器人异响怎么办", "异响")
    _assert_eq(_extract_symptom("充不进电", {}), "机器人充不进电怎么办", "充不进电")
    _assert_eq(_extract_symptom("找不到充电座", {}), "机器人找不到充电座怎么办", "找不到充电座")
    _assert_eq(_extract_symptom("吸力变小了", {}), "吸力下降怎么办", "吸力")
    _assert_eq(_extract_symptom("拖地后地面有水痕", {}), "拖地后地面有明显水痕", "水痕")


# ──────────────────────────────────────────────────────────────
# 2. repair SOP 触发
# ──────────────────────────────────────────────────────────────

def test_repair_trigger():
    _assert_eq(sops.base.match_sop("机器人不动了怎么办"), "repair", "故障词→repair")
    _assert_eq(sops.base.match_sop("水箱漏水了"), "repair", "漏水→repair")
    _assert_eq(sops.base.match_sop("帮我修一下扫地机器人"), "repair", "修→repair")
    _assert_eq(sops.base.match_sop("扫地机器人保修多久"), None, "售后→不触发 repair（guard）")


# ──────────────────────────────────────────────────────────────
# 3. repair SOP 流程（stub 掉 LLM/Chroma 的 action）
# ──────────────────────────────────────────────────────────────

def test_repair_sop_flow():
    _cleanup()
    orig = sops.repair.REPAIR_SOP["steps"][1]["action"]
    sops.repair.REPAIR_SOP["steps"][1]["action"] = lambda slots: {"answer": "第一步：检查电源是否接通。"}
    try:
        reply, done = sops.base.start_sop("r1", "repair", "机器人不动了怎么办")
        _assert(done, "触发词含症状→一次完成")
        _assert_in("第一步", reply, "返回排查步骤")
        _assert(not sops.base.has_active_sop("r1"), "会话已结束")
    finally:
        sops.repair.REPAIR_SOP["steps"][1]["action"] = orig
    _cleanup()


def test_repair_sop_ask_fallback():
    _cleanup()
    orig = sops.repair.REPAIR_SOP["steps"][0]["extract"]
    sops.repair.REPAIR_SOP["steps"][0]["extract"] = lambda text, slots: None
    try:
        reply, done = sops.base.start_sop("r2", "repair", "机器人不动了怎么办")
        _assert(not done, "提取失败未结束")
        _assert_in("故障现象", reply, "返回 ask 反问现象")
        _assert(sops.base.has_active_sop("r2"), "会话保持活跃")
    finally:
        sops.repair.REPAIR_SOP["steps"][0]["extract"] = orig
    _cleanup()


TESTS = [
    test_extract_symptom_rule,
    test_repair_trigger,
    test_repair_sop_flow,
    test_repair_sop_ask_fallback,
]


def run():
    reset()
    return run_tests("故障排查 SOP 测试（sops.repair）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
