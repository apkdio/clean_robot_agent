"""故障排查 SOP 测试：症状映射 + repair SOP 流程。

覆盖：
  - repair._extract_symptom 关键词规则分支（口语故障 → 标准检索 query）
  - repair SOP 触发（故障词 / 维修词）
  - repair SOP 流程：触发词含症状 → 直接出排查（stub 掉 LLM/Chroma 的 action）
  - repair SOP 提取失败 → 反问故障现象（stub 掉 extract）"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 与 test_scripts._runner 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
import sops  # 触发注册
import sops.base
import sops.repair_sop
from test_scripts import trace_reader
from tools import trace_store


def _pair(result):
    return result.reply, result.done


def _cleanup():
    sops.base._sessions = {}
    # 会话状态已迁到 Redis，同时清空 Redis 里的 SOP 会话，避免测试间残留
    try:
        from gateway.redis_store import get_redis
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
    from sops.repair_sop import _extract_symptom
    _assert_eq(_extract_symptom("机器人不动了", {}).value, "机器人不移动怎么办", "不动")
    _assert_eq(_extract_symptom("水箱漏水", {}).value, "水箱漏水怎么办", "漏水")
    _assert_eq(_extract_symptom("有异响", {}).value, "扫地机器人异响怎么办", "异响")
    _assert_eq(_extract_symptom("充不进电", {}).value, "机器人充不进电怎么办", "充不进电")
    _assert_eq(_extract_symptom("找不到充电座", {}).value, "机器人找不到充电座怎么办", "找不到充电座")
    _assert_eq(_extract_symptom("吸力变小了", {}).value, "吸力下降怎么办", "吸力")
    _assert_eq(_extract_symptom("拖地后地面有水痕", {}).value, "拖地后地面有明显水痕", "水痕")


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
    steps = sops.repair_sop.REPAIR_SOP["steps"]
    orig_action = steps[1]["action"]
    steps[1]["action"] = lambda slots: {"query": "机器人不移动怎么办", "domain": "repair"}
    try:
        result = sops.base.start_sop("r1", "repair", "机器人不动了怎么办")
        _assert(result.done, "触发词含症状→一次完成")
        _assert_eq(result.handoff.target, "retrieve", "交给主链 retrieve")
        _assert_eq(result.handoff.domain, "repair", "带 repair 域")
        _assert_eq(result.handoff.query, "机器人不移动怎么办", "带归一化 query")
        _assert(not sops.base.has_active_sop("r1"), "会话已结束")
    finally:
        steps[1]["action"] = orig_action
    _cleanup()


def test_repair_sop_ask_fallback():
    _cleanup()
    orig = sops.repair_sop.REPAIR_SOP["steps"][0]["extract"]
    from tools.dto.sop import ExtractResult
    sops.repair_sop.REPAIR_SOP["steps"][0]["extract"] = lambda text, slots: ExtractResult("fail")
    try:
        reply, done = _pair(sops.base.start_sop("r2", "repair", "机器人不动了怎么办"))
        _assert(not done, "提取失败未结束")
        _assert_in("故障现象", reply, "返回 ask 反问现象")
        _assert(sops.base.has_active_sop("r2"), "会话保持活跃")
    finally:
        sops.repair_sop.REPAIR_SOP["steps"][0]["extract"] = orig
    _cleanup()


def test_repair_dropped_symptom_does_not_search():
    """症状槽位被放弃后 slots 里没有该键，不能把 None 字符串拿去检索。"""
    out = sops.repair_sop._search_repair({"_slot_status": {"symptom": "dropped"}})
    _assert_eq(out.get("query"), "", "dropped 槽位直接返回空 query")


def test_repair_handoff_contract():
    """症状确认后只产出 handoff，不在 SOP 内检索/生成。"""
    result = sops.repair_sop._search_repair({"symptom": "机器人不移动"})
    _assert_eq(result.get("query"), "机器人不移动", "action 只归一 query")
    _assert_eq(result.get("domain"), "repair", "action 标记 repair 域")
    handoff = sops.repair_sop._repair_handoff(result)
    _assert_eq(handoff.target, "retrieve", "目标阶段 retrieve")
    _assert_eq(handoff.query, "机器人不移动", "query 透传")
    _assert_eq(handoff.domain, "repair", "domain 透传")
    _assert_eq(sops.repair_sop._repair_handoff({"query": ""}), None, "无 query 不 handoff")


def test_repair_empty_reply_falls_back():
    """没有 query 时 handoff 返回 None，reply 用步骤级 fallback。"""
    _cleanup()
    steps = sops.repair_sop.REPAIR_SOP["steps"]
    orig_action = steps[1]["action"]
    fallback = steps[2]["fallback"]
    steps[1]["action"] = lambda slots: {"query": "", "domain": "repair"}
    try:
        reply, done = _pair(sops.base.start_sop("r-empty", "repair", "机器人不动了怎么办"))
        _assert(done, "空回复仍应结束")
        _assert_in(fallback, reply, "空回复走步骤级兜底")
        _assert(not sops.base.has_active_sop("r-empty"), "会话已结束")
    finally:
        steps[1]["action"] = orig_action
    _cleanup()


def test_symptom_maps_in_sync():
    """两张症状表 + 工具 description 必须同步（v2.4 加「水痕/水印」时漏过一次）。"""
    from config.word_dict_config import SYMPTOM_MAP, SYMPTOM_QUERY_MAP
    from function_tools.schemas import EXTRACT_SYMPTOM_SCHEMA
    a, b = set(SYMPTOM_MAP.values()), set(SYMPTOM_QUERY_MAP.values())
    _assert_eq(sorted(a - b), [], "这些标准问句只在口语词表里 → LLM 兜底永远选不到")
    _assert_eq(sorted(b - a), [], "这些标准问句只在工具闭集里 → 规则侧没有对应口语词")
    ids = sorted(SYMPTOM_QUERY_MAP)
    _assert_eq(ids, list(range(1, len(ids) + 1)), "闭集 id 从 1 连续编号")
    desc = EXTRACT_SYMPTOM_SCHEMA["function"]["description"]
    missing = [i for i in ids if ("%d=" % i) not in desc]
    _assert_eq(missing, [], "这些编号没写进工具 description → 模型看不见这个选项")


def _path_of(sid, query, fn):
    """跑一轮 trace 并取回该轮的 path（写入侧那条路径，别按会话名捞——会捞到旧日期的目录）。"""
    reset_trace(sid)
    trace_store.begin_turn(sid, query)
    fn()
    trace_store.end_turn()
    rows = trace_reader.load(trace_store._path(sid))
    return (rows[0].get("path") or []) if rows else []


def test_symptom_mapping_leaves_trace():
    """症状映射来源要落 trace：extractor 只返回 extra["trace"]，由 base 统一写。"""
    import gateway.llm_tool as lt
    real = lt.chat_with_tools
    orig_action = sops.repair_sop.REPAIR_SOP["steps"][1]["action"]
    sops.repair_sop.REPAIR_SOP["steps"][1]["action"] = lambda slots: {"query": "", "domain": "repair"}
    try:
        lt.chat_with_tools = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no llm"))
        path = _path_of("sym1", "它不动了",
                        lambda: sops.base.start_sop("sym1", "repair", "它不动了"))
        node = next((n for n in path if n.get("node") == "sop.symptom"), None)
        _assert(node is not None, "规则命中留 sop.symptom 节点")
        _assert_eq(((node or {}).get("out") or {}).get("via"), "rule", "via=rule")
        _assert_eq(((node or {}).get("out") or {}).get("kw"), "不动", "带命中的口语词")

        path = _path_of("sym2", "说不上来",
                        lambda: sops.base.start_sop("sym2", "repair", "嗯…说不上来"))
        node = next((n for n in path if n.get("node") == "sop.symptom"), None)
        _assert_eq((node or {}).get("status"), "miss", "两条都没命中 → status=miss")
    finally:
        lt.chat_with_tools = real
        sops.repair_sop.REPAIR_SOP["steps"][1]["action"] = orig_action
        sops.base.end_sop("sym1")
        sops.base.end_sop("sym2")
        sops.base._sessions = {}


TESTS = [
    test_extract_symptom_rule,
    test_repair_trigger,
    test_repair_sop_flow,
    test_repair_sop_ask_fallback,
    test_repair_dropped_symptom_does_not_search,
    test_repair_handoff_contract,
    test_repair_empty_reply_falls_back,
    test_symptom_maps_in_sync,
    test_symptom_mapping_leaves_trace,
]


def run():
    reset()
    return run_tests("故障排查 SOP 测试（sops.repair_sop）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
