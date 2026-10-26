"""澄清状态存储测试：业务 id 隔离、重试计数、过期与关闭语义。"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 与 test_scripts._runner 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from tools import pending_store as ps


def test_ask_peek_shape():
    """写入形状：biz + state + data + retry=0；业务状态与展示载荷分开放。"""
    ps.close("ps-shape")
    ps.ask("ps-shape", ps.BIZ_OPS_CLARIFY, state={"query": "原问题"},
           options=["sop", "date"])
    rec = ps.peek("ps-shape")
    _assert_eq(rec.get("biz"), ps.BIZ_OPS_CLARIFY, "biz 落盘")
    _assert_eq(rec.get("retry"), 0, "retry 从 0 起")
    _assert_eq((rec.get("state") or {}).get("query"), "原问题", "业务状态在 state 里")
    _assert_eq((rec.get("data") or {}).get("options"), ["sop", "date"], "展示载荷在 data 里")
    ps.close("ps-shape")
    _assert(ps.peek("ps-shape") is None, "close 后为空")


def test_ask_rebuild_clears_old_flag():
    """同一业务再次开澄清：先清旧标记，retry 归零、状态换新（不沿用上次的 retry）。"""
    ps.close("ps-rebuild")
    ps.ask("ps-rebuild", ps.BIZ_OPS_CLARIFY, state={"query": "旧"})
    ps.bump("ps-rebuild", ps.BIZ_OPS_CLARIFY)
    ps.ask("ps-rebuild", ps.BIZ_OPS_CLARIFY, state={"query": "新"})
    rec = ps.peek("ps-rebuild")
    _assert_eq(rec.get("retry"), 0, "旧标记被清掉，retry 不残留")
    _assert_eq((rec.get("state") or {}).get("query"), "新", "状态换新")
    ps.close("ps-rebuild")


def test_biz_isolation():
    """业务 id 隔离：别的业务既读不到、也清不掉。"""
    ps.close("ps-iso")
    ps.ask("ps-iso", ps.BIZ_SOP_ENTER, sop="model_query", query="预算 3000")
    _assert(ps.take("ps-iso", ps.BIZ_OPS_CLARIFY) is None, "别的业务认领不到")
    _assert(ps.take("ps-iso", ps.BIZ_SOP_ENTER) is not None, "自己的业务能认领")
    _assert_eq(ps.close("ps-iso", ps.BIZ_OPS_CLARIFY), False, "别的业务关不掉")
    _assert(ps.has("ps-iso", ps.BIZ_SOP_ENTER), "状态仍在")
    _assert_eq(ps.close("ps-iso", ps.BIZ_SOP_ENTER), True, "自己的业务能关")
    _assert(ps.peek("ps-iso") is None, "关掉即空")


def test_bump_only_owner():
    """retry 只由状态所属业务推进。"""
    ps.close("ps-bump")
    ps.ask("ps-bump", ps.BIZ_SOP_EXIT, sop_id="model_query")
    _assert(ps.bump("ps-bump", ps.BIZ_OPS_CLARIFY) is None, "别人 bump 不动")
    _assert_eq(ps.peek("ps-bump").get("retry"), 0, "retry 没变")
    _assert_eq(ps.bump("ps-bump", ps.BIZ_SOP_EXIT).get("retry"), 1, "自己 bump +1")
    ps.close("ps-bump")


def test_ask_overwrites_other_biz():
    """新澄清覆盖旧澄清（状态单槽），biz 与 state 同步换掉。"""
    ps.close("ps-over")
    ps.ask("ps-over", ps.BIZ_SOP_ENTER, state={"sop": "model_query", "entry": "旧问题"})
    ps.ask("ps-over", ps.BIZ_OPS_CLARIFY, state={"query": "新问题"}, options=["sop", "date"])
    rec = ps.peek("ps-over")
    _assert_eq(rec.get("biz"), ps.BIZ_OPS_CLARIFY, "biz 换成新的")
    _assert_eq((rec.get("state") or {}).get("query"), "新问题", "state 换成新的")
    _assert_eq(rec.get("retry"), 0, "retry 重置")
    ps.close("ps-over")


def test_memory_expiry():
    """无 Redis 时按时间戳判过期（不静默返回旧状态）。"""
    saved = (ps.get_redis, ps.clarify_ttl)
    ps.get_redis = lambda: None
    ps.clarify_ttl = lambda: 0
    try:
        ps.close("ps-exp")
        ps.ask("ps-exp", ps.BIZ_SOP_EXIT, sop_id="model_query")
        _assert(ps.peek("ps-exp") is None, "TTL 0 → 立即过期")
    finally:
        ps.get_redis, ps.clarify_ttl = saved
        ps.close("ps-exp")


def test_empty_biz_rejected():
    """业务 id 是必填（空值会让状态无法被任何业务认领）。"""
    try:
        ps.ask("ps-empty", "", query="x")
        _assert(False, "空 biz 应抛 ValueError")
    except ValueError:
        _assert(True, "空 biz 被拒绝")


TESTS = [test_ask_peek_shape, test_ask_rebuild_clears_old_flag, test_biz_isolation,
         test_bump_only_owner, test_ask_overwrites_other_biz, test_memory_expiry,
         test_empty_biz_rejected]


def run():
    reset()
    return run_tests("澄清状态存储测试（业务隔离 / 重试 / 过期）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
