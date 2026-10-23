"""模型查询收敛测试：预算/日期约束 → find_models → 直出格式。"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from tools.model_query import (ModelQueryResult, format_model_query, resolve_model_constraints,
                               resolve_model_query)


def _publish_int(model):
    return int(str(model.get("publish_date") or "").replace("-", "")[:8] or 0)


def test_budget_only():
    c, kind = resolve_model_constraints("预算2000以内的扫地机器人")
    _assert_eq(kind, "budget", "只命中预算")
    _assert_eq((c["budget_min"], c["budget_max"]), (0, 2000), "预算上下限")
    r = resolve_model_query("预算2000以内的扫地机器人")
    _assert(r is not None and r.models, "预算能筛出型号")
    _assert(all(int(m.get("price") or 0) <= 2000 for m in r.models), "结果都在预算内")
    _assert_in("在您预算内的机器人有", format_model_query(r), "预算前缀")


def test_date_only():
    q = "2025年发布的扫地机器人有哪些"
    c, kind = resolve_model_constraints(q)
    _assert_eq(kind, "date", "只命中日期")
    _assert_eq((c["publish_start"], c["publish_end"]), ("2025-01-01", "2025-12-31"), "日期区间")
    r = resolve_model_query(q)
    _assert(r is not None and r.models, "日期能筛出型号")
    _assert(all(20250101 <= _publish_int(m) <= 20251231 for m in r.models), "结果都在区间内")
    _assert_in("该时间段内发布的机器人有", format_model_query(r), "日期前缀")


def test_budget_and_date():
    q = "2025年发布的3000以内的机器人"
    _c, kind = resolve_model_constraints(q)
    _assert_eq(kind, "budget+date", "预算 + 日期")
    r = resolve_model_query(q)
    _assert(r is not None and r.models, "组合条件能筛出型号")
    _assert(all(int(m.get("price") or 0) <= 3000 for m in r.models), "价格条件生效")
    _assert(all(20250101 <= _publish_int(m) <= 20251231 for m in r.models), "日期条件生效")
    _assert_in("符合您预算和时间要求", format_model_query(r), "组合前缀")


def test_date_gate_and_guards():
    _assert(resolve_model_constraints("今年买的机器人保修多久") is None,
            "日期无选型信号时不直出")
    _assert(resolve_model_constraints("买了半年，现在坏了") is None,
            "购买时长不当发布日期")
    _assert(resolve_model_constraints("双十一买的机器人") is None,
            "品牌活动词不当发布日期")
    _assert(resolve_model_constraints("边刷多久换一次") is None,
            "普通问题无约束")


def test_no_match_falls_back():
    _assert(resolve_model_query("预算1元以内的扫地机器人") is None,
            "无匹配型号返回 None，由 agent 回退 RAG")


def test_internal_publish_range():
    from function_tools.model_tool import find_models_by_args
    models = find_models_by_args({"publish_start": "2025-01-01",
                                  "publish_end": "2025-12-31", "limit": 50})
    _assert(models, "内部日期区间可用")
    _assert(all(20250101 <= _publish_int(m) <= 20251231 for m in models), "内部区间生效")
    _assert_eq(len(find_models_by_args({"publish_start": "2025-01-01",
                                        "publish_end": "2025-12-31", "limit": 1})), 1,
               "limit 生效")


def test_format_empty_result():
    r = ModelQueryResult(models=[{"name": "X"}], kind="budget", constraints={})
    _assert_in("在您预算内", format_model_query(r), "格式化不依赖 agent")


TESTS = [
    test_budget_only,
    test_date_only,
    test_budget_and_date,
    test_date_gate_and_guards,
    test_no_match_falls_back,
    test_internal_publish_range,
    test_format_empty_result,
]


def run():
    return run_tests("模型查询收敛测试（预算/日期/组合/守卫）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
