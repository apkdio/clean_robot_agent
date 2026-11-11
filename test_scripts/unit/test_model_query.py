"""模型查询收敛测试：预算/日期约束 → find_models → 直出格式。"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from service.model_query import (format_model_query, parse_budget, parse_publish,
                               resolve_model_constraints, resolve_model_query)
from utils.dto.model_query import FindModelsResult, ModelQueryConstraints
from utils.date_rules import publish_int


def _publish_int(model):
    return publish_int(model.get("publish_date"))


def test_budget_only():
    c, kind = resolve_model_constraints("预算2000以内的扫地机器人")
    _assert_eq(kind, "budget", "只命中预算")
    _assert_eq((c.budget.min, c.budget.max), (None, 2000), "预算上下限")
    r = resolve_model_query("预算2000以内的扫地机器人")
    _assert(r is not None and r.models, "预算能筛出型号")
    _assert(all(int(m.get("price") or 0) <= 2000 for m in r.models), "结果都在预算内")
    _assert_in("在您选择的条件内（预算 2000 元以内）", format_model_query(r), "预算前缀")


def test_colloquial_spend_budget():
    q = "我现在银行卡有3000块，但是我只能在机器人上支出2000块，有什么型号可以买"
    c, kind = resolve_model_constraints(q, allow_llm=False)
    _assert_eq(kind, "budget", "「支出 2000 块」按预算上限识别")
    _assert_eq((c.budget.min, c.budget.max), (None, 2000), "只取支出上限，不取银行卡余额")
    _assert_eq(resolve_model_constraints("随便聊聊支出"), None, "没有金额不误判预算")


def test_date_only():
    q = "2025年发布的扫地机器人有哪些"
    c, kind = resolve_model_constraints(q)
    _assert_eq(kind, "date", "只命中日期")
    _assert_eq((c.publish.start, c.publish.end), (20250101, 20251231), "日期区间统一 int YYYYMMDD")
    r = resolve_model_query(q)
    _assert(r is not None and r.models, "日期能筛出型号")
    _assert(all(20250101 <= _publish_int(m) <= 20251231 for m in r.models), "结果都在区间内")
    _assert_in("2025-01-01 至 2025-12-31 发布", format_model_query(r), "日期前缀")


def test_budget_and_date():
    q = "2025年发布的3000以内的机器人"
    _c, kind = resolve_model_constraints(q)
    _assert_eq(kind, "budget+date", "预算 + 日期")
    r = resolve_model_query(q)
    _assert(r is not None and r.models, "组合条件能筛出型号")
    _assert(all(int(m.get("price") or 0) <= 3000 for m in r.models), "价格条件生效")
    _assert(all(20250101 <= _publish_int(m) <= 20251231 for m in r.models), "日期条件生效")
    _assert_in("预算 3000 元以内、2025-01-01 至 2025-12-31 发布",
               format_model_query(r), "组合条件前缀")


def test_series_catalog_intent():
    c, kind = resolve_model_constraints("天工 T 哪几款", allow_llm=False)
    _assert_eq(kind, "series", "「哪几款」也触发系列枚举")
    _assert_eq(c.series, "天工 T", "系列约束从 repo 系列目录提取")


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
    from service.model_catalog import find_models_result
    models = find_models_result({"publish_start": "2025-01-01",
                                 "publish_end": "2025-12-31", "limit": 50}).models
    _assert(models, "内部日期区间可用")
    _assert(all(20250101 <= _publish_int(m) <= 20251231 for m in models), "内部区间生效")
    _assert_eq(len(find_models_result({"publish_start": "2025-01-01",
                                       "publish_end": "2025-12-31", "limit": 1}).models), 1,
               "limit 生效")


def test_parse_constraint_dtos():
    b = parse_budget("预算2000以内的扫地机器人", allow_llm=False)
    _assert(b is not None and b.min is None and b.max == 2000, "预算 DTO")
    p = parse_publish("2025年发布的扫地机器人", allow_llm=False)
    _assert(p is not None and (p.start, p.end) == (20250101, 20251231),
            "日期 DTO 内部统一 YYYYMMDD")
    _assert(parse_budget("随便聊聊", allow_llm=False) is None, "无预算 → None")
    _assert(parse_publish("边刷多久换一次", allow_llm=False) is None, "无日期 → None")


def test_format_empty_result():
    r = FindModelsResult(models=[{"name": "X"}], kind="budget",
                         constraints=ModelQueryConstraints())
    _assert_in("在您选择的条件内（全部型号）", format_model_query(r), "格式化不依赖 agent")


TESTS = [
    test_budget_only,
    test_colloquial_spend_budget,
    test_date_only,
    test_budget_and_date,
    test_series_catalog_intent,
    test_date_gate_and_guards,
    test_no_match_falls_back,
    test_internal_publish_range,
    test_parse_constraint_dtos,
    test_format_empty_result,
]


def run():
    return run_tests("模型查询收敛测试（预算/日期/组合/守卫）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
