"""find_models 参数校验：非法条件只丢弃自身，不把已筛结果清零。"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *


_FAKE = [
    {"name": "A", "price": 1000, "series": "净白 S", "publish_date": "2025-03-01",
     "param_吸力_num": 6000, "param_导航": "LDS激光", "tag_pet_friendly": True},
    {"name": "B", "price": 2000, "series": "净白 S", "publish_date": "2025-06-01",
     "param_吸力_num": 9000, "param_导航": "视觉导航", "tag_pet_friendly": False},
]


def _with_fake(fn):
    import tools.model_catalog as mt
    orig = mt.all_records
    mt.all_records = lambda: _FAKE
    try:
        return fn()
    finally:
        mt.all_records = orig


def _run(args):
    import tools.model_catalog as mt
    return _with_fake(lambda: mt.find_models_result(args))


def test_valid_and_compat_entry():
    import tools.model_catalog as mt
    result = _run({"budget_min": 1500, "budget_max": 2000})
    _assert_eq([m["name"] for m in result.models], ["B"], "合法预算正常筛选")
    _assert_eq(result.dropped, [], "合法参数无 dropped")
    names = _with_fake(lambda: [m["name"] for m in mt.find_models_result(
        {"budget_max": 1000}).models])
    _assert_eq(names, ["A"], "预算上限正常筛选")


def test_invalid_conditions_are_dropped_not_cleared():
    cases = [
        ({"budget_min": "abc"}, "budget_min"),
        ({"publish_start": "2025"}, "publish_start"),
        ({"params": [{"key": "吸力", "op": ">", "value": "6000"}]}, "params[0]"),
        ({"params": [{"key": "吸力", "op": ">=", "value": "强"}]}, "params[0]"),
        ({"params": [{"key": "", "op": "=", "value": "x"}]}, "params[0]"),
        ({"scope": ["unknown"]}, "scope"),
        ({"scope": ["prev"]}, "scope"),
        ({"sort_by": "不存在的参数"}, "sort_by"),
        ({"tags": ["unknown"]}, "tags"),
        ({"limit": "many"}, "limit"),
    ]
    for args, field in cases:
        result = _run(args)
        _assert_eq(len(result.models), 2, "%r 不应清空其他结果" % args)
        _assert_in(field, result.dropped, "%r 应记录 dropped" % field)
        _assert(result.notes, "%r 应有诊断说明" % field)


def test_invalid_date_and_unknown_series_do_not_return_all():
    result = _run({"publish_start": "2025"})
    _assert(result.dropped, "短日期要 dropped")
    result = _run({"series": "不存在系列"})
    _assert_eq(result.models, [], "未知系列不能返回全量")


def test_zero_none_sentinel_and_tags():
    result = _run({"budget_min": 0, "budget_max": None, "tags": ["pet_friendly"]})
    _assert_eq([m["name"] for m in result.models], ["A"], "0/None 预算视为无约束，标签仍然生效")


def test_direct_query_rejects_dropped_result():
    """有 dropped 时直出不能把剩余全量当答案，必须交回 RAG 兜底。"""
    import tools.model_query as mq
    from tools.dto.model_query import FindModelsResult, ModelQueryConstraints

    saved_constraints, saved_find = mq.resolve_model_constraints, mq.find_models
    mq.resolve_model_constraints = lambda query: (ModelQueryConstraints(limit=50), "budget")
    mq.find_models = lambda args: FindModelsResult(_FAKE, ["budget"], ["bad"])
    try:
        _assert_eq(mq.resolve_model_query("x"), None, "dropped 结果不能直出全量")
    finally:
        mq.resolve_model_constraints, mq.find_models = saved_constraints, saved_find


TESTS = [test_valid_and_compat_entry,
         test_invalid_conditions_are_dropped_not_cleared,
         test_invalid_date_and_unknown_series_do_not_return_all,
         test_zero_none_sentinel_and_tags,
         test_direct_query_rejects_dropped_result]


def run():
    return run_tests("find_models 参数校验测试", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
