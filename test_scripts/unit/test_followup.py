"""追问解析与相对维度：最后命中 index、相对排序、followup 迁移后的入口。"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *

import tools.model_catalog as mt
from tools.model_catalog import filter_by_relative
from tools import followup as f

_FAKE = [
    {"name": "A", "price": 1000, "publish_date": "2025-01-01"},
    {"name": "B", "price": 2000, "publish_date": "2025-06-01"},
    {"name": "C", "price": 3000, "publish_date": "2025-09-01"},
]
_PREV = [_FAKE[1]]


def test_relative_rels_keeps_last_in_pair():
    _assert_eq(f.relative_rels("我想要便宜的，不对，贵的"), ["pricier"],
               "同一互斥对取最后出现的「贵的」")
    _assert_eq(f.relative_rels("我想要贵的，不对，便宜的"), ["cheaper"],
               "反向也取最后出现的「便宜的」")
    _assert_eq(f.relative_rels("更新更便宜"), ["newer", "cheaper"],
               "不同维度并列保留")
    _assert_eq(f.relative_rels("更便宜更新款"), ["cheaper", "newer"],
               "并列按 query 顺序")
    _assert_eq(f.relative_rels("更新更旧"), ["older"], "同一对只留最后命中的")
    _assert_eq(f.relative_rels("随便看看"), [], "无相对维度")


def test_filter_by_relative_order_and_multi_dimension():
    _assert_eq([m["name"] for m in filter_by_relative(_FAKE, _PREV, ["cheaper"])],
               ["A"], "更便宜取边界下方")
    _assert_eq([m["name"] for m in filter_by_relative(_FAKE, _PREV, ["pricier"])],
               ["C"], "更贵取边界上方")
    _assert_eq([m["name"] for m in filter_by_relative(_FAKE, _PREV, ["newer"])],
               ["C"], "更新取边界之后")
    _assert_eq([m["name"] for m in filter_by_relative(_FAKE, _PREV, ["older"])],
               ["A"], "更旧取边界之前")
    _assert_eq(filter_by_relative(_FAKE, _PREV, ["newer", "cheaper"]), [],
               "不同维度同时满足时取交集")
    _assert_eq(filter_by_relative(_FAKE, [], ["cheaper"]), [],
               "没有上一轮边界 → 空")


def test_find_models_result_relative_scope_is_sorted():
    orig = mt.all_records
    mt.all_records = lambda: list(_FAKE)
    try:
        result = mt.find_models_result({"scope": ["pricier"], "limit": 2}, _PREV)
        _assert_eq([m["name"] for m in result.models], ["C"],
                   "find_models 的 scope 不只过滤，也按相对边界排序")
    finally:
        mt.all_records = orig


def test_handle_followup_uses_last_recommend():
    saved = {}
    orig_get, orig_set, orig_search = f.get_last_models, f.set_last_models, f._search_global
    f.get_last_models = lambda sid: _PREV
    f.set_last_models = lambda sid, models: saved.update(models=list(models))
    f._search_global = lambda *args, **kwargs: list(_FAKE)
    try:
        reply = f.handle_followup("s", "有没有更便宜的")
        _assert_in("更便宜", reply or "", "相对追问有抬头")
        _assert_eq([m["name"] for m in saved.get("models", [])], ["A"], "保存更便宜结果")

        saved.clear()
        reply = f.handle_followup("s", "我想要便宜的，不对，贵的")
        _assert_in("更贵", reply or "", "最后命中「贵的」走更贵")
        _assert_in("我猜您想要的是「更贵」", reply or "", "互斥对冲突先给猜测引导")
        _assert_eq([m["name"] for m in saved.get("models", [])], ["C"], "保存更贵结果")

        saved.clear()
        reply = f.handle_followup("s", "最便宜的是哪款")
        _assert_in("最便宜", reply or "", "全局最便宜")
        _assert_eq([m["name"] for m in saved.get("models", [])], ["A"], "保存最便宜结果")
    finally:
        f.get_last_models, f.set_last_models, f._search_global = \
            orig_get, orig_set, orig_search


def test_handle_followup_without_prev_returns_none():
    orig_get, orig_search = f.get_last_models, f._search_global
    f.get_last_models = lambda sid: None
    f._search_global = lambda *args, **kwargs: list(_FAKE)
    try:
        _assert_eq(f.handle_followup("s", "有没有更便宜的"), None,
                   "没有上一轮推荐时不接管相对追问")
    finally:
        f.get_last_models, f._search_global = orig_get, orig_search


TESTS = [
    test_relative_rels_keeps_last_in_pair,
    test_filter_by_relative_order_and_multi_dimension,
    test_find_models_result_relative_scope_is_sorted,
    test_handle_followup_uses_last_recommend,
    test_handle_followup_without_prev_returns_none,
]


def run():
    return run_tests("追问解析与相对维度", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
