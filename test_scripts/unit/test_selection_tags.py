"""标签选择与型号标签过滤测试：纯规则，不调 LLM。"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from utils.selection import is_skip, parse_index, parse_multi_select


def test_selection_normalization():
    _assert_eq(parse_multi_select("1,3", 5), ("ok", [1, 3]), "英文逗号")
    _assert_eq(parse_multi_select("1，3", 5), ("ok", [1, 3]), "中文逗号")
    _assert_eq(parse_multi_select("1、3", 5), ("ok", [1, 3]), "顿号")
    _assert_eq(parse_multi_select("１，３", 5), ("ok", [1, 3]), "全角数字")
    _assert_eq(parse_multi_select("1 3", 5), ("ok", [1, 3]), "空格分隔")
    _assert_eq(parse_multi_select("1,1,3", 5), ("ok", [1, 3]), "去重")
    _assert_eq(parse_multi_select("1,9", 5), ("invalid", []), "越界")
    _assert_eq(parse_multi_select("有猫", 5), ("invalid", []), "不做标签文字匹配")
    _assert_eq(parse_multi_select("跳过", 5), ("skip", []), "显式跳过")
    _assert(is_skip("不需要"), "固定跳过命令")


def test_index_parsing():
    _assert_eq(parse_index("2", 2), ("ok", 2), "裸数字")
    _assert_eq(parse_index("第2个", 2), ("ok", 2), "第/个")
    _assert_eq(parse_index("选 一", 2), ("ok", 1), "选/中文数字/空格")
    _assert_eq(parse_index("10", 2), ("invalid", None), "越界不截断")
    _assert_eq(parse_index("2个都不对，选1", 2), ("ok", 1), "句中明确选择取最后一个")
    _assert_eq(parse_index("跳过", 2, allow_skip=True), ("skip", None), "单序号也支持 skip")
    _assert_eq(parse_index("跳过", 0, allow_skip=True), ("skip", None), "无候选也认跳过")
    _assert_eq(parse_index("跳过", 2), ("invalid", None), "未开 skip 不消费跳过词")
    _assert_eq(parse_multi_select("2个都不对，选1", 5), ("invalid", []),
               "多选不猜句子里的局部选择")
    _assert_eq(parse_multi_select("1,10", 5), ("invalid", []), "多选同样拒绝越界")


def test_model_tags_mapping():
    from mapper.models_repo import all_records
    by_name = {m["name"]: m for m in all_records()}
    t1 = by_name["不染一尘天工 T1 Pro"]
    _assert(t1.get("tag_pet_friendly") is True and t1.get("tag_anti_tangle") is True,
            "T1 Pro 标签来自 KB")
    _assert(by_name["不染一尘净白 S1"].get("tag_pet_friendly") is False, "S1 不具养宠标签")
    p2 = by_name["不染一尘净界 P2"]
    _assert(p2.get("tag_auto_dust") is True and p2.get("tag_quiet") is True, "P2 标签集合")


def test_find_models_tags_filter_and_boost():
    from service.model_catalog import find_models_result
    filtered = find_models_result({"tags": ["pet_friendly"], "limit": 50}).models
    _assert(filtered, "pet_friendly 能筛出型号")
    _assert(all(m.get("tag_pet_friendly") is True for m in filtered), "只返回带标签型号")
    boost = find_models_result({"tags": ["quiet"], "limit": 50}).models
    _assert(boost and boost[0].get("tag_quiet") is True, "boost 标签排前")


TESTS = [
    test_selection_normalization,
    test_index_parsing,
    test_model_tags_mapping,
    test_find_models_tags_filter_and_boost,
]


def run():
    return run_tests("标签选择 / 型号标签过滤", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
