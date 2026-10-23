"""标签选择与型号标签过滤测试：纯规则，不调 LLM。"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from tools.model_tags import has_tags, reset_cache, tags_of
from tools.selection import is_skip, parse_multi_select


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


def test_model_tags_mapping():
    reset_cache()
    _assert(has_tags("不染一尘天工 T1 Pro", ["pet_friendly", "anti_tangle"]), "T1 Pro 标签")
    _assert(not has_tags("不染一尘净白 S1", ["pet_friendly"]), "S1 不具养宠标签")
    _assert_eq(tags_of("不染一尘净界 P2"), {"auto_dust", "quiet"}, "P2 标签集合")


def test_find_models_tags_filter_and_boost():
    from function_tools.model_tool import find_models_by_args
    filtered = find_models_by_args({"tags": ["pet_friendly"], "limit": 50})
    _assert(filtered, "pet_friendly 能筛出型号")
    _assert(all(has_tags(m["name"], ["pet_friendly"]) for m in filtered), "只返回带标签型号")
    boost = find_models_by_args({"tags": ["quiet"], "limit": 50})
    _assert(boost and has_tags(boost[0]["name"], ["quiet"]), "boost 标签排前")


TESTS = [
    test_selection_normalization,
    test_model_tags_mapping,
    test_find_models_tags_filter_and_boost,
]


def run():
    return run_tests("标签选择 / 型号标签过滤", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
