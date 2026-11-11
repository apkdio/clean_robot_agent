"""型号记录仓库单测：去重合并、确定性顺序、warm 缓存不读 Chroma。"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *


class _Doc:
    def __init__(self, name, metadata=None):
        self.page_content = name
        self.metadata = metadata or {}


def _extract(doc):
    data = doc.metadata.get("_info") or {}
    return dict(data)


def test_build_records_dedupes_and_merges():
    """同名多 chunk 只留一条，参数/标签取并集；后到只补空值，不覆盖非空值。"""
    from utils import metadata_rules as me
    from mapper import models_repo as mr
    import mapper.vector_store as vs

    docs = [
        _Doc("late", {"file_name": "b.md", "start_index": 10,
                      "_info": {"name": "不染一尘云顶 X2", "price": 1,
                                "param_吸力": "6000Pa", "param_吸力_num": 6000,
                                "tag_x": True, "tag_y": True}}),
        _Doc("first", {"file_name": "a.md", "start_index": 5,
                       "_info": {"name": "不染一尘云顶X2", "price": 0,
                                 "param_避障": "LDS", "tag_x": False}}),
    ]
    saved_search, saved_extract = vs.search_by_filter, me.extract_model_info
    vs.search_by_filter = lambda _f: docs
    me.extract_model_info = _extract
    try:
        rows = mr._build_records()
    finally:
        vs.search_by_filter = saved_search
        me.extract_model_info = saved_extract

    _assert_eq(len(rows), 1, "同名（去空白）合并为一条")
    row = rows[0]
    _assert_eq(row.get("price"), 0, "price=0 是有效价格，不被当成缺失")
    _assert_eq(row.get("param_吸力"), "6000Pa", "后到 chunk 补上缺失参数")
    _assert_eq(row.get("param_避障"), "LDS", "不同 chunk 的参数取并集")
    _assert_eq(row.get("tag_x"), True, "先到 False、后到 True 时标签取 True")
    _assert_eq(row.get("tag_y"), True, "标签字段一并保留")


def test_build_records_skips_non_models_and_sorts():
    from utils import metadata_rules as me
    from mapper import models_repo as mr
    import mapper.vector_store as vs

    docs = [
        _Doc("b", {"file_name": "b.md", "start_index": 2,
                   "_info": {"name": "B", "price": 2}}),
        _Doc("a", {"file_name": "a.md", "start_index": 1,
                   "_info": {"name": "A", "price": 1}}),
        _Doc("no-price", {"file_name": "a.md", "start_index": 3,
                          "_info": {"name": "C", "price": None}}),
    ]
    saved_search, saved_extract = vs.search_by_filter, me.extract_model_info
    vs.search_by_filter = lambda _f: docs
    me.extract_model_info = _extract
    try:
        rows = mr._build_records()
    finally:
        vs.search_by_filter = saved_search
        me.extract_model_info = saved_extract
    _assert_eq([r["name"] for r in rows], ["A", "B"], "按 file_name + start_index 稳定排序，无价记录跳过")


def test_warm_cache_does_not_touch_chroma():
    from mapper import models_repo as mr
    import mapper.vector_store as vs
    from service.model_catalog import find_models_result

    saved_cache = dict(mr._cache)
    saved_search = vs.search_by_filter
    saved_fingerprint = vs.store_fingerprint
    mr._cache["fingerprint"] = "cached"
    mr._cache["records"] = [{"name": "A", "price": 1000}]
    vs.search_by_filter = lambda _f: (_ for _ in ()).throw(AssertionError("不应读 Chroma"))
    vs.store_fingerprint = lambda: (_ for _ in ()).throw(AssertionError("不应校验指纹"))
    try:
        _assert(mr.all_records() is mr._cache["records"], "warm 直接返回内存缓存")
        _assert_eq([m["name"] for m in find_models_result({"limit": 5}).models], ["A"],
                   "warm find_models 不触发 Chroma")
    finally:
        mr._cache.clear()
        mr._cache.update(saved_cache)
        vs.search_by_filter = saved_search
        vs.store_fingerprint = saved_fingerprint


TESTS = [test_build_records_dedupes_and_merges,
         test_build_records_skips_non_models_and_sorts,
         test_warm_cache_does_not_touch_chroma]


def run():
    return run_tests("型号记录仓库测试", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
