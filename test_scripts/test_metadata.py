"""结构化元数据提取测试：tools/metadata_extractor.py 的全部纯规则函数。

覆盖：
  - extract_price_constraint（区间/上限/下限/浮动/中文数字）
  - build_filter（预算 → Chroma where）
  - extract_price_metadata / extract_publish_date（chunk → metadata）
  - extract_model_info（型号名/系列/参数）
  - format_model_line（全字段 + aspect 维度）
  - extract_model_aspect（属性维度）
  - extract_series（系列名，含映射）
  - _match_model_filter（内存 where 过滤）
  - _cn_to_int（中文数字）
  - enumerate_models_by_series（数据依赖，自动跳过）

运行：
  .venv\\Scripts\\python.exe test_metadata.py
"""
import sys
from _runner import *


class _FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


def _models_ready():
    """型号数据（Chroma / models.pkl 缓存）是否可用。"""
    try:
        from tools.metadata_extractor import get_all_models
        return len(get_all_models()) > 0
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────
# 1. 价格约束提取
# ──────────────────────────────────────────────────────────────

def test_extract_price_constraint():
    from tools.metadata_extractor import extract_price_constraint as f
    # 显式区间
    _assert_eq(f("1000-2000"), (1000, 2000), "区间 1000-2000")
    _assert_eq(f("1000到2000"), (1000, 2000), "区间 1000到2000")
    _assert_eq(f("1000~2000"), (1000, 2000), "区间 1000~2000")
    # 单一上限
    _assert_eq(f("1000以内"), (None, 1000), "上限 1000以内")
    _assert_eq(f("1000以下"), (None, 1000), "上限 1000以下")
    _assert_eq(f("1000元以内"), (None, 1000), "上限 1000元以内")
    _assert_eq(f("1000块以内"), (None, 1000), "上限 1000块以内")
    # 下限
    _assert_eq(f("1000以上"), (1000, None), "下限 1000以上")
    _assert_eq(f("1000起"), (1000, None), "下限 1000起")
    # 浮动（±500）
    _assert_eq(f("1000左右"), (500, 1500), "浮动 1000左右")
    _assert_eq(f("2000左右"), (1500, 2500), "浮动 2000左右")
    # 中文数字
    _assert_eq(f("一千以内"), (None, 1000), "中文上限 一千以内")
    _assert_eq(f("两千左右"), (1500, 2500), "中文浮动 两千左右")
    _assert_eq(f("三千以上"), (3000, None), "中文下限 三千以上")
    # 无效
    _assert(f("不知道") is None, "无预算→None")
    _assert(f("随便") is None, "无关→None")
    _assert(f("") is None, "空串→None")


def test_build_filter():
    from tools.metadata_extractor import build_filter as f
    _assert_eq(f("1000以内"), {"min_price": {"$lte": 1000}}, "上限→min_price lte")
    _assert_eq(
        f("1000-2000"),
        {"$and": [{"min_price": {"$lte": 2000}}, {"max_price": {"$gte": 1000}}]},
        "区间→$and 双条件",
    )
    _assert_eq(f("1000以上"), {"max_price": {"$gte": 1000}}, "下限→max_price gte")
    _assert(f("扫地机器人推荐") is None, "无预算→None")


# ──────────────────────────────────────────────────────────────
# 2. chunk → metadata（入库时）
# ──────────────────────────────────────────────────────────────

def test_extract_price_metadata():
    from tools.metadata_extractor import extract_price_metadata as f
    _assert_eq(f("参考价：7999"), {"min_price": 7999, "max_price": 7999}, "单价格")
    _assert_eq(f("参考价 899，参考价 1499"), {"min_price": 899, "max_price": 1499}, "多价格取 min/max")
    _assert_eq(f("这个型号没有价格"), {}, "无价格→空 dict")


def test_extract_publish_date():
    from tools.metadata_extractor import extract_publish_date as f
    _assert_eq(f("发布时间：2026-09-15"), {"publish_date": 20260915}, "YYYY-MM-DD")
    _assert_eq(f("发布时间：2024年1月1日"), {"publish_date": 20240101}, "中文年月日")
    _assert_eq(f("无日期"), {}, "无日期→空 dict")


# ──────────────────────────────────────────────────────────────
# 3. 型号信息提取 / 格式化
# ──────────────────────────────────────────────────────────────

def test_extract_model_info():
    from tools.metadata_extractor import extract_model_info
    text = (
        "1. **不染一尘云顶 X2**\n"
        "   - 系列：云顶 X\n"
        "   - 吸力：13500Pa｜导航：LDS激光｜避障：AI双摄\n"
        "   - 参考价：7999\n"
        "   - 发布时间：2026-09-15"
    )
    info = extract_model_info(_FakeDoc(text, {"min_price": 7999}))
    _assert_eq(info["name"], "不染一尘云顶 X2", "name 提取")
    _assert_eq(info["series"], "云顶 X", "series 含空格不截断")
    _assert_eq(info["price"], 7999, "price 来自 metadata")
    _assert_eq(info["suction"], "13500Pa", "suction 提取")
    _assert_eq(info["navigation"], "LDS激光", "navigation 提取")
    _assert_eq(info["obstacle"], "AI双摄", "obstacle 提取")
    _assert_eq(info["publish_date"], "2026-09-15", "publish_date 提取")


def test_format_model_line():
    from tools.metadata_extractor import format_model_line
    info = {
        "name": "不染一尘云顶 X2", "series": "云顶 X", "price": 7999,
        "suction": "13500Pa", "navigation": "LDS激光", "obstacle": "AI双摄",
        "publish_date": "2026-09-15",
    }
    full = format_model_line(info)
    _assert_in("云顶 X2", full, "全字段含型号名")
    _assert_in("吸力", full, "全字段含吸力")
    _assert_in("参考价 7999 元", full, "全字段含价格")

    price = format_model_line(info, "价格")
    _assert_in("7999", price, "价格 aspect 含价格")
    _assert_not_in("13500Pa", price, "价格 aspect 不含吸力")

    suction = format_model_line(info, "吸力")
    _assert_in("13500Pa", suction, "吸力 aspect 含吸力")
    _assert_not_in("参考价", suction, "吸力 aspect 不含价格")

    date_line = format_model_line(info, "发布时间")
    _assert_in("2026-09-15", date_line, "发布时间 aspect 含日期")


def test_extract_model_aspect():
    from tools.metadata_extractor import extract_model_aspect as f
    _assert_eq(f("多少钱"), "价格", "多少钱→价格")
    _assert_eq(f("吸力多大"), "吸力", "吸力→吸力")
    _assert_eq(f("什么时候发布"), "发布时间", "什么时候发布→发布时间")
    _assert_eq(f("导航怎么样"), "导航", "导航→导航")
    _assert_eq(f("避障能力如何"), "避障", "避障→避障")
    _assert_eq(f("这款怎么样"), "", "怎么样→无 aspect")
    _assert_eq(f("它和那个有什么区别"), "", "对比→无 aspect")


# ──────────────────────────────────────────────────────────────
# 4. 系列提取 / 中文数字 / 过滤
# ──────────────────────────────────────────────────────────────

def test_extract_series():
    from tools.metadata_extractor import extract_series as f
    _assert_eq(f("净白 S 系列有什么产品"), "净白 S", "净白 S 系列")
    _assert_eq(f("天工 T 系列有哪些型号"), "天工 T", "天工 T 系列")
    _assert_eq(f("云顶 X 系列推荐"), "云顶 X", "云顶 X 系列")
    _assert_eq(f("净界 P"), "净界 P", "净界 P 不带系列字样")
    _assert_eq(f("净白"), "净白 S", "映射：净白→净白 S")
    _assert_eq(f("云顶"), "云顶 X", "映射：云顶→云顶 X")
    _assert_eq(f("扫地机器人推荐"), "", "无系列→空")


def test_cn_to_int():
    from tools.metadata_extractor import _cn_to_int as f
    _assert_eq(f("一千"), 1000, "一千")
    _assert_eq(f("两千"), 2000, "两千")
    _assert_eq(f("三百"), 300, "三百")
    _assert_eq(f("五"), 5, "五")
    _assert_eq(f("十"), 10, "十")
    _assert_eq(f("十五"), 15, "十五")
    _assert_eq(f("一万"), 10000, "一万")


def test_match_model_filter():
    from tools.metadata_extractor import _match_model_filter as f
    info = {"name": "X2", "price": 7999, "series": "云顶 X", "publish_date": "2026-09-15"}
    _assert(f(info, {"min_price": {"$lte": 8000}}), "min_price lte 命中")
    _assert(not f(info, {"min_price": {"$lte": 1000}}), "min_price lte 未命中")
    _assert(f(info, {"max_price": {"$gte": 1000}}), "max_price gte 命中")
    _assert(f(info, {"$and": [{"min_price": {"$lte": 8000}}, {"max_price": {"$gte": 1000}}]}), "$and 命中")
    _assert(f(info, {"publish_date": {"$gte": 20260101}}), "publish_date gte 命中")
    _assert(f(info, {"series": "云顶 X"}), "series 命中")
    _assert(not f(info, {"series": "净白 S"}), "series 未命中")
    _assert(f(info, None), "None 过滤全通过")
    _assert(f(info, {}), "空过滤全通过")


def test_enumerate_models_by_series():
    if not _models_ready():
        _skip("无型号数据（Chroma/pkl 缓存缺失）")
        return
    from tools.metadata_extractor import enumerate_models_by_series, SERIES_LIST
    expected = {"净白 S": 3, "净界 P": 5, "天工 T": 5, "云顶 X": 3}
    for series, cnt in expected.items():
        models = enumerate_models_by_series(series)
        _assert_eq(len(models), cnt, f"{series} 系列应有 {cnt} 个型号")
    for series in SERIES_LIST:
        for m in enumerate_models_by_series(series):
            _assert_eq(m.get("series"), series, f"{m.get('name')} 的 series 应为 {series}")
    _assert_eq(len(enumerate_models_by_series("不存在系列")), 0, "不存在系列→0 个")


TESTS = [
    test_extract_price_constraint,
    test_build_filter,
    test_extract_price_metadata,
    test_extract_publish_date,
    test_extract_model_info,
    test_format_model_line,
    test_extract_model_aspect,
    test_extract_series,
    test_cn_to_int,
    test_match_model_filter,
    test_enumerate_models_by_series,
]


def run():
    reset()
    return run_tests("结构化元数据提取测试（metadata_extractor）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
