"""Function Tools 测试：日期 / 预算 / 型号 / 症状 / 网点 五个工具包。

覆盖：
  - date_tool：parse_date / parse_absolute_date / parse_relative_date /
    calc_date_range / build_date_filter / _iso_to_int / _cn_to_int
    （注入 _DATE_TODAY 固定日期，保证相对日期断言确定）
  - budget_tool：budget_args_to_filter（LLM 返回 args → Chroma where）
  - model_tool：search_models_by_names / model_name_in_query（monkeypatch 型号源）
  - symptom_tool：symptom_id_to_query（编号 → 标准检索 query）
  - service_point_tool：haversine / _pick_cn_name / search_service_points /
    format_service_points / geocode_city"""
import sys
from datetime import date
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_SCRIPT_DIR) not in sys.path:   # test_scripts/：供 `_runner` 导入
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

from _runner import *


# ──────────────────────────────────────────────────────────────
# 1. 日期工具（固定“今天”= 2026-09-04）
# ──────────────────────────────────────────────────────────────

def _fix_today():
    import function_tools.date_tool as dt
    dt._DATE_TODAY = date(2026, 9, 4)
    return dt


def test_parse_absolute_date():
    dt = _fix_today()
    _assert_eq(dt.parse_absolute_date("2025年三月"), ("2025-03-01", "2025-03-31"), "2025年三月")
    _assert_eq(dt.parse_absolute_date("2025年3月"), ("2025-03-01", "2025-03-31"), "2025年3月")
    _assert_eq(dt.parse_absolute_date("2025年"), ("2025-01-01", "2025-12-31"), "2025年")
    _assert_eq(dt.parse_absolute_date("2025-03"), ("2025-03-01", "2025-03-31"), "2025-03")
    _assert_eq(dt.parse_absolute_date("三月"), ("2026-03-01", "2026-03-31"), "三月→当年")
    _assert(dt.parse_absolute_date("随便") is None, "非日期→None")


def test_parse_relative_date():
    dt = _fix_today()
    _assert_eq(dt.parse_relative_date("今年"), ("2026-01-01", "2026-09-04"), "今年")
    _assert_eq(dt.parse_relative_date("去年"), ("2025-01-01", "2025-12-31"), "去年")
    _assert_eq(dt.parse_relative_date("前年"), ("2024-01-01", "2024-12-31"), "前年")
    _assert_eq(dt.parse_relative_date("最近半年"), ("2026-03-04", "2026-09-04"), "最近半年")
    _assert_eq(dt.parse_relative_date("近三个月"), ("2026-06-04", "2026-09-04"), "近三个月")
    _assert_eq(dt.parse_relative_date("半年"), ("2026-03-04", "2026-09-04"), "半年")
    _assert_eq(dt.parse_relative_date("两周内"), ("2026-08-21", "2026-09-04"), "两周内")
    _assert_eq(dt.parse_relative_date("30天内"), ("2026-08-05", "2026-09-04"), "30天内")
    _assert(dt.parse_relative_date("随便") is None, "非相对日期→None")


def test_calc_date_range_and_filter():
    dt = _fix_today()
    r = dt.calc_date_range("半年")
    _assert_eq(r["start_date"], "2026-03-04", "半年 start")
    _assert_eq(r["end_date"], "2026-09-04", "半年 end")
    _assert_eq(r["expression"], "半年", "半年 expression")
    err = dt.calc_date_range("瞎写")
    _assert_in("error", err, "无法识别→error")
    _assert_eq(dt._iso_to_int("2026-03-10"), 20260310, "_iso_to_int")
    f = dt.build_date_filter("2026-01-01", "2026-12-31")
    _assert_eq(
        f,
        {"$and": [{"publish_date": {"$gte": 20260101}}, {"publish_date": {"$lte": 20261231}}]},
        "build_date_filter 闭区间",
    )
    _assert_eq(
        dt.build_date_filter("2026-01-01"),
        {"publish_date": {"$gte": 20260101}},
        "build_date_filter 单边",
    )


def test_date_cn_to_int():
    dt = _fix_today()
    _assert_eq(dt._cn_to_int("三"), 3, "三")
    _assert_eq(dt._cn_to_int("十"), 10, "十")
    _assert_eq(dt._cn_to_int("十五"), 15, "十五")
    _assert_eq(dt._cn_to_int("二十"), 20, "二十")


# ──────────────────────────────────────────────────────────────
# 2. 预算工具
# ──────────────────────────────────────────────────────────────

def test_budget_args_to_filter():
    from function_tools.budget_tool import budget_args_to_filter as f
    _assert_eq(f({"budget_max": 1000}), {"min_price": {"$lte": 1000}}, "整数上限")
    _assert_eq(f({"budget_max": "1500"}), {"min_price": {"$lte": 1500}}, "字符串数字上限")
    _assert(f({"budget_max": 0}) is None, "0→None（无预算）")
    _assert(f({"budget_max": None}) is None, "None→None")
    _assert(f({}) is None, "空 args→None")
    _assert(f({"budget_max": "abc"}) is None, "非数字→None")


# ──────────────────────────────────────────────────────────────
# 3. 型号工具（monkeypatch 型号源，纯逻辑）
# ──────────────────────────────────────────────────────────────

def test_search_models_by_names():
    import function_tools.model_tool as mt
    fake = [
        {"name": "不染一尘云顶 X2", "price": 7999},
        {"name": "不染一尘净白 S1", "price": 899},
    ]
    orig = mt._enumerate_model_infos
    mt._enumerate_model_infos = lambda: fake
    try:
        _assert_eq([m["name"] for m in mt.search_models_by_names("云顶 X2")], ["不染一尘云顶 X2"], "单型号子串匹配")
        _assert_eq(len(mt.search_models_by_names("云顶 X2,净白 S1")), 2, "逗号分隔多型号")
        _assert_eq([m["name"] for m in mt.search_models_by_names("云顶 X 系列")], ["不染一尘云顶 X2"], "系列名兜底命中")
        _assert_eq(mt.search_models_by_names(""), [], "空→空")
        _assert_eq(mt.search_models_by_names("不存在"), [], "不存在→空")
    finally:
        mt._enumerate_model_infos = orig


def test_model_name_in_query():
    import function_tools.model_tool as mt
    orig = mt._model_names_cache
    mt._model_names_cache = ["不染一尘云顶 X2", "不染一尘净白 S1"]
    try:
        _assert(mt.model_name_in_query("云顶 X2 怎么样"), "含型号名→True")
        _assert(mt.model_name_in_query("净白 S1 多少钱"), "净白 S1→True")
        _assert(not mt.model_name_in_query("扫地机器人怎么样"), "泛咨询→False")
    finally:
        mt._model_names_cache = orig


# ──────────────────────────────────────────────────────────────
# 4. 症状工具
# ──────────────────────────────────────────────────────────────

def test_symptom_id_to_query():
    from function_tools.symptom_tool import symptom_id_to_query as f
    _assert_eq(f({"symptom_id": 1}), "机器人不移动怎么办", "id=1 不移动")
    _assert_eq(f({"symptom_id": 7}), "吸力下降怎么办", "id=7 吸力")
    _assert_eq(f({"symptom_id": "3"}), "扫地机器人异响怎么办", "字符串数字 id")
    _assert(f({"symptom_id": 0}) is None, "id=0→None")
    _assert(f({}) is None, "空→None")
    _assert(f({"symptom_id": 99}) is None, "越界→None")


# ──────────────────────────────────────────────────────────────
# 5. 网点工具
# ──────────────────────────────────────────────────────────────

def test_haversine():
    from function_tools.service_point_tool import haversine
    _assert_eq(haversine(0, 0, 0, 0), 0.0, "同点距离为 0")
    d = haversine(39.9042, 116.4074, 31.2304, 121.4737)  # 北京→上海
    _assert(1000 < d < 1150, f"北京-上海约 1067km（实际 {d:.0f}）")
    _assert_eq(haversine(31.2304, 121.4737, 39.9042, 116.4074), d, "距离对称")


def test_pick_cn_name():
    from function_tools.service_point_tool import _pick_cn_name
    _assert_eq(_pick_cn_name(["Beijing", "北京", "北京市"]), "北京市", "优先行政区划后缀")
    _assert(_pick_cn_name(["Beijing", "New York"]) is None, "无中文名→None")
    _assert_eq(_pick_cn_name(["上海"]), "上海", "仅一个中文名")


def test_search_service_points():
    from function_tools.service_point_tool import search_service_points
    pts, origin = search_service_points(lng=116.4074, lat=39.9042)
    _assert(len(pts) > 0, "有网点数据")
    _assert(len(pts) <= 5, "最多返回 5 个")
    _assert_eq(origin, "您当前位置", "带经纬度时来源为当前位置")
    for p in pts:
        _assert("distance_km" in p, f"{p.get('name')} 有距离")
    # 距离升序
    dists = [p["distance_km"] for p in pts if p.get("distance_km") is not None]
    _assert(dists == sorted(dists), "按距离升序")
    no_coord, _ = search_service_points()
    _assert_eq(len(no_coord), 5, "无经纬度返回前 5 个")


def test_format_service_points():
    from function_tools.service_point_tool import format_service_points as f
    _assert_in("没有查询到", f([], ""), "空网点话术")
    txt = f([{"name": "上海旗舰店", "address": "南京路1号", "phone": "400-1", "hours": "9-18", "distance_km": 1.5}], "您当前位置")
    _assert_in("上海旗舰店", txt, "含网点名")
    _assert_in("南京路1号", txt, "含地址")
    _assert_in("1.5", txt, "含距离")
    _assert_in("您当前位置", txt, "含来源描述")


def test_geocode_city():
    try:
        from function_tools.service_point_tool import geocode_city
    except Exception:
        _skip("geonamescache 不可用")
        return
    cands = geocode_city("上海")
    if not cands:
        _skip("geonamescache 未匹配到上海")
        return
    _assert(len(cands) >= 1, "上海有候选")
    _assert_in("上海", (cands[0].get("cn_name") or cands[0].get("name") or ""), "候选含上海")
    _assert_eq(geocode_city(""), [], "空城市名→空")


TESTS = [
    test_parse_absolute_date,
    test_parse_relative_date,
    test_calc_date_range_and_filter,
    test_date_cn_to_int,
    test_budget_args_to_filter,
    test_search_models_by_names,
    test_model_name_in_query,
    test_symptom_id_to_query,
    test_haversine,
    test_pick_cn_name,
    test_search_service_points,
    test_format_service_points,
    test_geocode_city,
]


def run():
    reset()
    return run_tests("Function Tools 测试（date/budget/model/symptom/service_point）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
