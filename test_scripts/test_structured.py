"""结构化维度提取测试：型号属性(aspect) + 系列(series)。

覆盖最近的改动：
  1. 方向2：型号属性精准查询（extract_model_aspect + format_model_line(aspect)）
  2. 系列字段 + 系列查询（extract_model_info.series + extract_series + enumerate_models_by_series）

运行：
  python test_structured.py            # 单元测试（纯规则，快，不依赖 Chroma/LLM）
  python test_structured.py --db       # + 集成测试（需 Chroma 已入库 series 字段）
  python test_structured.py --e2e      # + 端到端测试（需 Chroma + 本地 LLM）
"""

import os
import sys
import uuid

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

_total = 0
_passed = 0
_failed = []


def _assert(cond: bool, msg: str):
    global _total, _passed, _failed
    _total += 1
    if cond:
        _passed += 1
        print(f"  OK  {msg}")
    else:
        _failed.append((msg, "断言失败"))
        print(f"  FAIL {msg}")


def _assert_eq(a, b, msg: str):
    global _total, _passed, _failed
    _total += 1
    if a == b:
        _passed += 1
        print(f"  OK  {msg}")
    else:
        _failed.append((msg, f"期望={b!r}, 实际={a!r}"))
        print(f"  FAIL {msg}  期望={b!r}  实际={a!r}")


def _assert_in(content, container, msg: str):
    global _total, _passed, _failed
    _total += 1
    if content in container:
        _passed += 1
        print(f"  OK  {msg}")
    else:
        _failed.append((msg, f"内容 '{content}' 不在结果中"))
        print(f"  FAIL {msg}  — 未找到 '{content}'")


def _assert_not_in(content, container, msg: str):
    global _total, _passed, _failed
    _total += 1
    if content not in container:
        _passed += 1
        print(f"  OK  {msg}")
    else:
        _failed.append((msg, f"内容 '{content}' 不应出现"))
        print(f"  FAIL {msg}  — 不应出现 '{content}'")


# ================================================================
# 1. 单元测试：extract_model_aspect（属性维度规则提取）
# ================================================================

def test_extract_model_aspect():
    from tools.metadata_extractor import extract_model_aspect
    _assert_eq(extract_model_aspect("云顶 X2 多少钱"), "价格", "多少钱→价格")
    _assert_eq(extract_model_aspect("净白 S1 吸力多大"), "吸力", "吸力多大→吸力")
    _assert_eq(extract_model_aspect("云顶 X2 什么时候发布的"), "发布时间", "什么时候发布→发布时间")
    _assert_eq(extract_model_aspect("这款导航怎么样"), "导航", "导航→导航")
    _assert_eq(extract_model_aspect("避障能力如何"), "避障", "避障→避障")
    _assert_eq(extract_model_aspect("有没有便宜一点的"), "价格", "便宜→价格")
    _assert_eq(extract_model_aspect("这款怎么样"), "", "怎么样→无 aspect")
    _assert_eq(extract_model_aspect("它和那个有什么区别"), "", "区别→无 aspect（对比不是属性维度）")


# ================================================================
# 2. 单元测试：format_model_line(aspect)（按维度过滤输出）
# ================================================================

def test_format_model_line_aspect():
    from tools.metadata_extractor import format_model_line
    info = {
        "name": "不染一尘云顶 X2", "series": "云顶 X", "price": 7999,
        "suction": "13500Pa", "navigation": "LDS激光", "obstacle": "AI双摄",
        "publish_date": "2026-09-15",
    }
    line_price = format_model_line(info, "价格")
    _assert_in("7999", line_price, "价格 aspect 含价格")
    _assert_not_in("13500Pa", line_price, "价格 aspect 不含吸力")

    line_suction = format_model_line(info, "吸力")
    _assert_in("13500Pa", line_suction, "吸力 aspect 含吸力")
    _assert_not_in("参考价", line_suction, "吸力 aspect 不含价格")

    line_date = format_model_line(info, "发布时间")
    _assert_in("2026-09-15", line_date, "发布时间 aspect 含日期")

    line_full = format_model_line(info)
    _assert_in("吸力", line_full, "全字段含吸力")
    _assert_in("参考价", line_full, "全字段含价格")
    _assert_in("发布日期", line_full, "全字段含发布时间")


# ================================================================
# 3. 单元测试：extract_model_info 提取 series（含空格不截断）
# ================================================================

class _FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


def test_extract_model_info_series():
    from tools.metadata_extractor import extract_model_info
    cases = [
        ("净白 S1", "净白 S"), ("净界 P2 Lite", "净界 P"),
        ("天工 T2 Ultra", "天工 T"), ("云顶 X1 Pro Max", "云顶 X"),
    ]
    for name, expect_series in cases:
        text = (
            f"1. **不染一尘{name}**\n"
            f"   - 系列：{expect_series}\n"
            f"   - 吸力：5000Pa｜导航：LDS激光｜避障：结构光\n"
            f"   - 参考价：1999\n"
            f"   - 发布时间：2026-01-01"
        )
        info = extract_model_info(_FakeDoc(text, {"min_price": 1999}))
        _assert_eq(info["series"], expect_series, f"series 提取（{name}）不截断空格")
        _assert_eq(info["name"], f"不染一尘{name}", f"name 提取（{name}）")


# ================================================================
# 4. 单元测试：extract_series（系列名规则提取）
# ================================================================

def test_extract_series():
    from tools.metadata_extractor import extract_series
    _assert_eq(extract_series("净白 S 系列有什么产品"), "净白 S", "净白 S 系列")
    _assert_eq(extract_series("天工 T 系列有哪些型号"), "天工 T", "天工 T 系列")
    _assert_eq(extract_series("云顶 X 系列推荐"), "云顶 X", "云顶 X 系列")
    _assert_eq(extract_series("净界 P"), "净界 P", "净界 P（不带系列字样）")
    _assert_eq(extract_series("扫地机器人推荐"), "", "无系列名→空")
    _assert_eq(extract_series("帮我推荐一款"), "", "纯推荐（无系列）→空")


# ================================================================
# 5. 集成测试：enumerate_models_by_series（需 Chroma 已入库）
# ================================================================

def test_enumerate_models_by_series():
    from tools.metadata_extractor import enumerate_models_by_series, SERIES_LIST

    expected = {
        "净白 S": 3, "净界 P": 5, "天工 T": 5, "云顶 X": 3,
    }
    for series, cnt in expected.items():
        models = enumerate_models_by_series(series)
        _assert_eq(len(models), cnt, f"{series} 系列应有 {cnt} 个型号")

    # 每个型号的 series 字段应与查询系列一致
    for series in SERIES_LIST:
        models = enumerate_models_by_series(series)
        for m in models:
            _assert_eq(m.get("series"), series, f"{m.get('name')} 的 series 应为 {series}")

    # 不存在的系列 → 空
    _assert_eq(len(enumerate_models_by_series("不存在系列")), 0, "不存在系列→0 个")


# ================================================================
# 6. 端到端测试（需 Chroma + 本地 LLM）
# ================================================================

def _chat(query, sid):
    """模拟 webapp：收集流式回答 + 写 assistant 消息。"""
    from tools.agent import ask_stream
    from tools.context_store import append_message
    parts = []
    for chunk in ask_stream(query, sid):
        parts.append(chunk)
    answer = "".join(parts)
    append_message(sid, "assistant", answer)
    return answer


def test_e2e_aspect_query():
    _assert_in("7999", _chat("云顶 X2 多少钱", str(uuid.uuid4())), "多少钱→只答价格 7999")
    ans = _chat("净白 S1 吸力多大", str(uuid.uuid4()))
    _assert_in("3000Pa", ans, "吸力多大→含吸力 3000Pa")
    _assert_not_in("899", ans, "吸力多大→不含价格")
    _assert_in("2026-09-15", _chat("云顶 X2 什么时候发布的", str(uuid.uuid4())), "什么时候发布→含日期")


def test_e2e_series_query():
    ans = _chat("净白 S 系列有什么产品", str(uuid.uuid4()))
    for name in ("净白 S1", "净白 S2", "净白 S3"):
        _assert_in(name, ans, f"净白 S 系列含 {name}")
    _assert_not_in("净界 P", ans, "净白 S 系列不含净界 P 型号")

    # 关键回归：系列 + 推荐 不应进入选购 SOP（不应问预算）
    ans2 = _chat("净白 S 系列有什么产品推荐", str(uuid.uuid4()))
    _assert_in("净白 S1", ans2, "系列推荐枚举出型号")
    _assert_not_in("预算", ans2, "系列推荐不走 SOP（不问预算）")

    ans3 = _chat("天工 T 系列有哪些型号", str(uuid.uuid4()))
    _assert_in("天工 T1", ans3, "天工 T 系列含 T1")
    _assert_in("天工 T2 Ultra", ans3, "天工 T 系列含 T2 Ultra")


def test_e2e_series_consulting():
    # 系列咨询（无枚举意图）应走 RAG，不应走"有以下型号"直出
    ans = _chat("净白 S 系列怎么样", str(uuid.uuid4()))
    _assert_not_in("有以下型号", ans, "系列咨询不走枚举直出")
    _assert(len(ans.strip()) > 0, "系列咨询有回答（走 RAG）")


# ================================================================
# 主入口
# ================================================================

def run_unit():
    print("── 1. extract_model_aspect ──")
    test_extract_model_aspect()
    print("\n── 2. format_model_line(aspect) ──")
    test_format_model_line_aspect()
    print("\n── 3. extract_model_info.series ──")
    test_extract_model_info_series()
    print("\n── 4. extract_series ──")
    test_extract_series()


def run_db():
    print("\n── 5. enumerate_models_by_series（Chroma）──")
    test_enumerate_models_by_series()


def run_e2e():
    print("\n── 6. 端到端：属性查询 ──")
    test_e2e_aspect_query()
    print("\n── 7. 端到端：系列查询 ──")
    test_e2e_series_query()
    print("\n── 8. 端到端：系列咨询 ──")
    test_e2e_series_consulting()


def _summary():
    print()
    print("=" * 70)
    print(f"结果: {_passed}/{_total} 通过", end="")
    if _failed:
        print(f", {len(_failed)} 失败:")
        for name, reason in _failed:
            print(f"  FAIL {name}: {reason}")
    else:
        print()
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("结构化维度提取测试（aspect + series）")
    print("=" * 70)
    run_unit()
    if "--db" in sys.argv or "--e2e" in sys.argv:
        run_db()
    if "--e2e" in sys.argv:
        run_e2e()
    _summary()
    sys.exit(0 if _passed == _total else 1)
