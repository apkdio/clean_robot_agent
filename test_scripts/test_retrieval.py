"""知识库分域召回测试脚本（品牌化后：6 域 + 不染一尘知识库）。

测试覆盖：
  1. 域路由（agent._route_domain，真实代码）
  2. 选购咨询检测（sops.base.is_consulting，真实代码）
  3. 症状映射（config SYMPTOM_MAP，真实数据）
  4. 混合检索 + 域过滤（mock 检索器，品牌化文件名）
  5. 稠密检索 + 域过滤（mock 检索器）
  6. 组合过滤（域 + 预算）
  7. 故障 SOP 域定向检索
  8. 边界场景

运行：
  python test_retrieval.py
  python test_retrieval.py --e2e    # 含端到端检索（需 Chroma 有数据）
"""

import os
import re
import sys
from typing import Any, Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

# 真实代码（不再复制 mock 副本，测的就是线上逻辑）
from tools.agent import _route_domain
from sops.base import is_consulting
from config.word_dict_config import SYMPTOM_MAP


def _symptom_query(text: str) -> str | None:
    """规则匹配症状 → 标准 query（与 repair._extract_symptom 的规则部分对齐）。"""
    for kw, query in SYMPTOM_MAP.items():
        if kw in text:
            return query
    return None


# ──────────────────────────────────────────────────────────────────
# mock 检索数据：品牌化后的知识库文件（6 域，repair/maintain 共用维修文件）
# ──────────────────────────────────────────────────────────────────

_MOCK_CHUNKS: Dict[str, List[Dict[str, Any]]] = {
    "不染一尘品牌介绍.txt": [
        {"content": "不染一尘是专注极致清洁体验的高端扫拖机器人品牌，隶属于云境智能集团。", "metadata": {"file_name": "不染一尘品牌介绍.txt"}},
        {"content": "不染一尘有哪些产品系列？净白 S、净界 P、天工 T、云顶 X 四个系列。", "metadata": {"file_name": "不染一尘品牌介绍.txt"}},
    ],
    "不染一尘选购指南.txt": [
        {"content": "选购扫地机器人时，先定预算，再看户型面积和是否有宠物。", "metadata": {"file_name": "不染一尘选购指南.txt"}},
        {"content": "吸力参数：全系 3000Pa 到 13000Pa，小户型 4000Pa 足够。", "metadata": {"file_name": "不染一尘选购指南.txt"}},
        {"content": "预算 1500 以内入门级推荐净白 S 系列。", "metadata": {"file_name": "不染一尘选购指南.txt", "min_price": 899, "max_price": 1499}},
    ],
    "不染一尘常见维修问题.txt": [
        {"content": "故障现象：机器人不移动。检查电源是否接通，轮子是否被异物卡住。", "metadata": {"file_name": "不染一尘常见维修问题.txt"}},
        {"content": "故障现象：吸力下降。检查尘盒是否已满，滤网是否需要清洗。", "metadata": {"file_name": "不染一尘常见维修问题.txt"}},
        {"content": "故障现象：水箱漏水。检查水箱盖是否盖紧，密封圈是否老化。", "metadata": {"file_name": "不染一尘常见维修问题.txt"}},
        {"content": "边刷建议每 2-3 个月更换一次，刷毛变形需提前更换。", "metadata": {"file_name": "不染一尘常见维修问题.txt"}},
    ],
    "不染一尘具体型号.txt": [
        {"content": "1. **不染一尘净白 S1**\n   - 系列：净白 S\n   - 吸力：3000Pa｜导航：视觉导航｜避障：红外\n   - 参考价：899", "metadata": {"file_name": "不染一尘具体型号.txt", "min_price": 899, "max_price": 899}},
        {"content": "2. **不染一尘净界 P2**\n   - 系列：净界 P\n   - 吸力：5500Pa｜导航：LDS激光｜避障：结构光\n   - 参考价：2299", "metadata": {"file_name": "不染一尘具体型号.txt", "min_price": 2299, "max_price": 2299}},
        {"content": "3. **不染一尘云顶 X2**\n   - 系列：云顶 X\n   - 吸力：13500Pa｜导航：LDS激光｜避障：AI双摄\n   - 参考价：7999", "metadata": {"file_name": "不染一尘具体型号.txt", "min_price": 7999, "max_price": 7999}},
    ],
    "不染一尘售后服务.txt": [
        {"content": "整机保修 2 年，电机保修 5 年。", "metadata": {"file_name": "不染一尘售后服务.txt"}},
        {"content": "云顶 X 系列维修超 3 天可申请备用机。", "metadata": {"file_name": "不染一尘售后服务.txt"}},
    ],
}


def _resolve_file_name(filter: dict | None) -> str | None:
    """从 filter 里解析 file_name（支持直接值、$eq、$and）。"""
    if not filter:
        return None
    if "file_name" in filter:
        fn = filter["file_name"]
        if isinstance(fn, dict) and "$eq" in fn:
            return fn["$eq"]
        return fn
    if "$and" in filter:
        for cond in filter["$and"]:
            if isinstance(cond, dict) and "file_name" in cond:
                return cond["file_name"]
    return None


def _mock_dense_search(query: str, filter: dict | None = None, top_k: int = 10) -> List[Tuple[Dict, float]]:
    """模拟稠密检索：按域名过滤 + 逐字匹配（模拟 BM25 的字符级命中）。"""
    file_name = _resolve_file_name(filter)
    candidates = []
    for fname, chunks in _MOCK_CHUNKS.items():
        if file_name and fname != file_name:
            continue
        candidates.extend(chunks)

    query_chars = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", query))
    scored = []
    for c in candidates:
        content_chars = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", c["content"]))
        overlap = len(query_chars & content_chars)
        if overlap > 0:
            scored.append((c, overlap / max(len(query_chars), 1)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _mock_bm25_search(query: str, filter: dict | None = None, top_k: int = 10) -> List[Tuple[Dict, float]]:
    """模拟 BM25 稀疏检索：与 dense 相同逻辑。"""
    return _mock_dense_search(query, filter, top_k)


def _mock_hybrid_search(query: str, filter: dict | None = None) -> List[Dict]:
    """模拟混合检索：dense + sparse → RRF 融合。"""
    dense = _mock_dense_search(query, filter, top_k=10)
    sparse = _mock_bm25_search(query, filter, top_k=10)

    all_keys = {}
    for rank, (doc, _) in enumerate(dense, 1):
        key = doc["content"]
        all_keys.setdefault(key, {"doc": doc, "dense_rank": rank, "sparse_rank": 999})
    for rank, (doc, _) in enumerate(sparse, 1):
        key = doc["content"]
        if key in all_keys:
            all_keys[key]["sparse_rank"] = rank
        else:
            all_keys[key] = {"doc": doc, "dense_rank": 999, "sparse_rank": rank}

    scored = []
    for key, item in all_keys.items():
        rrf = 1.0 / (60 + item["dense_rank"]) + 1.0 / (60 + item["sparse_rank"])
        scored.append((item["doc"], rrf))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:5]]


# ──────────────────────────────────────────────────────────────────
# 断言辅助
# ──────────────────────────────────────────────────────────────────

_total = 0
_passed = 0
_failed: List[Tuple[str, str]] = []


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


# ================================================================
# 1. 域路由测试（真实 agent._route_domain，返回 file_name）
# ================================================================

def test_route_domain_brand():
    _assert_eq(_route_domain("为什么买不染一尘"), "不染一尘品牌介绍.txt", "为什么买→品牌域")
    _assert_eq(_route_domain("你们家有什么优势"), "不染一尘品牌介绍.txt", "优势→品牌域")


def test_route_domain_aftersales():
    _assert_eq(_route_domain("扫地机器人保修多久"), "不染一尘售后服务.txt", "保修→售后域")
    _assert_eq(_route_domain("零件掉了可以报修吗"), "不染一尘售后服务.txt", "报修→售后域")


def test_route_domain_consulting():
    _assert_eq(_route_domain("选购扫地机器人要注意什么"), "不染一尘选购指南.txt", "选购注意→选购域")
    _assert_eq(_route_domain("买扫地机器人有什么技巧"), "不染一尘选购指南.txt", "购买技巧→选购域")


def test_route_domain_repair_maintain():
    _assert_eq(_route_domain("机器人不动了怎么办"), "不染一尘常见维修问题.txt", "不动→维修域")
    _assert_eq(_route_domain("水箱漏水怎么处理"), "不染一尘常见维修问题.txt", "漏水→维修域")
    _assert_eq(_route_domain("边刷多久换一次"), "不染一尘常见维修问题.txt", "边刷→维护域（与维修同文件）")


def test_route_domain_none():
    _assert(_route_domain("扫地机器人对哪个牌子好") is None, "品牌对比→None（全库兜底）")
    _assert(_route_domain("今天天气怎么样") is None, "领域外→None")
    _assert(_route_domain("你好") is None, "闲聊→None")


# ================================================================
# 2. 选购咨询检测（真实 sops.base.is_consulting）
# ================================================================

def test_is_consulting_true():
    _assert_eq(is_consulting("选购扫地机器人要注意什么"), True, "选购+注意")
    _assert_eq(is_consulting("买扫地机器人有什么技巧"), True, "买+技巧")
    _assert_eq(is_consulting("入手扫地机器人有哪些坑"), True, "入手+坑")


def test_is_consulting_false():
    _assert(is_consulting("边刷多久换一次") is False, "纯维护→False")
    _assert(is_consulting("扫地机器人不走了怎么办") is False, "纯故障→False")
    _assert(is_consulting("你好") is False, "闲聊→False")
    _assert(is_consulting("选购一款扫地机器人") is False, "纯选购动作→False")


# ================================================================
# 3. 症状映射（真实 config.SYMPTOM_MAP）
# ================================================================

def test_symptom_map_hit():
    _assert_eq(_symptom_query("机器人不动了"), "机器人不移动怎么办", "不动")
    _assert_eq(_symptom_query("水箱漏水"), "水箱漏水怎么办", "漏水")
    _assert_eq(_symptom_query("有异响"), "扫地机器人异响怎么办", "异响")
    _assert_eq(_symptom_query("充不进电"), "机器人充不进电怎么办", "充不进电")
    _assert_eq(_symptom_query("找不到充电座"), "机器人找不到充电座怎么办", "找不到充电座")
    _assert_eq(_symptom_query("吸力变小了"), "吸力下降怎么办", "吸力")
    _assert_eq(_symptom_query("拖地后地面有水痕"), "拖地后地面有明显水痕", "水痕")


def test_symptom_map_miss():
    _assert(_symptom_query("机器人冒烟了") is None, "冒烟→None（危险现象走安全拦截）")
    _assert(_symptom_query("电池鼓包") is None, "电池鼓包→None")
    _assert(_symptom_query("你好") is None, "闲聊→None")


# ================================================================
# 4. 稠密检索 + 域过滤（mock）
# ================================================================

def test_dense_search_domain_filter():
    results = _mock_dense_search("吸力下降", filter={"file_name": "不染一尘常见维修问题.txt"})
    files = {r[0]["metadata"]["file_name"] for r in results}
    _assert_eq(len(files), 1, "只返回一个域的结果")
    _assert("不染一尘常见维修问题.txt" in files, "结果来自维修域")
    _assert(len(results) > 0, "有结果")


def test_dense_search_no_filter():
    results = _mock_dense_search("吸力")
    files = {r[0]["metadata"]["file_name"] for r in results}
    _assert(len(files) > 1, "无过滤时返回多个域的结果")
    _assert("不染一尘常见维修问题.txt" in files, "含维修域")
    _assert("不染一尘选购指南.txt" in files, "含选购指南域")


def test_dense_search_cross_domain():
    shopping = _mock_dense_search("吸力", filter={"file_name": "不染一尘选购指南.txt"})
    repair = _mock_dense_search("吸力", filter={"file_name": "不染一尘常见维修问题.txt"})
    _assert(len(shopping) > 0, "选购指南含吸力参数内容")
    _assert(len(repair) > 0, "维修域含吸力下降内容")


# ================================================================
# 5. 混合检索 + 域过滤（mock）
# ================================================================

def test_hybrid_search_domain_filter():
    results = _mock_hybrid_search("边刷更换", filter={"file_name": "不染一尘常见维修问题.txt"})
    files = {r["metadata"]["file_name"] for r in results}
    _assert_eq(len(files), 1, "只返回维修域")
    _assert("不染一尘常见维修问题.txt" in files, "结果来自维修域")
    _assert(len(results) > 0, "有结果")


def test_hybrid_search_domain_isolation():
    maintain = _mock_hybrid_search("边刷多久换一次", filter={"file_name": "不染一尘常见维修问题.txt"})
    _assert(len(maintain) > 0, "维修/维护域有结果")
    _assert_in("边刷", maintain[0]["content"], "top-1 含边刷内容")


# ================================================================
# 6. 组合过滤（域 + 预算，mock）
# ================================================================

def test_combined_domain_and_budget_filter():
    filter_dict = {"$and": [
        {"file_name": "不染一尘具体型号.txt"},
        {"min_price": {"$lte": 1000}},
    ]}
    results = _mock_dense_search("不染一尘", filter=filter_dict)
    files = {r[0]["metadata"]["file_name"] for r in results}
    _assert_eq(len(files), 1, "只返回具体型号域")
    _assert("不染一尘具体型号.txt" in files, "结果来自具体型号")


# ================================================================
# 7. 故障 SOP 域定向检索（mock）
# ================================================================

def test_repair_sop_retrieval():
    query = SYMPTOM_MAP["吸力"]  # → "吸力下降怎么办"
    results = _mock_hybrid_search(query, filter={"file_name": "不染一尘常见维修问题.txt"})
    _assert(len(results) > 0, "故障 SOP 在维修域找到了结果")
    _assert_in("吸力下降", results[0]["content"], "结果包含吸力下降内容")


def test_symptom_mapping_retrieval():
    for kw, std_query in SYMPTOM_MAP.items():
        results = _mock_hybrid_search(std_query, filter={"file_name": "不染一尘常见维修问题.txt"})
        _assert(len(results) > 0, f"症状'{kw}'→'{std_query}' 在维修域有结果")


# ================================================================
# 8. 边界场景（mock）
# ================================================================

def test_domain_filter_empty_query():
    results = _mock_dense_search("", filter={"file_name": "不染一尘常见维修问题.txt"})
    _assert_eq(len(results), 0, "空 query 无结果")


def test_domain_filter_nonexistent_domain():
    results = _mock_dense_search("吸力", filter={"file_name": "不存在的文件.txt"})
    _assert_eq(len(results), 0, "不存在域名返回空")


def test_domain_filter_special_chars():
    results = _mock_dense_search("吸力！！！", filter={"file_name": "不染一尘常见维修问题.txt"})
    _assert(len(results) > 0, "特殊字符不影响检索")


def test_hybrid_search_short_query():
    results = _mock_hybrid_search("异响", filter={"file_name": "不染一尘常见维修问题.txt"})
    _assert(len(results) > 0, "短 query 在维修域有结果")


# ================================================================
# 主入口
# ================================================================

def run_all():
    global _total, _passed, _failed
    _total = 0
    _passed = 0
    _failed = []

    print("=" * 70)
    print("知识库分域召回测试（品牌化后）")
    print("=" * 70)

    print("\n── 1. 域路由 ──")
    test_route_domain_brand()
    test_route_domain_aftersales()
    test_route_domain_consulting()
    test_route_domain_repair_maintain()
    test_route_domain_none()

    print("\n── 2. 选购咨询检测 ──")
    test_is_consulting_true()
    test_is_consulting_false()

    print("\n── 3. 症状映射 ──")
    test_symptom_map_hit()
    test_symptom_map_miss()

    print("\n── 4. 稠密检索 + 域过滤 ──")
    test_dense_search_domain_filter()
    test_dense_search_no_filter()
    test_dense_search_cross_domain()

    print("\n── 5. 混合检索 + 域过滤 ──")
    test_hybrid_search_domain_filter()
    test_hybrid_search_domain_isolation()

    print("\n── 6. 组合过滤 ──")
    test_combined_domain_and_budget_filter()

    print("\n── 7. 故障 SOP 域定向 ──")
    test_repair_sop_retrieval()
    test_symptom_mapping_retrieval()

    print("\n── 8. 边界场景 ──")
    test_domain_filter_empty_query()
    test_domain_filter_nonexistent_domain()
    test_domain_filter_special_chars()
    test_hybrid_search_short_query()

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
    return _passed, _total


def run_e2e():
    """端到端测试：需要 Chroma 有数据。"""
    print("\n" + "=" * 70)
    print("端到端检索测试（需 Chroma + 知识库已入库）")
    print("=" * 70)

    try:
        from tools.vector_store import list_collections_info, DenseRetriever
        from tools.hybrid_retriever import HybridRetriever

        info = list_collections_info()
        if info.get("chunk_count", 0) == 0:
            print("  SKIP 向量库为空，跳过端到端测试")
            return

        hr = HybridRetriever()
        hr.ensure_sparse_index()

        for label, fname in [
            ("品牌介绍", "不染一尘品牌介绍.txt"),
            ("选购指南", "不染一尘选购指南.txt"),
            ("常见维修", "不染一尘常见维修问题.txt"),
            ("售后服务", "不染一尘售后服务.txt"),
        ]:
            chunks = hr.search("不染一尘", filter={"file_name": fname})
            print(f"  [{label}] 域过滤 '不染一尘' → {len(chunks)} 条结果")

        combined = hr.search("扫地机器人", filter={"$and": [
            {"file_name": "不染一尘具体型号.txt"},
            {"min_price": {"$lte": 1000}},
        ]})
        print(f"  [组合] 具体型号 + 预算1000以内 → {len(combined)} 条结果")

    except Exception as e:
        print(f"  SKIP 端到端测试失败: {e}")


if __name__ == "__main__":
    passed, total = run_all()
    if "--e2e" in sys.argv:
        run_e2e()
    exit_code = 0 if passed == total else 1
    sys.exit(exit_code)
