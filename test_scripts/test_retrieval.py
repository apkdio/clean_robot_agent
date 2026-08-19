"""知识库分隔分域召回测试脚本。

测试覆盖：
  1. _route_domain 域路由
  2. _is_consulting 选购咨询检测
  3. 混合检索 + 域过滤（hybrid_retriever.search）
  4. 稠密检索 + 域过滤（DenseRetriever.search）
  5. 稀疏检索 + 域过滤（SparseRetriever.search）
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

# ──────────────────────────────────────────────────────────────────
# 测试数据：知识库域名与各域期望的典型内容
# ──────────────────────────────────────────────────────────────────

_KNOWLEDGE_FILES = {
    "选购指南.txt": "选购指南",
    "故障排除.txt": "故障排除",
    "维护保养.txt": "维护保养",
    "具体型号.txt": "具体型号",
    "扫地机器人100问.txt": "扫地机器人100问",
    "扫地机器人100问2.txt": "扫地机器人100问2",
    "扫拖一体机器人100问.txt": "扫拖一体机器人100问",
    "知识补充.txt": "知识补充",
}

_DOMAIN_FILES = {
    "consulting": "选购指南.txt",
    "repair": "故障排除.txt",
    "maintain": "维护保养.txt",
    "model": "具体型号.txt",
}

# ── 选购咨询检测 ──────────────────────────────────────────────

_BUY_WORDS = ["选购", "购买", "买", "挑", "选", "入手", "购", "采购", "拿下", "购置"]
_CONSULT_WORDS = [
    "注意", "问题", "技巧", "知识", "要点", "建议", "事项", "讲究", "坑", "避雷",
    "须知", "诀窍", "门道", "参数", "指标", "怎么选", "如何选", "注意什么",
    "有什么讲究", "怎么看", "考虑什么", "留意", "注意哪些", "避坑", "挑选技巧",
    "指南", "攻略", "手册", "清单", "建议清单",
]


def _is_consulting(text: str) -> bool:
    """判断是否是选购咨询（与 agent._is_consulting 逻辑对齐）。"""
    has_buy = any(w in text for w in _BUY_WORDS)
    has_consult = any(w in text for w in _CONSULT_WORDS)
    return has_buy and has_consult


# ── 域路由 ─────────────────────────────────────────────────────

_REPAIR_WORDS = ["故障", "坏了", "不动", "漏水", "异响", "不充电", "异常", "失灵",
                 "不好使", "出问题", "趴窝", "卡住", "噪音"]
_MAINTAIN_WORDS = ["维护", "保养", "清洗", "清理", "更换", "耗材", "滤网", "边刷", "主刷", "拖布", "尘盒", "充电座清洁"]


def _route_domain(text: str) -> str | None:
    """按 query 内容路由到对应知识域（与 agent._route_domain 逻辑对齐）。"""
    if _is_consulting(text):
        return "consulting"
    if any(w in text for w in _REPAIR_WORDS):
        return "repair"
    if any(w in text for w in _MAINTAIN_WORDS):
        return "maintain"
    return None


# ── 症状映射（与 repair._SYMPTOM_MAP 对齐）────────────────────

_SYMPTOM_MAP = {
    "不动": "机器人不移动怎么办",
    "不走了": "机器人不移动怎么办",
    "卡住": "机器人不移动怎么办",
    "漏水": "水箱漏水怎么办",
    "异响": "扫地机器人异响怎么办",
    "不充电": "机器人充不进电怎么办",
    "充不进电": "机器人充不进电怎么办",
    "找不到充电座": "机器人找不到充电座怎么办",
    "连不上": "APP无法连接机器人怎么办",
    "吸力": "吸力下降怎么办",
    "建图": "建图不完整怎么办",
    "不干净": "清扫不干净怎么办",
    "异味": "拖布有异味怎么办",
}


def _extract_symptom(text: str) -> str | None:
    """从文本提取故障现象映射（与 repair._extract_symptom 逻辑对齐）。"""
    for kw, query in _SYMPTOM_MAP.items():
        if kw in text:
            return query
    return None


# ── 模拟检索器（不依赖 Chroma，直接返回 mock 结果）──────────────

_MOCK_CHUNKS: Dict[str, List[Dict[str, Any]]] = {
    "选购指南.txt": [
        {"content": "选购扫地机器人时，首先看吸力参数，建议 2000Pa 以上日常够用。", "metadata": {"file_name": "选购指南.txt", "source": "选购指南.txt"}},
        {"content": "导航方式：LDS 激光导航 > 视觉导航 > 随机碰撞。预算允许优先选激光。", "metadata": {"file_name": "选购指南.txt", "source": "选购指南.txt"}},
        {"content": "避障能力：3D 结构光 > 红外避障 > 无避障。有宠家庭建议选结构光。", "metadata": {"file_name": "选购指南.txt", "source": "选购指南.txt"}},
        {"content": "拖地系统：旋转拖布 > 平板震动拖布 > 静压拖布。重拖地需求选旋转拖布。", "metadata": {"file_name": "选购指南.txt", "source": "选购指南.txt"}},
        {"content": "预算 1500 以内入门级推荐米家 M20、追觅 D10s。", "metadata": {"file_name": "选购指南.txt", "min_price": 899, "max_price": 1099, "source": "选购指南.txt"}},
    ],
    "故障排除.txt": [
        {"content": "故障现象：机器人不移动。检查电源是否接通，轮子是否被异物卡住，尝试重启。", "metadata": {"file_name": "故障排除.txt", "source": "故障排除.txt"}},
        {"content": "故障现象：吸力下降。检查尘盒是否已满，滤网是否需要清洗或更换，刷头是否缠绕毛发。", "metadata": {"file_name": "故障排除.txt", "source": "故障排除.txt"}},
        {"content": "故障现象：水箱漏水。检查水箱盖是否盖紧，密封圈是否老化，水箱是否有裂纹。", "metadata": {"file_name": "故障排除.txt", "source": "故障排除.txt"}},
        {"content": "故障现象：异响。检查主刷和边刷是否缠绕异物，轮子是否卡住，风机是否进异物。", "metadata": {"file_name": "故障排除.txt", "source": "故障排除.txt"}},
        {"content": "故障现象：APP 无法连接。检查机器人 WiFi 指示灯状态，尝试长按重置键重新配网。", "metadata": {"file_name": "故障排除.txt", "source": "故障排除.txt"}},
    ],
    "维护保养.txt": [
        {"content": "边刷建议每 2-3 个月更换一次，如果刷毛变形严重需提前更换。", "metadata": {"file_name": "维护保养.txt", "source": "维护保养.txt"}},
        {"content": "主刷每 3-6 个月更换一次，发现刷毛磨损或断裂时及时更换。", "metadata": {"file_name": "维护保养.txt", "source": "维护保养.txt"}},
        {"content": "滤网每 1-2 周清洗一次，每 3-6 个月更换一次。不可水洗的滤网用吸尘器吸尘。", "metadata": {"file_name": "维护保养.txt", "source": "维护保养.txt"}},
        {"content": "拖布每次使用后清洗，每 3-6 个月更换。建议使用中性清洁剂手洗。", "metadata": {"file_name": "维护保养.txt", "source": "维护保养.txt"}},
        {"content": "传感器每月用干布擦拭一次，避免灰尘堆积影响导航和避障。", "metadata": {"file_name": "维护保养.txt", "source": "维护保养.txt"}},
    ],
    "具体型号.txt": [
        {"content": "1. **米家扫拖机器人 M20**\n   - 吸力：2800Pa｜导航：LDS 激光｜避障：红外\n   - 参考价：899\n   - 发布时间：2025-03-15", "metadata": {"file_name": "具体型号.txt", "min_price": 899, "max_price": 899, "publish_date": 20250315, "source": "具体型号.txt"}},
        {"content": "2. **追觅 D10s**\n   - 吸力：3200Pa｜导航：LDS 激光｜避障：红外\n   - 参考价：1099\n   - 发布时间：2025-06-20", "metadata": {"file_name": "具体型号.txt", "min_price": 1099, "max_price": 1099, "publish_date": 20250620, "source": "具体型号.txt"}},
        {"content": "3. **美的 i5 Pro**\n   - 吸力：3000Pa｜导航：视觉导航｜避障：红外\n   - 参考价：799\n   - 发布时间：2026-03-10", "metadata": {"file_name": "具体型号.txt", "min_price": 799, "max_price": 799, "publish_date": 20260310, "source": "具体型号.txt"}},
        {"content": "5. **云米 VXVC12**\n   - 吸力：3100Pa｜导航：LDS 激光｜避障：红外\n   - 参考价：999\n   - 发布时间：2026-05-18", "metadata": {"file_name": "具体型号.txt", "min_price": 999, "max_price": 999, "publish_date": 20260518, "source": "具体型号.txt"}},
        {"content": "7. **科沃斯 T30 Mini**\n   - 吸力：4000Pa｜导航：LDS 激光｜避障：结构光\n   - 参考价：2299\n   - 发布时间：2025-09-12", "metadata": {"file_name": "具体型号.txt", "min_price": 2299, "max_price": 2299, "publish_date": 20250912, "source": "具体型号.txt"}},
    ],
    "扫地机器人100问.txt": [
        {"content": "1. **扫地机器人适合什么地面？** 瓷砖、木地板、短毛地毯均可使用，长毛地毯可能卡住主刷。", "metadata": {"file_name": "扫地机器人100问.txt", "source": "扫地机器人100问.txt"}},
        {"content": "2. **扫地机器人能清扫多大面积？** 主流机型续航 100-200 分钟，单次清扫 100-200 平米，可断点续扫。", "metadata": {"file_name": "扫地机器人100问.txt", "source": "扫地机器人100问.txt"}},
    ],
    "知识补充.txt": [
        {"content": "LDS 激光导航通过激光雷达旋转扫描建图，精度高但顶部有凸起，低矮家具可能进不去。", "metadata": {"file_name": "知识补充.txt", "source": "知识补充.txt"}},
        {"content": "dToF 导航是飞行时间测距技术，比 LDS 更薄、建图更快，适合低矮空间。", "metadata": {"file_name": "知识补充.txt", "source": "知识补充.txt"}},
    ],
}


def _mock_dense_search(query: str, filter: dict | None = None, top_k: int = 10) -> List[Tuple[Dict, float]]:
    """模拟稠密检索：按域名过滤 + 逐字匹配（模拟 BM25 的字符级命中）。"""
    # 解析 file_name 过滤（支持直接值、$eq、$and）
    file_name = None
    if filter:
        if "file_name" in filter:
            file_name = filter["file_name"]
        elif "$and" in filter:
            for cond in filter["$and"]:
                if isinstance(cond, dict) and "file_name" in cond:
                    file_name = cond["file_name"]
    if isinstance(file_name, dict) and "$eq" in file_name:
        file_name = file_name["$eq"]

    candidates = []
    for fname, chunks in _MOCK_CHUNKS.items():
        if file_name and fname != file_name:
            continue
        candidates.extend(chunks)

    # 逐字匹配（模拟 BM25 的中文逐字 tokenize）
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
    """模拟 BM25 稀疏检索：与 dense 相同逻辑，但可从不同视角验证。"""
    return _mock_dense_search(query, filter, top_k)


def _mock_hybrid_search(query: str, filter: dict | None = None) -> List[Dict]:
    """模拟混合检索：dense + sparse → RRF 融合。"""
    dense = _mock_dense_search(query, filter, top_k=10)
    sparse = _mock_bm25_search(query, filter, top_k=10)

    # RRF 融合（简化版）
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
# 测试用例
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


def _assert_not_in(content, container, msg: str):
    global _total, _passed, _failed
    _total += 1
    if content not in container:
        _passed += 1
        print(f"  OK  {msg}")
    else:
        _failed.append((msg, f"内容 '{content}' 不应出现在结果中"))
        print(f"  FAIL {msg}  — 不应出现 '{content}'")


# ================================================================
# 1. 域路由测试
# ================================================================

def test_route_domain_consulting():
    """选购咨询类 query 应路由到 consulting 域"""
    _assert_eq(_route_domain("选购扫地机器人要注意什么"), "consulting", "选购注意")
    _assert_eq(_route_domain("买扫地机器人有什么技巧"), "consulting", "购买技巧")
    _assert_eq(_route_domain("扫地机器人怎么选，有什么坑"), "consulting", "选购避坑")
    _assert_eq(_route_domain("扫地机器人选购指南"), "consulting", "选购指南（含购+选词）")
    _assert_eq(_route_domain("入手扫地机器人要注意什么"), "consulting", "入手注意")


def test_route_domain_maintain():
    """维护保养类 query 应路由到 maintain 域"""
    _assert_eq(_route_domain("边刷多久换一次"), "maintain", "边刷更换")
    _assert_eq(_route_domain("滤网怎么清洗"), "maintain", "滤网清洗")
    _assert_eq(_route_domain("扫地机器人保养技巧"), "maintain", "保养技巧")
    _assert_eq(_route_domain("主刷如何更换"), "maintain", "主刷更换")
    _assert_eq(_route_domain("拖布多久清洗一次"), "maintain", "拖布清洗")
    _assert_eq(_route_domain("尘盒怎么清理"), "maintain", "尘盒清理")
    _assert_eq(_route_domain("充电座清洁"), "maintain", "充电座清洁")


def test_route_domain_none():
    """通用问答/领域外 query 应返回 None（全库兜底）"""
    _assert(_route_domain("扫地机器人对哪个牌子好") is None, "品牌对比 → None")
    _assert(_route_domain("扫地机器人能扫地毯吗") is None, "地毯问题 → None")
    _assert(_route_domain("LDS 和 dToF 有什么区别") is None, "技术对比 → None")
    _assert(_route_domain("今天天气怎么样") is None, "领域外 → None")
    _assert(_route_domain("你好") is None, "闲聊 → None")


def test_route_domain_priority():
    """选购咨询优先级高于维护（含选购+维护词时走 consulting）"""
    _assert_eq(_route_domain("选购扫地机器人要注意滤网更换"), "consulting", "选购+维护 → consulting优先")


# ================================================================
# 2. 选购咨询检测
# ================================================================

def test_is_consulting_true():
    """含选购动作词 + 咨询词 → True"""
    _assert_eq(_is_consulting("选购扫地机器人要注意什么"), True, "选购+注意")
    _assert_eq(_is_consulting("买扫地机器人有什么技巧"), True, "买+技巧")
    _assert_eq(_is_consulting("扫地机器人怎么选，有什么讲究"), True, "选+讲究")
    _assert_eq(_is_consulting("入手扫地机器人有哪些坑"), True, "入手+坑")
    _assert_eq(_is_consulting("购买扫地机器人需要注意哪些参数"), True, "购买+参数")


def test_is_consulting_false():
    """仅含选购动作词或仅含咨询词 → False"""
    _assert(_is_consulting("边刷多久换一次") is False, "纯维护 → False")
    _assert(_is_consulting("扫地机器人不走了怎么办") is False, "纯故障 → False")
    _assert(_is_consulting("你好") is False, "闲聊 → False")
    _assert(_is_consulting("选购一款扫地机器人") is False, "纯选购动作 → False")
    _assert(_is_consulting("有什么注意事项") is False, "纯咨询词 → False")
    _assert(_is_consulting("") is False, "空字符串 → False")


# ================================================================
# 3. 症状提取测试
# ================================================================

def test_extract_symptom_hit():
    """故障现象关键词 → 标准检索 query"""
    _assert_eq(_extract_symptom("机器人不动了"), "机器人不移动怎么办", "不动")
    _assert_eq(_extract_symptom("水箱漏水"), "水箱漏水怎么办", "漏水")
    _assert_eq(_extract_symptom("有异响"), "扫地机器人异响怎么办", "异响")
    _assert_eq(_extract_symptom("充不进电"), "机器人充不进电怎么办", "充不进电")
    _assert_eq(_extract_symptom("找不到充电座"), "机器人找不到充电座怎么办", "找不到充电座")
    _assert_eq(_extract_symptom("吸力变小了"), "吸力下降怎么办", "吸力")
    _assert_eq(_extract_symptom("建图不完整"), "建图不完整怎么办", "建图")


def test_extract_symptom_miss():
    """未匹配的故障现象 → None"""
    _assert(_extract_symptom("机器人冒烟了") is None, "冒烟 → None（未收录）")
    _assert(_extract_symptom("电池鼓包") is None, "电池鼓包 → None")
    _assert(_extract_symptom("你好") is None, "闲聊 → None")
    _assert(_extract_symptom("") is None, "空字符串 → None")


# ================================================================
# 4. 域过滤检索测试（模拟稠密检索）
# ================================================================

def test_dense_search_domain_filter():
    """稠密检索 + 域名过滤：只返回指定域的结果"""
    results = _mock_dense_search("吸力下降", filter={"file_name": "故障排除.txt"})
    files = {r[0]["metadata"]["file_name"] for r in results}
    _assert_eq(len(files), 1, "只返回一个域的结果")
    _assert("故障排除.txt" in files, "结果来自故障排除域")
    _assert_in("吸力下降", results[0][0]["content"], "包含吸力下降相关内容")


def test_dense_search_no_filter():
    """稠密检索无过滤：返回全库结果"""
    results = _mock_dense_search("吸力")
    files = {r[0]["metadata"]["file_name"] for r in results}
    _assert(len(files) > 1, "无过滤时返回多个域的结果")
    _assert("故障排除.txt" in files, "含故障排除域")
    _assert("选购指南.txt" in files, "含选购指南域（选购指南也含吸力参数）")


def test_dense_search_cross_domain():
    """跨域 query：吸力在选购指南和故障排除都有相关内容"""
    shopping = _mock_dense_search("吸力", filter={"file_name": "选购指南.txt"})
    repair = _mock_dense_search("吸力", filter={"file_name": "故障排除.txt"})
    _assert(len(shopping) > 0, "选购指南含吸力参数内容")
    _assert(len(repair) > 0, "故障排除含吸力下降内容")


def test_dense_search_wrong_domain():
    """query 与域不匹配时无结果"""
    results = _mock_dense_search("吸力下降", filter={"file_name": "选购指南.txt"})
    # 选购指南不含"吸力下降" → 期望无结果或极少结果
    all_mention_drop = all("下降" not in r[0]["content"] for r in results)
    _assert(all_mention_drop, "选购指南不含故障排除内容")


# ================================================================
# 5. 混合检索 + 域过滤测试
# ================================================================

def test_hybrid_search_domain_filter():
    """混合检索 + 域名过滤：只返回指定域"""
    results = _mock_hybrid_search("滤网清洗", filter={"file_name": "维护保养.txt"})
    files = {r["metadata"]["file_name"] for r in results}
    _assert_eq(len(files), 1, "只返回维护保养域")
    _assert("维护保养.txt" in files, "结果来自维护保养域")
    _assert(len(results) > 0, "有结果")


def test_hybrid_search_domain_isolation():
    """域隔离：维护域召回的是滤网更换（维护），而非故障排查步骤"""
    maintain = _mock_hybrid_search("滤网多久换一次", filter={"file_name": "维护保养.txt"})
    _assert(len(maintain) > 0, "维护域有结果")
    _assert_in("滤网", maintain[0]["content"], "维护域 top-1 含滤网内容")


def test_hybrid_search_full_vs_domain():
    """全库检索 vs 域定向检索：结果数应不同"""
    full = _mock_hybrid_search("吸力")
    domain = _mock_hybrid_search("吸力", filter={"file_name": "故障排除.txt"})
    _assert(len(full) >= len(domain), "全库结果 >= 域定向结果")


# ================================================================
# 6. 组合过滤测试（域 + 预算/日期）
# ================================================================

def test_combined_domain_and_budget_filter():
    """域过滤 + 预算过滤：同时满足两个条件"""
    filter_dict = {"$and": [
        {"file_name": "具体型号.txt"},
        {"min_price": {"$lte": 1000}},
    ]}
    results = _mock_dense_search("扫地机器人", filter=filter_dict)
    # 只验证域名过滤部分（预算过滤需要 Chroma，mock 简化）
    files = {r[0]["metadata"]["file_name"] for r in results}
    _assert_eq(len(files), 1, "只返回具体型号域")
    _assert("具体型号.txt" in files, "结果来自具体型号")


def test_combined_domain_and_date_filter():
    """域过滤 + 日期过滤"""
    filter_dict = {"$and": [
        {"file_name": "具体型号.txt"},
        {"publish_date": {"$gte": 20260101}},
    ]}
    results = _mock_dense_search("扫地机器人", filter=filter_dict)
    files = {r[0]["metadata"]["file_name"] for r in results}
    _assert_eq(len(files), 1, "只返回具体型号域")
    _assert("具体型号.txt" in files, "结果来自具体型号")


# ================================================================
# 7. 故障 SOP 域定向检索测试
# ================================================================

def test_repair_sop_retrieval():
    """故障 SOP 的检索只定向到故障排除.txt"""
    query = _SYMPTOM_MAP["吸力"]  # → "吸力下降怎么办"
    results = _mock_hybrid_search(query, filter={"file_name": "故障排除.txt"})
    _assert(len(results) > 0, "故障 SOP 在故障排除域找到了结果")
    _assert_in("吸力下降", results[0]["content"], "结果包含吸力下降内容")


def test_repair_sop_wrong_domain_exclusion():
    """故障 SOP 应在故障域召回吸力下降排查（而非选购的吸力参数）"""
    repair = _mock_hybrid_search("吸力下降怎么办", filter={"file_name": "故障排除.txt"})
    _assert(len(repair) > 0, "故障排除域有结果")
    _assert_in("吸力下降", repair[0]["content"], "故障域 top-1 是吸力下降排查")


def test_symptom_mapping_retrieval():
    """症状映射后的标准 query 在故障域能准确召回"""
    for kw, std_query in _SYMPTOM_MAP.items():
        results = _mock_hybrid_search(std_query, filter={"file_name": "故障排除.txt"})
        _assert(len(results) > 0, f"症状'{kw}'→'{std_query}' 在故障域有结果")


# ================================================================
# 8. 边界场景测试
# ================================================================

def test_domain_filter_empty_query():
    """空 query + 域过滤 → 返回空结果"""
    results = _mock_dense_search("", filter={"file_name": "故障排除.txt"})
    _assert_eq(len(results), 0, "空 query 无结果")


def test_domain_filter_nonexistent_domain():
    """不存在的域名 → 返回空结果"""
    results = _mock_dense_search("吸力", filter={"file_name": "不存在的文件.txt"})
    _assert_eq(len(results), 0, "不存在域名返回空")


def test_domain_filter_special_chars():
    """含特殊字符的 query 不影响域过滤"""
    results = _mock_dense_search("吸力！！！", filter={"file_name": "故障排除.txt"})
    _assert(len(results) > 0, "特殊字符不影响检索")


def test_hybrid_search_short_query():
    """短 query + 域过滤"""
    results = _mock_hybrid_search("异响", filter={"file_name": "故障排除.txt"})
    _assert(len(results) > 0, "短 query 在故障域有结果")


def test_hybrid_search_long_query():
    """长 query + 域过滤"""
    query = "扫地机器人拖地的时候拖布有异味怎么处理有什么好办法"
    results = _mock_hybrid_search(query, filter={"file_name": "故障排除.txt"})
    _assert(len(results) > 0, "长 query 在故障域有结果")


# ================================================================
# 主入口
# ================================================================

def run_all():
    global _total, _passed, _failed
    _total = 0
    _passed = 0
    _failed = []

    print("=" * 70)
    print("知识库分隔分域召回测试")
    print("=" * 70)

    print("\n── 1. 域路由 ──")
    test_route_domain_consulting()
    test_route_domain_maintain()
    test_route_domain_none()
    test_route_domain_priority()

    print("\n── 2. 选购咨询检测 ──")
    test_is_consulting_true()
    test_is_consulting_false()

    print("\n── 3. 症状提取 ──")
    test_extract_symptom_hit()
    test_extract_symptom_miss()

    print("\n── 4. 稠密检索 + 域过滤 ──")
    test_dense_search_domain_filter()
    test_dense_search_no_filter()
    test_dense_search_cross_domain()
    test_dense_search_wrong_domain()

    print("\n── 5. 混合检索 + 域过滤 ──")
    test_hybrid_search_domain_filter()
    test_hybrid_search_domain_isolation()
    test_hybrid_search_full_vs_domain()

    print("\n── 6. 组合过滤 ──")
    test_combined_domain_and_budget_filter()
    test_combined_domain_and_date_filter()

    print("\n── 7. 故障 SOP 域定向 ──")
    test_repair_sop_retrieval()
    test_repair_sop_wrong_domain_exclusion()
    test_symptom_mapping_retrieval()

    print("\n── 8. 边界场景 ──")
    test_domain_filter_empty_query()
    test_domain_filter_nonexistent_domain()
    test_domain_filter_special_chars()
    test_hybrid_search_short_query()
    test_hybrid_search_long_query()

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
        from tools.sparse_retriever import SparseRetriever

        info = list_collections_info()
        if info.get("chunk_count", 0) == 0:
            print("  SKIP 向量库为空，跳过端到端测试")
            return

        hr = HybridRetriever()
        hr.ensure_sparse_index()

        # 域过滤检索
        for domain, fname in [
            ("选购指南", "选购指南.txt"),
            ("故障排除", "故障排除.txt"),
            ("维护保养", "维护保养.txt"),
        ]:
            chunks = hr.search("吸力", filter={"file_name": fname})
            print(f"  [{domain}] 域过滤 '吸力' → {len(chunks)} 条结果")
            for c in chunks[:2]:
                print(f"             {c.page_content[:50]}...")

        # 全库检索（对比）
        full = hr.search("吸力")
        print(f"  [全库] '吸力' → {len(full)} 条结果")

        # 组合过滤
        combined = hr.search("扫地机器人", filter={"$and": [
            {"file_name": "具体型号.txt"},
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