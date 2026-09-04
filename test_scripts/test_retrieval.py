"""检索链路测试：域路由 + 词表 + BM25 分词 + 过滤 + RRF 融合 + 条目切分。

覆盖（全部测真实代码，不再复制 mock 副本）：
  - agent._route_domain（知识域 → file_name）
  - sops.base.is_consulting / is_aftersales / is_brand
  - config.word_dict_config 词表完整性（DOMAIN_MAP / SYMPTOM_MAP / SYMPTOM_QUERY_MAP）
  - SparseRetriever._tokenize（中英混合分词）
  - sparse_retriever._matches_filter（Chroma where 内存过滤）
  - rrf_fusion.reciprocal_rank_fusion（稠密+稀疏融合排序）
  - entry_splitter.split_numbered_entries（编号条目 + 章节前缀）
  - [--e2e] HybridRetriever 域过滤端到端（需 Ollama + Chroma）

运行：
  .venv\\Scripts\\python.exe test_retrieval.py [--e2e]
"""
import sys
from _runner import *
from langchain_core.documents import Document


# ──────────────────────────────────────────────────────────────
# 1. 域路由（真实 agent._route_domain）
# ──────────────────────────────────────────────────────────────

def test_route_domain():
    from tools.agent import _route_domain
    # 品牌域（优先于“买”误判）
    _assert_eq(_route_domain("为什么买不染一尘"), "不染一尘品牌介绍.txt", "为什么买→品牌域")
    _assert_eq(_route_domain("你们家有什么优势"), "不染一尘品牌介绍.txt", "优势→品牌域")
    # 售后域（优先于“修”误判）
    _assert_eq(_route_domain("扫地机器人保修多久"), "不染一尘售后服务.txt", "保修→售后域")
    _assert_eq(_route_domain("零件掉了可以报修吗"), "不染一尘售后服务.txt", "报修→售后域")
    # 选购咨询域
    _assert_eq(_route_domain("选购扫地机器人要注意什么"), "不染一尘选购指南.txt", "选购注意→选购域")
    _assert_eq(_route_domain("买扫地机器人有什么技巧"), "不染一尘选购指南.txt", "购买技巧→选购域")
    # 故障域
    _assert_eq(_route_domain("机器人不动了怎么办"), "不染一尘常见维修问题.txt", "不动→维修域")
    _assert_eq(_route_domain("水箱漏水怎么处理"), "不染一尘常见维修问题.txt", "漏水→维修域")
    # 维护域（与维修同文件）
    _assert_eq(_route_domain("边刷多久换一次"), "不染一尘常见维修问题.txt", "边刷→维护域")
    _assert_eq(_route_domain("尘盒怎么清洗"), "不染一尘常见维修问题.txt", "尘盒→维护域")
    # 兜底 None
    _assert(_route_domain("扫地机器人对哪个牌子好") is None, "品牌对比→None")
    _assert(_route_domain("今天天气怎么样") is None, "领域外→None")
    _assert(_route_domain("你好") is None, "闲聊→None")


# ──────────────────────────────────────────────────────────────
# 2. 选购咨询 / 售后 / 品牌判定
# ──────────────────────────────────────────────────────────────

def test_is_consulting():
    from sops.base import is_consulting
    _assert(is_consulting("选购扫地机器人要注意什么"), "选购+注意→咨询")
    _assert(is_consulting("买扫地机器人有什么技巧"), "买+技巧→咨询")
    _assert(is_consulting("入手扫地机器人有哪些坑"), "入手+坑→咨询")
    _assert(not is_consulting("边刷多久换一次"), "纯维护→非咨询")
    _assert(not is_consulting("扫地机器人不走了怎么办"), "纯故障→非咨询")
    _assert(not is_consulting("选购一款扫地机器人"), "纯选购动作→非咨询")
    _assert(not is_consulting("你好"), "闲聊→非咨询")


def test_is_aftersales():
    from sops.base import is_aftersales
    _assert(is_aftersales("扫地机器人保修多久"), "保修→售后")
    _assert(is_aftersales("可以退货吗"), "退货→售后")
    _assert(not is_aftersales("帮我推荐一款"), "推荐→非售后")
    _assert(not is_aftersales("边刷多久换一次"), "维护→非售后")


def test_is_brand():
    from sops.base import is_brand
    _assert(is_brand("为什么买不染一尘"), "为什么买→品牌")
    _assert(is_brand("不染一尘是什么品牌"), "品牌→品牌")
    _assert(not is_brand("帮我推荐一款"), "推荐→非品牌")
    _assert(not is_brand("边刷多久换一次"), "维护→非品牌")


# ──────────────────────────────────────────────────────────────
# 3. 词表完整性
# ──────────────────────────────────────────────────────────────

def test_word_dict_integrity():
    from config.word_dict_config import DOMAIN_MAP, SYMPTOM_MAP, SYMPTOM_QUERY_MAP
    _assert_eq(
        set(DOMAIN_MAP.keys()),
        {"brand", "consulting", "model", "repair", "aftersales", "maintain"},
        "DOMAIN_MAP 六个域",
    )
    _assert_eq(DOMAIN_MAP["maintain"], DOMAIN_MAP["repair"], "维护与维修同文件")
    _assert(len(SYMPTOM_MAP) > 0, "SYMPTOM_MAP 非空")
    _assert(len(SYMPTOM_QUERY_MAP) > 0, "SYMPTOM_QUERY_MAP 非空")
    map_values = set(SYMPTOM_MAP.values())
    for sid, q in SYMPTOM_QUERY_MAP.items():
        _assert(q in map_values, f"SYMPTOM_QUERY_MAP[{sid}] 出现在 SYMPTOM_MAP 值中")
    # 危险现象不应进入症状映射（由 agent 前置拦截）
    for danger in ("冒烟", "鼓包", "着火"):
        _assert(danger not in SYMPTOM_MAP, f"危险词 {danger} 不在 SYMPTOM_MAP")


# ──────────────────────────────────────────────────────────────
# 4. BM25 分词（真实 SparseRetriever._tokenize）
# ──────────────────────────────────────────────────────────────

def test_tokenize():
    from tools.sparse_retriever import SparseRetriever
    t = SparseRetriever._tokenize("Python 在机器学习中的应用")
    _assert_eq(t, ["python", "在", "机", "器", "学", "习", "中", "的", "应", "用"], "中英混合分词")
    t2 = SparseRetriever._tokenize("扫地 robot100")
    _assert_in("扫", t2, "中文单字 token")
    _assert_in("robot100", t2, "数字英文保留为整 token")
    _assert_eq(SparseRetriever._tokenize("，。！？"), [], "纯标点→空")


# ──────────────────────────────────────────────────────────────
# 5. Chroma where 内存过滤（真实 _matches_filter）
# ──────────────────────────────────────────────────────────────

def test_matches_filter():
    from tools.sparse_retriever import _matches_filter
    doc = Document(page_content="x", metadata={"min_price": 899, "file_name": "a.txt"})
    _assert(_matches_filter(doc, {"min_price": {"$lte": 1000}}), "lte 命中")
    _assert(not _matches_filter(doc, {"min_price": {"$gte": 1000}}), "gte 未命中")
    _assert(_matches_filter(doc, {"$and": [{"file_name": "a.txt"}, {"min_price": {"$lte": 900}}]}), "$and 命中")
    _assert(not _matches_filter(doc, {"file_name": "b.txt"}), "file_name 未命中")
    _assert(not _matches_filter(doc, {"unknown_field": {"$eq": 1}}), "缺字段→False")
    _assert(_matches_filter(doc, None), "None→True")
    _assert(_matches_filter(doc, {}), "空→True")


# ──────────────────────────────────────────────────────────────
# 6. RRF 融合（真实 reciprocal_rank_fusion）
# ──────────────────────────────────────────────────────────────

def test_rrf_fusion():
    from tools.rrf_fusion import reciprocal_rank_fusion
    d1, d2, d3 = Document(page_content="A"), Document(page_content="B"), Document(page_content="C")
    dense = [(d1, 0.9), (d2, 0.8)]   # dense 排名：A=1, B=2
    sparse = [(d2, 5.0), (d3, 3.0)]  # sparse 排名：B=1, C=2
    fused = reciprocal_rank_fusion(dense, sparse, k=60, top_k=10)
    _assert_eq(len(fused), 3, "去重后 3 个候选")
    _assert_eq(fused[0][0].page_content, "B", "双路都命中的 B 排第一")
    _assert_eq(len(fused[0]), 3, "返回 (doc, score, meta) 三元组")
    _assert_in("dense_rank", fused[0][2], "meta 含 dense_rank")
    _assert_in("sparse_rank", fused[0][2], "meta 含 sparse_rank")
    top2 = reciprocal_rank_fusion(dense, sparse, k=60, top_k=2)
    _assert_eq(len(top2), 2, "top_k=2 截断")


# ──────────────────────────────────────────────────────────────
# 7. 编号条目切分（真实 split_numbered_entries）
# ──────────────────────────────────────────────────────────────

def test_split_numbered_entries():
    from tools.entry_splitter import split_numbered_entries
    text = (
        "## 入门级\n"
        "1. **型号A**\n"
        "   - 吸力：1000Pa\n"
        "2. **型号B**\n"
        "   - 吸力：2000Pa"
    )
    docs = split_numbered_entries(text, {"file_name": "x.txt"})
    _assert_eq(len(docs), 2, "切出 2 个条目")
    _assert_eq(docs[0].page_content, "## 入门级\n1. **型号A**\n   - 吸力：1000Pa", "条目1含章节前缀")
    _assert_eq(docs[0].metadata.get("file_name"), "x.txt", "metadata 透传")
    _assert_eq(split_numbered_entries("普通段落文字", {}), [], "无编号条目→空")


# ──────────────────────────────────────────────────────────────
# 8. 端到端：HybridRetriever 域过滤（需 Ollama + Chroma）
# ──────────────────────────────────────────────────────────────

@e2e("HybridRetriever 端到端需要 Ollama + Chroma 数据")
def test_hybrid_domain_filter_e2e():
    from tools.hybrid_retriever import HybridRetriever
    hr = HybridRetriever()
    hr.ensure_sparse_index()
    chunks = hr.search("吸力下降", filter={"file_name": "不染一尘常见维修问题.txt"})
    files = {c.metadata.get("file_name") for c in chunks}
    _assert(len(chunks) > 0, "维修域有召回")
    _assert_eq(len(files), 1, "只返回维修域")
    _assert_in("不染一尘常见维修问题.txt", files, "结果来自维修域")


TESTS = [
    test_route_domain,
    test_is_consulting,
    test_is_aftersales,
    test_is_brand,
    test_word_dict_integrity,
    test_tokenize,
    test_matches_filter,
    test_rrf_fusion,
    test_split_numbered_entries,
    test_hybrid_domain_filter_e2e,
]


def run():
    reset()
    return run_tests("检索链路测试（域路由/分词/过滤/RRF/切分）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
