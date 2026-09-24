"""检索链路测试：域路由 + 词表 + BM25 分词 + 过滤 + RRF 融合 + 阈值 + 精排 + 条目切分。

覆盖（全部测真实代码，不再复制 mock 副本）：
  - agent._route_domain / _domain_filter（知识域 → file_name；两域接近时返回 $in）
  - sops.base.is_consulting / is_aftersales / is_brand
  - config.word_dict_config 词表完整性（DOMAIN_MAP / SYMPTOM_MAP / SYMPTOM_QUERY_MAP）
  - SparseRetriever._tokenize（中英混合分词）
  - sparse_retriever._matches_filter（Chroma where 内存过滤）
  - rrf_fusion.reciprocal_rank_fusion（稠密+稀疏融合排序）
  - entry_splitter.split_numbered_entries（编号条目 + 章节前缀）
  - HybridRetriever._drop_without_dense_support（融合侧兜底）
  - reranker.rerank 的降级契约（模型不可用 → None）
  - [--e2e] HybridRetriever 端到端：域过滤 / 阈值过滤 / 精排打分（需 Ollama + Chroma + reranker 模型）"""
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


def test_domain_scoring_and_filter():
    """域路由新形态（2026-09-24 打分式）：打分排序 + 平局按优先级 + 可选双域 filter。

    与 _route_domain 的分工：后者保留旧语义（只给 top1 主文件，供 _window_domain 与兼容
    调用点用）；**实际检索走 _domain_filter**，两域咬得近时会返回 `$in` 两文件。
    """
    from tools.agent import _domain_filter, _domain_scores, _route_domain

    # 打分 = 命中词的**长度之和**（词越长越具体）；同分按历史优先级
    _assert_eq(_domain_scores("为什么买不染一尘")[0][0], "brand", "品牌优先于「买」误判的选购")
    _assert_eq(_domain_scores("机器人不动了怎么办"), [("repair", 2.0)], "单域：不动→维修")
    _assert_eq(_domain_scores("今天天气怎么样"), [], "无域命中")
    # 售后与故障咬得近 → 两个域一起搜
    value, via = _domain_filter("保修期内维修要多少钱")
    files = set(value.get("$in") or []) if isinstance(value, dict) else {value}
    _assert_eq(via, "top2", "两域接近 → 搜两个域")
    _assert_eq(sorted(files), ["不染一尘售后服务.txt", "不染一尘常见维修问题.txt"], "$in 覆盖两个域")
    # 单域 / 无域
    _assert_eq(_domain_filter("扫地机器人保修多久")[0], "不染一尘售后服务.txt", "top1 单域")
    _assert_eq(_domain_filter("今天天气怎么样"), (None, "no-domain"), "无域 → 全库兜底")
    _assert_eq(_route_domain("保修期内维修要多少钱"), "不染一尘售后服务.txt", "_route_domain 仍是 top1 主文件")


def test_domain_files_config():
    """域 → 文件映射新形态：一个域可挂多文件，且文件必须真实存在（防改名后静默失效）。"""
    from config.word_dict_config import DOMAIN_FILES, DOMAIN_MAP, domain_files_of
    from config.word_dict_config import domain_filter, validate_knowledge_domains
    from tools.config_tool import get_data_dir

    _assert_eq(sorted(DOMAIN_MAP), sorted(DOMAIN_FILES), "DOMAIN_MAP 由 DOMAIN_FILES 派生")
    _assert_eq(domain_files_of("aftersales"), ["不染一尘售后服务.txt"], "域键 → 文件列表")
    _assert_eq(domain_files_of("不染一尘售后服务.txt"), ["不染一尘售后服务.txt"], "传文件名也认")
    _assert_eq(domain_files_of("不存在"), ["不存在"], "未知名字原样返回（不改调用方行为）")
    _assert_eq(domain_filter("aftersales"), "不染一尘售后服务.txt", "单文件退化为字符串")
    problems = validate_knowledge_domains(get_data_dir())
    _assert_eq(problems, [], f"域自检无问题（指向的文件都存在、无未认领文件）：{problems}")


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
# 8. 融合侧兜底：丢弃无稠密支撑的候选（P0-1）
# ──────────────────────────────────────────────────────────────

def test_drop_without_dense_support():
    from tools.hybrid_retriever import HybridRetriever
    a, b, c = (Document(page_content=t) for t in ("A", "B", "C"))
    fused = [(a, 0.03, {}), (b, 0.02, {}), (c, 0.01, {})]
    dense = [(a, 0.5), (c, 0.4)]        # B 只被稀疏路命中
    kept = HybridRetriever._drop_without_dense_support(fused, dense)
    _assert_eq([d.page_content for d, _, _ in kept], ["A", "C"], "无稠密支撑的 B 被丢弃")
    _assert_eq(len(HybridRetriever._drop_without_dense_support(fused, [])), 0, "稠密全空→全部丢弃")
    _assert_eq(len(HybridRetriever._drop_without_dense_support([], dense)), 0, "空融合→空")
    all_dense = [(a, 0.5), (b, 0.4), (c, 0.3)]
    _assert_eq(len(HybridRetriever._drop_without_dense_support(fused, all_dense)), 3, "全有支撑→不丢")


# ──────────────────────────────────────────────────────────────
# 9. 精排模块的降级契约（P0-2）
# ──────────────────────────────────────────────────────────────

def test_reranker_unavailable_returns_none():
    """模型不可用时 rerank 必须返回 None（调用方据此退回 RRF 顺序），且不抛异常。"""
    from tools import reranker
    cands = [(Document(page_content="边刷每 6 个月更换一次"), 0.03, {"dense_rank": 1})]
    _assert_eq(reranker.rerank("边刷多久换一次", []), [], "空候选→空列表（不触发模型加载）")
    orig = reranker._ensure_model
    reranker._ensure_model = lambda: False
    try:
        _assert_eq(reranker.rerank("边刷多久换一次", cands), None, "模型不可用→None")
    finally:
        reranker._ensure_model = orig


# ──────────────────────────────────────────────────────────────
# 10. 端到端：阈值过滤 + 精排（需 Ollama + Chroma + 本地 reranker 模型）
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


@e2e("阈值/精排端到端需要 Ollama + Chroma 数据")
def test_score_threshold_filters_out_of_domain_e2e():
    """P0-1 验收：领域外 query 召回应为 0 或显著减少，领域内不受影响。"""
    from tools.hybrid_retriever import HybridRetriever
    hr = HybridRetriever()
    hr.ensure_sparse_index()
    for q in ("今天天气怎么样", "给我讲个笑话", "asdfghjkl 123456"):
        _assert_eq(len(hr.search(q)), 0, f"领域外无召回: {q}")
    for q in ("边刷多久换一次", "机器人不动了怎么办", "吸力下降"):
        _assert(len(hr.search(q)) > 0, f"领域内仍有召回: {q}")


@e2e("精排端到端需要本地 bge-reranker-v2-m3 模型")
def test_rerank_scores_e2e():
    from tools import reranker
    if not reranker._ensure_model():
        _skip("精排模型不可用（未下载或依赖缺失）——降级契约另由 test_reranker_unavailable_returns_none 覆盖")
        return
    on = Document(page_content="边刷建议每 6 个月更换一次，磨损后清扫效果明显下降")
    off = Document(page_content="今天天气晴朗，适合外出散步和野餐")
    res = reranker.rerank("边刷多久换一次", [(on, 0.03, {}), (off, 0.02, {})])
    _assert(res is not None, "模型已就绪时 rerank 不应返回 None")
    _assert_eq(res[0][0].page_content, on.page_content, "相关文档排第一")
    _assert(res[0][1] > res[1][1], "相关分高于无关分")
    _assert_in("rerank_score", res[0][2], "meta 内含 rerank_score")


TESTS = [
    test_route_domain,
    test_domain_scoring_and_filter,
    test_domain_files_config,
    test_is_consulting,
    test_is_aftersales,
    test_is_brand,
    test_word_dict_integrity,
    test_tokenize,
    test_matches_filter,
    test_rrf_fusion,
    test_split_numbered_entries,
    test_drop_without_dense_support,
    test_reranker_unavailable_returns_none,
    test_hybrid_domain_filter_e2e,
    test_score_threshold_filters_out_of_domain_e2e,
    test_rerank_scores_e2e,
]


def run():
    reset()
    return run_tests("检索链路测试（域路由/分词/过滤/RRF/切分）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
