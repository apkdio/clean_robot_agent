"""知识域路由：给 query 打域分并产出检索 filter。

域是软约束：top1 领先不足时把 top2 域一起搜，检索层仍有撤域二次召回兜底。
"""
from __future__ import annotations

from tools.config_tool import load_config
from tools.log_tool import get_logger

logger = get_logger(name="domain_router")

# 域路由的历史优先级（打分平手时用它排序）。顺序本身承载了修正：
# 品牌优先于选购——「为什么买」的"买"会被误判选购；售后优先于故障——"报修"含"修"会被误判维修。
_DOMAIN_ORDER = ["brand", "aftersales", "consulting", "repair", "maintain", "operation"]

_rag_cfg_cache = None


def _rag_cfg() -> dict:
    """rag.yaml（读失败返回空 dict，走内置默认）。"""
    global _rag_cfg_cache
    if _rag_cfg_cache is None:
        try:
            _rag_cfg_cache = load_config("rag") or {}
        except Exception as e:  # noqa: BLE001
            logger.warning("[Domain] load rag config failed: %s", e)
            _rag_cfg_cache = {}
    return _rag_cfg_cache


def domain_margin() -> float:
    """retrieval.domain_margin：低于它就把 top2 域一起搜；0 = 关闭。"""
    return float((_rag_cfg().get("retrieval") or {}).get("domain_margin", 0) or 0)


def domain_scores(query: str) -> list:
    """给每个知识域打分：命中词的长度之和（词越长越具体，权重越高）。"""
    from config.word_dict_config import (
        AFTERSALES_WORDS, BRAND_WORDS, BUY_WORDS, CONSULT_WORDS,
        MAINTAIN_WORDS, OPERATION_WORDS, REPAIR_BASE_WORDS,
    )

    scores = {}
    for name, words in (("brand", BRAND_WORDS), ("aftersales", AFTERSALES_WORDS),
                        ("repair", REPAIR_BASE_WORDS), ("maintain", MAINTAIN_WORDS),
                        ("operation", OPERATION_WORDS)):
        hit = [w for w in words if w in query]
        if hit:
            scores[name] = float(sum(len(w) for w in hit))
    buy = [w for w in BUY_WORDS if w in query]
    consult = [w for w in CONSULT_WORDS if w in query]
    if buy and consult:
        scores["consulting"] = float(sum(len(w) for w in buy + consult))
    return sorted(scores.items(), key=lambda kv: (-kv[1], _DOMAIN_ORDER.index(kv[0])))


def route_domain(query: str):
    """按 query 内容路由到对应知识域的主文件（识别不准返回 None → 全库兜底）。"""
    from config.word_dict_config import DOMAIN_MAP

    ranked = domain_scores(query)
    return DOMAIN_MAP[ranked[0][0]] if ranked else None


def domain_filter(query: str):
    """域路由 → 检索 filter 取值 + 路径说明。

    top1 领先达到 domain_margin → 只搜主域；
    两域咬得很近（margin 低于阈值）→ 把两个域的文件一起搜（$in）。
    """
    from config.word_dict_config import domain_files_of

    ranked = domain_scores(query)
    if not ranked:
        return None, "no-domain"
    files = domain_files_of(ranked[0][0])
    limit = domain_margin()
    if len(ranked) > 1 and limit > 0:
        margin = ranked[0][1] - ranked[1][1]
        if margin < limit:
            for f in domain_files_of(ranked[1][0]):
                if f not in files:
                    files.append(f)
            logger.info("[Domain] %s=%.1f vs %s=%.1f (margin=%.1f < %.1f) → 搜两个域",
                        ranked[0][0], ranked[0][1], ranked[1][0], ranked[1][1], margin, limit)
            return ({"$in": files} if len(files) > 1 else files[0]), "top2"
    return (files[0] if len(files) == 1 else {"$in": files}), "top1"
