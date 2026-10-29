"""操作判别器：判断一句话想让系统干哪个操作，供主链路按表序派发。

每个操作一条注册：predicate 判入口（纯函数，只读 query）、allow 管意图合法性、order 定表序。
判据全部复用现成实现（match_sop、领域词表、时间正则等），本文件不重写判据。
"""
from __future__ import annotations

import threading

from tools.log_tool import get_logger

logger = get_logger(name="operations")

# 表序 = 现行主链路的先后（追问 → 型号详情 → 网点 → SOP → 型号筛选 → 域路由）
# 派发按这个顺序尝试；多命中且未过阈值门时，它也充当裁决顺序。
ORDER = ("followup", "model", "service_point", "sop", "find_models", "domain")

_ROBOTISH = ("robot", "unknown")     # 型号/SOP/型号筛选这些分支的合法性门


def _p_followup(query: str) -> bool:
    """命中追问触发词（要不要取上一轮推荐，由执行体判断）。"""
    from config.word_dict_config import (LATEST_WORDS, RECENT_VAGUE_WORDS, CHEAPER_WORDS,
                                         PRICIER_WORDS, NEWER_WORDS, OLDER_WORDS)
    words = LATEST_WORDS + RECENT_VAGUE_WORDS + CHEAPER_WORDS + PRICIER_WORDS + NEWER_WORDS + OLDER_WORDS
    return any(w in query for w in words) or any(
        w in query for w in ("最贵", "最便宜", "价格最高", "价格最低"))


def _p_model(query: str) -> bool:
    """命中型号查询入口：触发词，或咨询词加指代特征。"""
    from config.word_dict_config import MODEL_QUERY_WORDS, MODEL_CONSULT_WORDS
    from tools.model_ref import has_model_ref

    if any(w in query for w in MODEL_QUERY_WORDS):
        return True
    return any(w in query for w in MODEL_CONSULT_WORDS) and has_model_ref(query)


def _p_find_models(query: str) -> bool:
    """预算 / 日期 / 系列任一规则约束命中，就交给型号筛选（判别阶段不调 LLM）。"""
    from tools.model_query import resolve_model_constraints
    return resolve_model_constraints(query, allow_llm=False) is not None


def _p_service_point(query: str) -> bool:
    from sops import match_sop
    return bool(match_sop(query, "service_point", log=False))


def _p_sop(query: str) -> bool:
    from sops import match_sop
    return bool(match_sop(query, log=False))


def _p_domain(query: str) -> bool:
    from tools.domain_router import route_domain
    return bool(route_domain(query))


# name → (predicate, allow, order)
OPERATIONS = {
    "followup": (_p_followup, _ROBOTISH, ORDER.index("followup")),
    "model": (_p_model, _ROBOTISH, ORDER.index("model")),
    "service_point": (_p_service_point, None, ORDER.index("service_point")),   # 不依赖意图
    "sop": (_p_sop, _ROBOTISH, ORDER.index("sop")),
    "find_models": (_p_find_models, _ROBOTISH, ORDER.index("find_models")),
    "domain": (_p_domain, _ROBOTISH, ORDER.index("domain")),
}


def probe(query: str) -> list:
    """命中哪些操作（按表序返回名字列表）；单个谓词出错只跳过它，不影响其余。"""
    out = []
    for name, (pred, _allow, _order) in sorted(OPERATIONS.items(), key=lambda kv: kv[1][2]):
        try:
            if pred(query):
                out.append(name)
        except Exception as e:  # noqa: BLE001
            logger.warning("[Ops] predicate %s failed: %s", name, e)
    return out


# ──────────────────────────────────────────────────────────────
# 操作打分与三段过滤
#   score = norm(dense_max) + norm(bm25_max) + bonus(强判据 0.5)
#   关键句见 config/word_dict_config.py::OPERATION_KEYPHRASES
# ──────────────────────────────────────────────────────────────

_index_lock = threading.Lock()
_bm25 = None                 # BM25Okapi（每条关键句一个短文档）
_bm25_rows = []              # [(op, phrase), ...]，与 _bm25 语料一一对应
_phrase_vecs = {}            # op -> [向量, ...]
_index_failed = False
_dense_failed_logged = False


def _phrases(op: str) -> list:
    from config.word_dict_config import OPERATION_KEYPHRASES
    return [p for p in (OPERATION_KEYPHRASES.get(op) or "").split("##") if p.strip()]


def _ops_cfg() -> dict:
    """operations 段配置（缺失返回空 dict，走内置默认）。"""
    try:
        from tools.config_tool import load_config
        return (load_config("agent").get("operations") or {})
    except Exception as e:  # noqa: BLE001
        logger.warning("[Ops] load config failed: %s", e)
        return {}


_SCORING_MODES = ("off", "shadow", "on")
_DEFAULTS = {"scoring": "on", "floor": 1.20, "gap": 0.10}


def _cfg_opts(cfg: dict) -> tuple:
    """取 (scoring, floor, gap)；缺省或非法值显式告警后回模板默认，避免静默退回表序。"""
    mode = str(cfg.get("scoring", _DEFAULTS["scoring"]) or "").strip().lower()
    if mode not in _SCORING_MODES:
        logger.warning("[Ops] scoring=%r 不在 %s 内，按 on 处理", cfg.get("scoring"), _SCORING_MODES)
        mode = "on"

    def _num(key):
        raw = cfg.get(key, _DEFAULTS[key])
        try:
            val = float(raw)
        except (TypeError, ValueError):
            logger.warning("[Ops] %s=%r 不是数字，按默认 %.2f", key, raw, _DEFAULTS[key])
            return _DEFAULTS[key]
        if val < 0:
            logger.warning("[Ops] %s=%r 为负，按默认 %.2f", key, raw, _DEFAULTS[key])
            return _DEFAULTS[key]
        return val

    return mode, _num("floor"), _num("gap")


def _ensure_index() -> bool:
    """建 BM25 索引与关键句向量；失败返回 False（调用方退化为原有 bool 排序）。"""
    global _bm25, _bm25_rows, _index_failed
    if _bm25 is not None:
        return True
    if _index_failed:
        return False
    with _index_lock:
        if _bm25 is not None:
            return True
        if _index_failed:
            return False
        try:
            from rank_bm25 import BM25Okapi
            from tools.sparse_retriever import SparseRetriever   # 复用同一套分词

            rows = [(op, ph) for op in OPERATIONS for ph in _phrases(op)]
            if not rows:
                raise ValueError("关键句为空")
            _bm25_rows = rows
            _bm25 = BM25Okapi([SparseRetriever._tokenize(ph) for _op, ph in rows])
        except Exception as e:  # noqa: BLE001
            logger.warning("[Ops] BM25 索引构建失败，打分解除：%s", e)
            _index_failed = True
            return False
    return True


def _dense_scores(query: str) -> dict | None:
    """稠密：query 与每个操作的所有关键句取最大余弦；embedding 不可用返回 None。"""
    global _phrase_vecs, _dense_failed_logged
    from tools.intent_router import get_query_embedding

    qv = get_query_embedding(query)
    if qv is None:
        return None
    try:
        if not _phrase_vecs:
            from tools.llm_tool import get_embedding_model
            emb = get_embedding_model()
            for op in OPERATIONS:
                vecs = emb.embed_documents(_phrases(op))
                _phrase_vecs[op] = vecs
        import numpy as np
        q = np.array(qv, dtype=float)
        qn = np.linalg.norm(q) or 1.0
        out = {}
        for op, vecs in _phrase_vecs.items():
            best = 0.0
            for v in vecs:
                a = np.array(v, dtype=float)
                d = qn * (np.linalg.norm(a) or 1.0)
                if d:
                    best = max(best, float(q @ a / d))
            out[op] = best
        return out
    except Exception as e:  # noqa: BLE001
        _phrase_vecs = {}
        if not _dense_failed_logged:
            logger.warning("[Ops] 关键句向量不可用，操作打分退化为 BM25：%s", e)
            _dense_failed_logged = True
        return None


def _strong_ops(query: str) -> set:
    """强判据命中的操作（确定性证据，给固定加分；不是"像"）。"""
    out = set()
    try:
        from sops import match_sop, sop_has_strong_trigger
        from tools.model_query import resolve_model_constraints
        from tools.model_ref import has_model_ref

        if match_sop(query, "service_point", log=False):
            out.add("service_point")
        sid = match_sop(query, log=False)
        if sid and sop_has_strong_trigger(sid, query):
            out.add("sop")
        if resolve_model_constraints(query, allow_llm=False):
            out.add("find_models")
        if has_model_ref(query):
            out.add("model")
    except Exception as e:  # noqa: BLE001
        logger.warning("[Ops] 强判据计算失败（不影响派发）：%s", e)
    return out


def score_all(query: str) -> tuple[dict, str]:
    """给 6 个操作打分，返回 (scores, status)。

    status：ok / dense_degraded（只有 BM25 + 强判据）/ disabled（退回原有 bool 排序）。
    归一化按每轮最大分做，抵消余弦基线（关键句间 ~0.57）与 BM25 无上界。
    """
    if not _ensure_index():
        return {}, "disabled"
    raw = {}
    try:
        bscores = _bm25.get_scores(_tokenize(query))
        for (op, _ph), s in zip(_bm25_rows, bscores):
            raw[op] = max(raw.get(op, 0.0), float(s))
    except Exception as e:  # noqa: BLE001
        logger.warning("[Ops] BM25 打分失败，打分解除：%s", e)
        return {}, "disabled"

    dense = _dense_scores(query)
    status = "ok" if dense is not None else "dense_degraded"

    def _norm(d: dict) -> dict:
        top = max(d.values() or [0.0]) or 1.0
        return {k: v / top for k, v in d.items()}

    nb, nd = _norm(raw), (_norm(dense) if dense else {})
    strong = _strong_ops(query)
    scores = {}
    for op in OPERATIONS:
        bonus = 0.5 if op in strong else 0.0
        # 稠密不可用时把 BM25 放大到同一量纲（0~2），否则 floor 在降级态会失效
        bm_part = nb.get(op, 0.0) * (2.0 if not dense else 1.0)
        scores[op] = round(bm_part + nd.get(op, 0.0) + bonus, 3)
    return scores, status


def _tokenize(text: str) -> list:
    from tools.sparse_retriever import SparseRetriever
    return SparseRetriever._tokenize(text)


def _allowed(name: str, intent: str) -> bool:
    allow = OPERATIONS[name][1]
    return allow is None or intent is None or intent in allow


def resolve(ops: list, *, intent: str = None, margin: float = None,
            low_conf: float = 0.2, scores: dict = None, floor: float = 0.0,
            gap: float = 0.0) -> tuple:
    """三态出口。

    - 不给 scores：按表序裁决（旧行为），多命中且意图低置信 → ambiguous；
    - 给 scores：先按 floor 过门、再按分数降序（同分回表序），top1-top2 < gap → ambiguous。
    """
    if scores is not None:
        usable = [o for o in ops if _allowed(o, intent) and scores.get(o, 0.0) >= floor]
        usable.sort(key=lambda o: (-scores.get(o, 0.0), ORDER.index(o)))
        if not usable:
            return "none", None
        winner = usable[0]
        if len(usable) > 1 and gap > 0 and (scores[winner] - scores[usable[1]]) < gap:
            return "ambiguous", winner
        return "win", winner

    usable = [o for o in ops if _allowed(o, intent)]
    if not usable:
        return "none", None
    winner = usable[0]                      # ops 已按表序
    if len(usable) > 1 and margin is not None and low_conf > 0 and margin < low_conf:
        return "ambiguous", winner
    return "win", winner


def _prefer_structured(usable: list, query: str) -> list:
    """有明确预算/系列/时间约束时，型号筛选优先于泛选购 SOP。

    否则「预算 2000 推荐」会先开 SOP 补槽，后一句普通追问被槽位吞掉；
    只有没有结构约束的「推荐/买」才需要 SOP 收集条件。
    """
    if "find_models" not in usable or "sop" not in usable:
        return usable
    try:
        from sops import match_sop
        if match_sop(query, log=False) == "model_query":
            return ["find_models"] + [o for o in usable if o != "find_models"]
    except Exception as e:  # noqa: BLE001
        logger.warning("[Ops] structured priority check failed: %s", e)
    return usable


def note_ops(query: str, *, intent: str = None, margin: float = None,
             prefer: str = None) -> dict:
    """判定并落 trace 的 route.ops，返回 {ops, usable, state, winner, why}。

    usable 是主链路派发顺序：scoring=on 时按分数、否则按表序。
    """
    from tools import trace_store
    mode, floor, gap = _cfg_opts(_ops_cfg())
    raw = probe(query)
    scores, scoring = score_all(query) if mode != "off" else ({}, "off")

    if mode == "on" and scores:
        # ① 分数阈值：6 个操作各自过 floor
        above_floor = [o for o in ORDER if scores.get(o, 0.0) >= floor]
        # ② 词表/结构命中 + 意图合法性：只有命中现成判据的才算候选
        usable = [o for o in above_floor if o in raw and _allowed(o, intent)]
        usable.sort(key=lambda o: (-scores.get(o, 0.0), ORDER.index(o)))
        # 明确的结构约束优先直出：只有没有预算/系列/时间约束的泛选购才进 SOP 补槽
        preferred = _prefer_structured(usable, query)
        structured = preferred != usable
        usable = preferred
        # ③ 分差：交给调用方判 ambiguous（这里给出状态，agent 据此问用户）
        if not usable:
            state, winner = "none", None
        elif structured:
            state, winner = "win", usable[0]
        elif len(usable) > 1 and (scores[usable[0]] - scores[usable[1]]) < gap:
            state, winner = "ambiguous", usable[0]
        else:
            state, winner = "win", usable[0]
    else:
        state, winner = resolve(raw, intent=intent, margin=margin)
        usable = [o for o in raw if _allowed(o, intent)]
        preferred = _prefer_structured(usable, query)
        if preferred != usable:
            usable = preferred
            state, winner = "win", usable[0]
        above_floor = usable
    candidates = [o for o in raw if _allowed(o, intent)]   # trace 里始终按表序（稳定），派发顺序看 usable
    if prefer and prefer in candidates:                    # 用户澄清选的操作：显式顶到最前
        usable = [prefer] + [o for o in usable if o != prefer]
        state, winner = "win", prefer

    if not usable:
        why = "none"
    elif len(usable) == 1:
        why = "single"
    elif state == "ambiguous":
        why = "threshold"
    elif mode == "on" and scores:
        why = "score"
    else:
        why = "order"

    note = {"ops": "/".join(candidates) or "-", "n": len(candidates), "state": state,
            "winner": winner or "-", "why": why}
    if mode == "on":
        # 两道门各留下多少：标定阈值/词表覆盖时看这两个数
        note["n_candidates"] = len(above_floor)
    if prefer:
        note["override"] = prefer
    # 分数只落 trace（标量字符串）；scoring=on 时它同时决定上面的 usable 顺序
    if scores:
        note["scores"] = "/".join("%s=%.2f" % (n, scores[n])
                                  for n in sorted(scores, key=scores.get, reverse=True))
        below = [o for o in raw if _allowed(o, intent) and scores.get(o, 0.0) < floor]
        if below and mode == "on":
            note["not_mine"] = "/".join("%s=%.2f" % (o, scores.get(o, 0.0)) for o in below)
    note["scoring_status"] = scoring
    trace_store.route("ops", **note)
    # 注意顺序：note 里的 scores 是给 trace 的字符串，返回给主链路的是分数表
    return {"ops": raw, "usable": usable, **note, "scores": scores}
