"""SOP（标准操作流程）基础设施：会话状态 + 执行器。

一个 SOP 是一组「有状态」的步骤，通过槽位（slot）记忆上下文，
逐步引导用户完成一个业务场景（选购推荐、故障排查等）。

步骤类型：
  - ask     ：向用户提问，收集一个槽位
  - action  ：调用 skill 执行动作（如按预算检索）
  - reply   ：输出最终结果并结束 SOP

会话状态为内存存储，按 session_id 隔离（支持多会话；服务重启后清空）。
"""

import re
from tools.log_tool import get_logger
from config.word_dict_config import (
    DOMAIN_MAP, REPAIR_WORDS, MAINTAIN_WORDS, AFTERSALES_WORDS,
    BUY_WORDS, CONSULT_WORDS,
    LATEST_WORDS, RECENT_VAGUE_WORDS, CHEAPER_WORDS,
)

logger = get_logger(name="sops")
SOPS = {}  # sop_id → sop 定义


def is_consulting(query: str) -> bool:
    """判断是否是「选购咨询」（选购要注意什么），而非「选购动作」（我要买）。

    咨询类含"选购/购买"等动作词 + "注意/问题/技巧"等咨询词，
    这类是 FAQ 问答，不应触发选购 SOP。
    """
    has_buy = any(w in query for w in BUY_WORDS)
    has_consult = any(w in query for w in CONSULT_WORDS)
    return has_buy and has_consult


def is_aftersales(query: str) -> bool:
    """判断是否是「售后咨询」（保修/售后/退换货），而非「选购咨询」。"""
    return any(w in query for w in AFTERSALES_WORDS)


def register(sop: dict) -> None:
    """注册一个 SOP 定义。"""
    SOPS[sop["id"]] = sop


# 上一轮推荐的结构化结果（型号列表），用于「追问」处理，按 session_id 隔离
_last_recommends = {}


def save_recommend(session_id: str, models) -> None:
    """保存指定会话的上一轮推荐结果（供追问「有没有更新的/更便宜的」使用）。"""
    _last_recommends[session_id] = models


def get_last_recommend(session_id: str):
    return _last_recommends.get(session_id)


def clear_recommend(session_id: str) -> None:
    _last_recommends.pop(session_id, None)


# 当前活跃会话，按 session_id 隔离
_sessions = {}


def has_active_sop(session_id: str) -> bool:
    return session_id in _sessions


def get_active_sop_id(session_id: str):
    """返回指定会话活跃 SOP 的 id；无活跃 SOP 时返回 None。"""
    s = _sessions.get(session_id)
    return s["sop_id"] if s else None


def _start(session_id: str, sop_id: str):
    _sessions[session_id] = {"sop_id": sop_id, "step": 0, "slots": {}, "retry_count": 0}
    return _sessions[session_id]


def _end(session_id: str):
    _sessions.pop(session_id, None)


def end_sop(session_id: str):
    """主动结束指定会话的 SOP（用户说「算了/退出」等场景）。"""
    s = _sessions.get(session_id)
    if s:
        logger.info(f"[SOP] SOP End:{s['sop_id']}")
    _end(session_id)


def start_sop(session_id: str, sop_id: str, query: str):
    """进入一个 SOP。返回 (reply, done)。

    首轮先尝试用触发 query 提取第一个槽位（预填用户已给的信息），
    提取失败再问第一个问题。开场提示（intro）可选，有则先输出。
    """
    _start(session_id, sop_id)
    logger.info(f"[SOP] Start SOP: {sop_id}")
    reply, done = _run(session_id, query)
    intro = SOPS[sop_id].get("intro")
    if intro:
        return intro + "\n\n" + reply, done
    return reply, done


def continue_sop(session_id: str, query: str):
    """继续指定会话的 SOP。返回 (reply, done)；无活跃 SOP 时返回 None。"""
    if session_id not in _sessions:
        return None
    return _run(session_id, query)


def match_sop(query: str):
    """按 trigger 匹配 SOP，并评估 guards（排除条件）。

    trigger 命中但任一 guard 命中 → 返回 None（该场景不触发 SOP，走后续分支）；
    trigger 命中且 guards 均未命中 → 返回 sop_id。
    """
    for sop in SOPS.values():
        if any(kw in query for kw in sop.get("trigger", [])):
            if any(g["check"](query) for g in sop.get("guards", [])):
                return None
            logger.info(f"[SOP] Match SOP:{sop['id']}")
            return sop["id"]
    return None


def _run(session_id: str, user_input: str):
    """执行/推进指定会话的 SOP。返回 (reply_text, done)。"""
    session = _sessions[session_id]
    sop = SOPS[session["sop_id"]]
    steps = sop["steps"]

    while session["step"] < len(steps):
        step = steps[session["step"]]

        if step["type"] == "ask":
            # 有用户输入 → 尝试提取槽位；否则 → 问问题
            if user_input is None:
                return step["ask"], False
            value = step["extract"](user_input, session["slots"])
            if value is None:
                # 提取失败：累加重试次数，超过上限则放弃该槽位（记 None 跳过）
                session["retry_count"] = session.get("retry_count", 0) + 1
                if session["retry_count"] > sop.get("max_retry", 2):
                    session["slots"][step["slot"]] = None
                    session["step"] += 1
                    session["retry_count"] = 0
                    user_input = None
                    continue
                # 首轮提取失败（retry_count==1）→ 用 ask 话术（还没问过用户）
                # 后续提取失败（retry_count≥2）→ 用 retry 话术（重问）
                if session["retry_count"] == 1:
                    return step["ask"], False
                return step.get("retry", step["ask"]), False
            # 提取成功，重置重试计数
            session["slots"][step["slot"]] = value
            session["step"] += 1
            session["retry_count"] = 0
            user_input = None
            continue

        elif step["type"] == "action":
            # 执行 skill，结果存入 result；若含结构化型号列表则保存供追问
            result = step["action"](session["slots"])
            session["result"] = result
            if isinstance(result, dict) and "models" in result:
                save_recommend(session_id, result["models"])
            session["step"] += 1
            continue

        elif step["type"] == "reply":
            # 输出结果并结束
            ctx = {**session["slots"], **session.get("result", {})}
            try:
                reply = step["template"].format(**ctx)
            except (KeyError, IndexError):
                reply = step.get("fallback", "抱歉，出了一点小问题，请重新提问～")
            _end(session_id)
            return reply, True

    # 步骤走完但无 reply（异常防御），安全结束
    _end(session_id)
    return "", True


def handle_followup(session_id: str, query: str):
    """回答追问。返回回复文本或 None。

    两类追问语义不同：
      - 最近发布/新款 → 重定向：全局按发布时间倒序（不限上一轮预算区间）
      - 更便宜/划算 → 限定：上一轮结果内按价格升序
    """
    # 最近发布类：全局检索（不依赖上一轮上下文）
    #   - 明确词："新款/最新/比较新/最近发布/新出/上市"
    #   - 笼统"最近"（后面不带时间单位）+ "发布/新/出"
    latest_hit = any(w in query for w in LATEST_WORDS)
    vague_recent = (
        "最近" in query
        and not re.search(r"最近(?:半|几|[0-9一二两三四五六七八九十]|个?[月年周天])", query)
        and any(w in query for w in RECENT_VAGUE_WORDS)
    )
    if latest_hit or vague_recent:
        return _format_models(_search_latest_global()[:5], "最近发布的机器人有这几款：")

    # 全局价格极值：最贵/最便宜 → 全局按价格排序取极值（不依赖上一轮）
    if "最贵" in query:
        models = _search_price_global()
        if models:
            return _format_models(models[-1:], "目前最贵的是这一款：")
    if "最便宜" in query:
        models = _search_price_global()
        if models:
            return _format_models(models[:1], "目前最便宜的是这一款：")

    # 限定追问：更便宜 → 上一轮结果内按价格升序
    if any(w in query for w in CHEAPER_WORDS):
        models = get_last_recommend(session_id)
        if not models:
            return None
        sorted_models = sorted(models, key=lambda m: m.get("price") or 0)
        return _format_models(sorted_models[:3], "在刚才的推荐里，更便宜的有这几款：")

    return None


def _format_models(models, header: str) -> str:
    """把型号列表格式化成回复文本。"""
    from tools.metadata_extractor import format_model_line
    lines = [format_model_line(m) for m in models]
    return header + "\n\n" + "\n".join(lines)


def _search_latest_global():
    """全局检索所有型号，按发布时间倒序。"""
    from tools.metadata_extractor import enumerate_models
    models = enumerate_models({"file_name": {"$ne": "__never__"}}, require_field="publish_date")
    models.sort(key=lambda m: m.get("publish_date") or "", reverse=True)
    return models


def _search_price_global():
    """全局检索所有型号，按价格升序（供「最贵/最便宜」极值查询）。"""
    from tools.metadata_extractor import enumerate_models
    models = enumerate_models({"file_name": {"$ne": "__never__"}})
    models.sort(key=lambda m: m.get("price") or 0)
    return models
