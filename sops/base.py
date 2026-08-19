"""SOP（标准操作流程）基础设施：会话状态 + 执行器。

一个 SOP 是一组「有状态」的步骤，通过槽位（slot）记忆上下文，
逐步引导用户完成一个业务场景（选购推荐、故障排查等）。

步骤类型：
  - ask     ：向用户提问，收集一个槽位
  - action  ：调用 skill 执行动作（如按预算检索）
  - reply   ：输出最终结果并结束 SOP

会话状态为内存存储（当前单机单用户；多用户时改为按 user_id 隔离）。
"""

import re

SOPS = {}  # sop_id → sop 定义

# 知识域映射（场景 → 知识库文件名，即"轻量文档隔离"的域）
DOMAIN_MAP = {
    "consulting": "选购指南.txt",   # 选购咨询（走 RAG 域过滤）
    "repair": "故障排除.txt",       # 故障排查（走 SOP，定向该域）
    "maintain": "维护保养.txt",     # 维护保养（走 RAG 域过滤）
    "model": "具体型号.txt",        # 购买推荐（走结构化直出，metadata 过滤）
}

# 故障关键词（路由到 repair 域）
REPAIR_WORDS = ["故障", "坏了", "不动", "漏水", "异响", "不充电", "异常", "失灵",
                "不好使", "出问题", "趴窝", "卡住", "噪音"]

# 维护关键词（路由到 maintain 域）
MAINTAIN_WORDS = ["维护", "保养", "清洗", "清理", "更换", "耗材", "滤网", "边刷",
                  "主刷", "拖布", "尘盒", "充电座清洁"]


def register(sop: dict) -> None:
    """注册一个 SOP 定义。"""
    SOPS[sop["id"]] = sop


# 上一轮推荐的结构化结果（型号列表），用于「追问」处理
_last_recommend = None


def save_recommend(models) -> None:
    """保存上一轮推荐结果（供追问「有没有更新的/更便宜的」使用）。"""
    global _last_recommend
    _last_recommend = models


def get_last_recommend():
    return _last_recommend


def clear_recommend() -> None:
    global _last_recommend
    _last_recommend = None


# 当前活跃会话（单用户）
_session = None


def has_active_sop() -> bool:
    return _session is not None


def get_active_sop_id():
    """返回当前活跃 SOP 的 id；无活跃 SOP 时返回 None。"""
    return _session["sop_id"] if _session else None


def _start(sop_id: str):
    global _session
    _session = {"sop_id": sop_id, "step": 0, "slots": {}, "retry_count": 0}
    return _session


def _end():
    global _session
    _session = None


def end_sop():
    """主动结束当前 SOP（用户说「算了/退出」等场景）。"""
    _end()


def start_sop(sop_id: str, query: str):
    """进入一个 SOP。返回 (reply, done)。

    第一轮直接问第一个问题（不把触发 query 当作槽位回答，
    避免用户刚说「想买」就被回「预算没听清」）。
    """
    _start(sop_id)
    return _run(None)


def continue_sop(query: str):
    """继续当前 SOP。返回 (reply, done)；无活跃 SOP 时返回 None。"""
    if _session is None:
        return None
    return _run(query)


def match_sop(query: str):
    """按 trigger 关键词匹配一个 SOP。返回 sop_id 或 None。"""
    for sop in SOPS.values():
        if any(kw in query for kw in sop.get("trigger", [])):
            return sop["id"]
    return None


def _run(user_input: str):
    """执行/推进当前 SOP。返回 (reply_text, done)。"""
    session = _session
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
                save_recommend(result["models"])
            session["step"] += 1
            continue

        elif step["type"] == "reply":
            # 输出结果并结束
            ctx = {**session["slots"], **session.get("result", {})}
            try:
                reply = step["template"].format(**ctx)
            except (KeyError, IndexError):
                reply = step.get("fallback", "抱歉，出了一点小问题，请重新提问～")
            _end()
            return reply, True

    # 步骤走完但无 reply（异常防御），安全结束
    _end()
    return "", True


def handle_followup(query: str):
    """回答追问。返回回复文本或 None。

    两类追问语义不同：
      - 最近发布/新款 → 重定向：全局按发布时间倒序（不限上一轮预算区间）
      - 更便宜/划算 → 限定：上一轮结果内按价格升序
    """
    # 最近发布类：全局检索（不依赖上一轮上下文）
    #   - 明确词："新款/最新/比较新/最近发布/新出/上市"
    #   - 笼统"最近"（后面不带时间单位）+ "发布/新/出"
    latest_hit = any(w in query for w in ["新款", "最新", "比较新", "最近发布", "新出", "上市"])
    vague_recent = (
        "最近" in query
        and not re.search(r"最近(?:半|几|[0-9一二两三四五六七八九十]|个?[月年周天])", query)
        and any(w in query for w in ["发布", "新", "出", "上市"])
    )
    if latest_hit or vague_recent:
        return _format_models(_search_latest_global()[:5], "最近发布的机器人有这几款：")

    # 限定追问：更便宜 → 上一轮结果内按价格升序
    if any(w in query for w in ["便宜", "低价", "划算"]):
        models = get_last_recommend()
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
    from tools.metadata_extractor import extract_model_info
    from tools.vector_store import search_by_filter
    docs = search_by_filter({"file_name": {"$ne": "__never__"}})
    models, seen = [], set()
    for c in docs:
        info = extract_model_info(c)
        if info.get("publish_date") and info.get("name") and info["name"] not in seen:
            seen.add(info["name"])
            models.append(info)
    models.sort(key=lambda m: m.get("publish_date") or "", reverse=True)
    return models
