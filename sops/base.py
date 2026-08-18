"""SOP（标准操作流程）基础设施：会话状态 + 执行器。

一个 SOP 是一组「有状态」的步骤，通过槽位（slot）记忆上下文，
逐步引导用户完成一个业务场景（选购推荐、故障排查等）。

步骤类型：
  - ask     ：向用户提问，收集一个槽位
  - action  ：调用 skill 执行动作（如按预算检索）
  - reply   ：输出最终结果并结束 SOP

会话状态为内存存储（当前单机单用户；多用户时改为按 user_id 隔离）。
"""

SOPS = {}  # sop_id → sop 定义


def register(sop: dict) -> None:
    """注册一个 SOP 定义。"""
    SOPS[sop["id"]] = sop


# 当前活跃会话（单用户）
_session = None


def has_active_sop() -> bool:
    return _session is not None


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
            # 执行 skill，结果存入 result
            session["result"] = step["action"](session["slots"])
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
