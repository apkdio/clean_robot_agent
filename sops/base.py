"""SOP（标准操作流程）基础设施：会话状态 + 执行器。

一个 SOP 是一组有状态的步骤，通过槽位（slot）记忆上下文，逐步引导用户完成一个业务场景。
步骤类型：ask（提问收集槽位）、action（执行动作，如按预算检索）、reply（输出结果并结束）。
"""
import hashlib
import json
import re
import time

from tools import trace_store
from tools.log_tool import get_logger
from config.word_dict_config import (
    AFTERSALES_WORDS,BUY_WORDS, CONSULT_WORDS, BRAND_WORDS,
    LATEST_WORDS, RECENT_VAGUE_WORDS, CHEAPER_WORDS, PRICIER_WORDS, NEWER_WORDS, OLDER_WORDS,
    REL_LABEL as _REL_LABEL, REL_SUPER as _REL_SUPER,
)

logger = get_logger(name="sops_base")
SOPS = {}  # sop_id → sop 定义
# 定义形如 {"id": xxx, "trigger": [xxx], "guards": [xxx], "intro": xxx, "max_retry": xxx, "steps": [xxx]}
# 步骤形如 {"id": xxx, "type": ask|action|reply, "slot": xxx, "ask": xxx, "retry": xxx, "extract": xxx}
#          （action 步换成 "action"，reply 步换成 "template"（字符串或接收 ctx 的函数）；可选 "when" 条件跳过）


def is_consulting(query: str) -> bool:
    """判断是否是「选购咨询」（选购要注意什么），而非「选购动作」（我要买）。
    需要同时命中两个词表才能判True
    """
    has_buy = any(w in query for w in BUY_WORDS)
    has_consult = any(w in query for w in CONSULT_WORDS)
    return has_buy and has_consult


def is_aftersales(query: str) -> bool:
    """判断是否是「售后咨询」（保修/售后/退换货），而非「选购咨询」。"""
    return any(w in query for w in AFTERSALES_WORDS)


def is_brand(query: str) -> bool:
    """判断是否是「品牌咨询」（为什么买/优势/介绍），而非「选购动作」。"""
    return any(w in query for w in BRAND_WORDS)


def _def_ver(sop: dict) -> str:
    """SOP 定义里「与续跑状态耦合」那部分的短指纹：步骤序列（id / 类型 / 槽位 / 是否带 when）与重试上限。

    只用于判断 Redis 里的半截状态还能不能用：文案（ask / retry / intro / template）与触发词、guards
    变了不影响续跑，不进指纹。
    """
    shape = [[s.get("id"), s.get("type"), s.get("slot"), bool(s.get("when"))] for s in sop.get("steps") or []]
    raw = json.dumps([shape, sop.get("max_retry", 2)], ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def register(sop: dict) -> None:
    """注册一个 SOP 定义，并算好它的定义指纹（供 Redis 里的半截状态对账）。"""
    sop["def_ver"] = _def_ver(sop)
    SOPS[sop["id"]] = sop


def save_recommend(session_id: str, models) -> None:
    """保存指定会话的上一轮推荐结果到 meta（供追问使用，跨重启有效）。"""
    from tools.context_store import set_last_models
    set_last_models(session_id, models)


def get_last_recommend(session_id: str):
    """读取上一轮推荐结果（从 meta，跨重启有效）。"""
    from tools.context_store import get_last_models
    return get_last_models(session_id)


# 当前活跃会话，按 session_id 隔离（Redis 可用时存 Redis，否则本地降级）
_sessions = {}
# 状态形如 {"sop_id": xxx, "def_ver": xxx, "step": xxx, "slots": {xxx}, "retry_count": xxx, "result": {xxx}}
# （无 Redis 的降级态另带 "ts"，读时按 sop_ttl 校验过期；Redis 在时不带，由 Redis 自己过期）
_finished = {}  # session_id -> 刚收尾的那份状态（形状同 _sessions，供 CaseTrace 记槽位/结果，取走即清）


def _session_key(session_id: str) -> str:
    return f"sop:session:{session_id}"


def make_session(sop_id: str, step: int = 0, slots: dict = None, retry_count: int = 0) -> dict:
    """造一份带定义指纹的会话状态（_start 与测试注入共用，避免手搓状态漏指纹）。"""
    return {"sop_id": sop_id, "def_ver": (SOPS.get(sop_id) or {}).get("def_ver"),
            "step": int(step), "slots": dict(slots or {}), "retry_count": int(retry_count)}


def _same_def(session: dict) -> bool:
    """状态与当前 SOP 定义是否同一版（没带指纹的老状态、SOP 已被删掉，都算对不上）。"""
    sop = SOPS.get(session.get("sop_id") or "")
    return bool(sop) and session.get("def_ver") == sop.get("def_ver")


def _load_session(session_id: str):
    """读会话状态；与当前 SOP 定义对不上就丢掉它（代码改过 SOP、Redis 里的旧状态还在）。

    状态的 step 是步骤下标、slots 是槽位键名，定义一动就会指向别的步骤，续跑必然串场；
    这类状态一律作废，本轮按「没有活跃 SOP」正常作答，并留一条 WARNING 便于对账。
    """
    from tools.redis_store import get_redis, json_loads, sop_ttl
    r = get_redis()
    if r is not None:
        raw = r.get(_session_key(session_id))
        session = json_loads(raw) if raw else None
    else:
        # 降级的内存态没有 Redis 的 TTL，得自己算：ts 每轮重盖，与 SET ... EX 一样是滑动窗口
        session = _sessions.get(session_id)
        if session and time.time() - float(session.get("ts") or 0) > sop_ttl():
            logger.warning("[SOP] 本地降级会话已过期，作废：sop=%s（session=%s）",
                           session.get("sop_id"), session_id)
            _delete_session(session_id)
            return None
    if session and not _same_def(session):
        cur = (SOPS.get(session.get("sop_id") or "") or {}).get("def_ver") or "未注册"
        logger.warning("[SOP] SOP 定义已变，作废旧状态：sop=%s 状态指纹=%s 现定义=%s（session=%s）",
                       session.get("sop_id"), session.get("def_ver") or "无", cur, session_id)
        # 留下痕（节点与状态都在闭集里；by 的新取值见 notes/TRACE.md）
        trace_store.node("sop.active", status="skipped", by="stale_def",
                         in_={"sop": session.get("sop_id"),
                              "def": "%s→%s" % (session.get("def_ver") or "无", cur)})
        _delete_session(session_id)
        return None
    return session


def _save_session(session_id: str, session: dict):
    """写会话状态。Redis 可用写 Redis（带 TTL），否则写本地 _sessions。"""
    from tools.redis_store import get_redis, json_dumps, sop_ttl
    r = get_redis()
    if r is not None:
        r.set(_session_key(session_id), json_dumps(session), ex=sop_ttl())
    else:
        session["ts"] = time.time()      # 降级态自己算过期时间（Redis 路径由 Redis 管）
        _sessions[session_id] = session


def _delete_session(session_id: str):
    """删除会话状态。"""
    from tools.redis_store import get_redis
    r = get_redis()
    if r is not None:
        r.delete(_session_key(session_id))
    _sessions.pop(session_id, None)


def pop_finished(session_id: str) -> dict:
    """取走「刚收尾」的 SOP 状态（含槽位与 action 结果），给 trace 用；没取过就是空 dict。"""
    return _finished.pop(session_id, None) or {}


def has_active_sop(session_id: str) -> bool:
    return _load_session(session_id) is not None


def get_active_sop_id(session_id: str):
    """返回指定会话活跃 SOP 的 id；无活跃 SOP 时返回 None。"""
    s = _load_session(session_id)
    return s["sop_id"] if s else None


def get_sop_state(session_id: str) -> dict:
    """返回活跃 SOP 的进度（sop_id / step / slots / retry_count）；无活跃 SOP 时空 dict。"""
    return _load_session(session_id) or {}


def _start(session_id: str, sop_id: str):
    session = make_session(sop_id)
    _save_session(session_id, session)
    return session


def _end(session_id: str):
    _delete_session(session_id)


def end_sop(session_id: str):
    """主动结束指定会话的 SOP（用户说「算了/退出」等场景）。"""
    s = _load_session(session_id)
    if s:
        logger.info(f"[SOP] SOP End:{s['sop_id']}")
    _end(session_id)


def _with_exit_hint(intro: str) -> str:
    """给 SOP 开场语追加退出提示（所有 SOP 通用，用户可回复「0」退出
    """
    return intro + "\n（随时可回复「0」退出本环节）"


def start_sop(session_id: str, sop_id: str, query: str):
    """进入一个 SOP。返回 (reply, done)。

    首轮先尝试用触发 query 提取第一个槽位（预填用户已给的信息），
    提取失败再问第一个问题。开场提示（intro）可选，有则先输出。
    """
    from tools.context_store import set_enter_declined
    set_enter_declined(session_id, False)   # 用户这次真进了流程，之前「别再问」的标记作废
    _start(session_id, sop_id)
    logger.info(f"[SOP] Start SOP: {sop_id}")
    reply, done = _run(session_id, query)
    intro = SOPS[sop_id].get("intro")
    if intro:
        return _with_exit_hint(intro) + "\n\n" + reply, done
    return reply, done


def continue_sop(session_id: str, query: str):
    """继续指定会话的 SOP。返回 (reply, done)；无活跃 SOP 时返回 None。"""
    if not has_active_sop(session_id):
        return None
    return _run(session_id, query)


def match_sop(query: str, sop_id: str = None):
    """找命中的 SOP：trigger 命中，且它自己的 guards 都没命中（guard 只否掉自己这条）。

    给了 sop_id 就只看这一条（「这条 SOP 该不该走」的判定）；不给则按注册顺序返回第一条，都不命中返回 None。
    """
    for sid, sop in SOPS.items():
        if sop_id and sid != sop_id:
            continue
        if not any(kw in query for kw in sop.get("trigger", [])):
            continue
        if any(g["check"](query) for g in sop.get("guards", [])):
            continue
        logger.info(f"[SOP] Match SOP:{sid}")
        return sid
    return None


def sop_needs_enter_confirm(sop_id: str, query: str) -> bool:
    """判断是否要走「进入确认门」：只命中弱触发词、没有明确诉求时为 True。"""
    sop = SOPS.get(sop_id) or {}
    weak = sop.get("weak_trigger") or []
    if not weak:
        return False
    strong = [w for w in sop.get("trigger", []) if w not in weak]
    if any(w in query for w in strong):
        return False
    return any(w in query for w in weak)


def _ask_text(step: dict, key: str, ctx: dict) -> str:
    """取问话文案；文案里带 {} 占位时用槽位/结果填充，缺键就原样返回。"""
    text = step.get(key, step["ask"]) or ""
    try:
        return text.format(**ctx)
    except (KeyError, IndexError):
        return text


def _run(session_id: str, user_input: str):
    """执行/推进指定会话的 SOP。返回 (reply_text, done)。"""
    session = _load_session(session_id)
    if session is None:
        return "", True
    sop = SOPS[session["sop_id"]]
    steps = sop["steps"]
    ended = False  # 是否已结束（reply/异常防御里 _end 置 True）
    flow = []      # 本轮走到的动作链（进 trace）

    try:
        while session["step"] < len(steps):
            step = steps[session["step"]]
            ctx = {**session["slots"], **session.get("result", {})}
            # 针对网点的结果，如果结果只有一个，跳过反问澄清环节
            if step.get("when") and not step["when"](ctx):
                flow.append({"i": session["step"], "type": step["type"], "outcome": "skipped"})
                session["step"] += 1
                continue

            if step["type"] == "ask":
                # 有用户输入 → 尝试提取槽位；否则 → 问问题
                if user_input is None:
                    logger.info("[SOP] %s ask slot=%s", sop["id"], step.get("slot"))
                    flow.append({"i": session["step"], "type": "ask",
                                 "slot": step.get("slot"), "outcome": "asked"})
                    return _ask_text(step, "ask", ctx), False
                value = step["extract"](user_input, session["slots"])
                if value is None:
                    # 提取失败：累加重试次数，超过上限则放弃该槽位（记 None 跳过）
                    session["retry_count"] = session.get("retry_count", 0) + 1
                    if session["retry_count"] > sop.get("max_retry", 2):
                        session["slots"][step["slot"]] = None
                        session["step"] += 1
                        session["retry_count"] = 0
                        user_input = None
                        logger.warning("[SOP] %s slot %s abandoned after retries", sop["id"], step["slot"])
                        flow.append({"i": session["step"] - 1, "type": "ask",
                                     "slot": step.get("slot"), "outcome": "abandoned"})
                        continue
                    # 首轮提取失败（retry_count==1）→ 用 ask 话术（还没问过用户）
                    # 后续提取失败（retry_count≥2）→ 用 retry 话术（重问）
                    logger.info("[SOP] %s slot %s extract failed (retry=%d)", sop["id"], step["slot"], session["retry_count"])
                    if session["retry_count"] == 1:
                        flow.append({"i": session["step"], "type": "ask",
                                     "slot": step.get("slot"), "outcome": "asked"})
                        return _ask_text(step, "ask", ctx), False
                    flow.append({"i": session["step"], "type": "ask",
                                 "slot": step.get("slot"), "outcome": "retry"})
                    return _ask_text(step, "retry", ctx), False
                # 提取成功，重置重试计数
                session["slots"][step["slot"]] = value
                session["step"] += 1
                session["retry_count"] = 0
                user_input = None
                logger.info("[SOP] %s slot %s = %s", sop["id"], step["slot"], value)
                flow.append({"i": session["step"] - 1, "type": "ask", "slot": step.get("slot"),
                             "outcome": "filled",
                             "value": value if len(str(value)) <= 40 else str(value)[:40] + "…"})
                continue

            elif step["type"] == "action":
                # 执行 skill，结果存入 result；若含结构化型号列表则保存供追问
                result = step["action"](session["slots"])
                session["result"] = result
                n_models = len(result.get("models") or []) if isinstance(result, dict) else 0
                flow.append({"i": session["step"], "type": "action", "outcome": "ok",
                             "n_models": n_models})
                if isinstance(result, dict) and "models" in result:
                    save_recommend(session_id, result["models"])
                    logger.info("[SOP] %s action executed (%d models)", sop["id"], n_models)
                else:
                    logger.info("[SOP] %s action executed",sop["id"])
                session["step"] += 1
                continue

            elif step["type"] == "reply":
                # 输出结果并结束；template 可以是字符串（.format 取槽位/结果）或函数（按结果分支，如 0 款换话术）
                ctx = {**session["slots"], **session.get("result", {})}
                flow.append({"i": session["step"], "type": "reply", "outcome": "done"})
                tpl = step["template"]
                try:
                    reply = tpl(ctx) if callable(tpl) else tpl.format(**ctx)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[SOP] reply template failed: %s", e)
                    reply = step.get("fallback", "抱歉，出了一点小问题，请重新提问～")
                ended = True
                _finished[session_id] = session
                _end(session_id)
                logger.info("[SOP] %s finished", sop["id"])
                return reply, True

        # 步骤走完但无 reply（异常防御），安全结束
        ended = True
        _end(session_id)
        logger.warning("[SOP] %s ended without reply", sop["id"])
        return "", True
    except Exception as e:  # noqa: BLE001
        # 步骤函数（when / extract / action）炸了：结束会话，别让下一轮还停在这一步重复失败；
        # 异常照常向上抛——外层 ask_stream 出兜底话术、trace 记 error
        logger.warning("[SOP] %s 第 %s 步执行失败（%s）：%s",
                       sop["id"], session["step"], type(e).__name__, e)
        cur = steps[session["step"]] if session["step"] < len(steps) else {}
        trace_store.node("sop.step_error", status="error",
                         in_={"step": session["step"], "type": cur.get("type"), "slot": cur.get("slot")})
        ended = True        # 让下面的 finally 不再把状态写回去
        _end(session_id)
        raise
    finally:
        trace_store.step("sop", flow=flow)
        # 会话未结束（停在 ask 等下一轮）→ 保存最新状态
        if not ended:
            _save_session(session_id, session)


# 相对上一轮展示范围：rel -> (取值字段, 边界函数, 比较方向, 降序?)
_REL_SPEC = {
    "cheaper": ("price", min, "lt", True),
    "pricier": ("price", max, "gt", False),
    "newer": ("publish", max, "gt", False),
    "older": ("publish", min, "lt", True),
}
# 追问维度的话术（_REL_LABEL / _REL_SUPER）见 config/word_dict_config.py


def _rel_header(rels: list) -> str:
    """相对追问的回复头（多维度时并列）。"""
    return "比刚才那几款%s的有：" % "、".join(_REL_LABEL[r] for r in rels)


def _rel_none_msg(rels: list) -> str:
    """越过门槛的型号一个都没有时的回复；多维度要点名组合，避免误读成单维度已到顶。"""
    if len(rels) == 1:
        return "刚才推荐的那几款已经是目前%s的了～" % _REL_SUPER[rels[0]]
    return "比刚才那几款%s的型号暂时没有～" % "、".join(_REL_LABEL[r] for r in rels)


def _publish_key(model: dict) -> int:
    """型号发布时间 → int YYYYMMDD（记录里可能是文本）。"""
    digits = re.sub(r"\D", "", str(model.get("publish_date") or ""))
    return int(digits) if len(digits) >= 8 else 0


def _relative_filter(prev: list, models: list, rel: str) -> list:
    """按单个相对维度过滤出越过上一轮展示边界的型号（不排序、不截断）。"""
    field, pick, op, _desc = _REL_SPEC[rel]

    def value(m: dict) -> int:
        return int(m.get("price") or 0) if field == "price" else _publish_key(m)

    edges = [value(m) for m in prev if value(m)]
    if not edges:
        return []
    edge = pick(edges)
    return [m for m in models if value(m) and (value(m) < edge if op == "lt" else value(m) > edge)]


def _relative_slice(prev: list, models: list, rels: list) -> list:
    """按一个或多个相对维度过滤（同时满足），最接近门槛的在前，最多 3 条。"""
    for rel in rels:
        models = _relative_filter(prev, models, rel)
    if not models:
        return []
    field, _pick, _op, desc = _REL_SPEC[rels[0]]

    def value(m: dict) -> int:
        return int(m.get("price") or 0) if field == "price" else _publish_key(m)

    return sorted(models, key=value, reverse=desc)[:3]


def handle_followup(session_id: str, query: str):
    """回答追问。返回回复文本或 None。

    全局类（最近发布 / 最贵 / 最便宜）不依赖上一轮；相对类（更便宜 / 更贵 / 更新 / 更旧）
    以上一轮展示过的边界为门槛，在全库取越过边界、最接近边界的几款。
    """
    from tools.metadata_extractor import format_models
    # 最近发布类：全局检索（不依赖上一轮上下文）
    latest_hit = any(w in query for w in LATEST_WORDS)
    vague_recent = (
            "最近" in query
            and not re.search(r"最近(?:半|几|[0-9一二两三四五六七八九十]|个?[月年周天])", query)
            and any(w in query for w in RECENT_VAGUE_WORDS)
    )
    if latest_hit or vague_recent:
        models = _search_global("publish_date", True, "publish_date")[:5]
        save_recommend(session_id, models)
        return format_models(models, "最近发布的机器人有这几款：")

    # 全局价格极值：最贵/最便宜 → 全局按价格排序取极值（不依赖上一轮）
    if "最贵" in query or "价格最高" in query:
        models = _search_global("price")
        if models:
            save_recommend(session_id, models[-1:])
            return format_models(models[-1:], "目前最贵的是这一款：")
    if "最便宜" in query or "价格最低" in query:
        models = _search_global("price")
        if models:
            save_recommend(session_id, models[:1])
            return format_models(models[:1], "目前最便宜的是这一款：")

    # 相对上一轮展示范围的追问：更便宜 / 更贵 / 更新 / 更旧
    # 找到第一个命中四个"更加"域的词
    # 多个相对维度可并列（「更新更便宜的」= 更新 且 更便宜），逐个收窄；
    # 裸「更新」不进球表（与「固件更新」撞），但「更X更Y」结构无歧义，用 regex 兜
    newer_hit = any(w in query for w in NEWER_WORDS) or bool(re.search(r"更新更(?:便宜|贵|新|旧)", query))
    rels = [k for k, hit in (("cheaper", any(w in query for w in CHEAPER_WORDS)),
                             ("pricier", any(w in query for w in PRICIER_WORDS)),
                             ("newer", newer_hit),
                             ("older", any(w in query for w in OLDER_WORDS))) if hit]
    if rels:
        prev = get_last_recommend(session_id)
        if not prev:
            return None
        field = "publish_date" if _REL_SPEC[rels[0]][0] == "publish" else "price"
        got = _relative_slice(prev, _search_global(field, False, field), rels)
        if not got:
            return _rel_none_msg(rels)
        save_recommend(session_id, got)
        return format_models(got, _rel_header(rels))

    return None


def _search_global(sort_key: str = "price", reverse: bool = False, require_field: str = "price"):
    """全局枚举所有型号并按指定字段排序（价格升序 / 发布时间倒序等）。"""
    from tools.metadata_extractor import enumerate_models
    models = enumerate_models({"file_name": {"$ne": "__never__"}}, require_field=require_field)
    default = "" if sort_key == "publish_date" else 0
    models.sort(key=lambda m: m.get(sort_key) or default, reverse=reverse)
    return models
