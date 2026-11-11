"""选购/型号查询 SOP 测试：SOP 状态机基础设施 + model_query 选购流程。

覆盖：
  - service.sops.base：register / match_sop（trigger + guards）/ start_sop /
    continue_sop / end_sop / has_active_sop / get_active_sop_id
  - 槽位提取：预算（extract_price_constraint，真实；＋纯数字短答按上限）
  - reply 步：函数模板（按结果分支：0 款换 MODEL_QUERY_EMPTY_REPLIES、有结果保持原格式）
  - 重试循环 / 槽位放弃（max_retry）/ 模板 fallback / 异常防御
  - model_query 触发与 guard（选购咨询/售后/品牌不触发选购）
  - [数据可用时] 选购 SOP 全流程（预算 → 系列 → 时间 → 标签）"""
import sys
import re
import os
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 与 test_scripts._runner 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
import service.sops  # 触发 repair/model_query SOP 注册
import service.sops.base
from test_scripts import trace_reader
from tools import trace_store
from tools.dto.sop import ExtractResult


def _pair(result):
    return result.reply, result.done


def _models_ready():
    try:
        from tools.metadata_extractor import get_all_models
        return len(get_all_models()) > 0
    except Exception:
        return False


def _cleanup():
    service.sops.base._sessions = {}
    # 会话状态已迁到 Redis，同时清空 Redis 里的 SOP 会话，避免测试间残留
    try:
        from gateway.redis_store import get_redis
        r = get_redis()
        if r is not None:
            for k in r.keys("sop:session:*"):
                r.delete(k)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# 测试用自包含 SOP（不依赖 Chroma/LLM）
# ──────────────────────────────────────────────────────────────

def _extract_budget(text, _slots):
    from tools.metadata_extractor import extract_price_constraint
    parsed = extract_price_constraint(text)
    return ExtractResult("done", parsed) if parsed is not None else ExtractResult("fail")


def _extract_has_pet(text, _slots):
    if re.search(r"没|无|不养|没有|没养", text):
        return ExtractResult("done", False)
    if re.search(r"有|养|猫|狗|宠物", text):
        return ExtractResult("done", True)
    return ExtractResult("fail")


def _mock_search(slots):
    budget = slots.get("budget")
    models = [
        {"name": "型号A", "price": 699},
        {"name": "型号B", "price": 999},
        {"name": "型号C", "price": 1299},
    ]
    if not budget:
        matched = models
    else:
        lo, hi = budget
        matched = [m for m in models if (lo is None or m["price"] >= lo) and (hi is None or m["price"] <= hi)]
    return {"count": len(matched), "list": "\n".join(m["name"] for m in matched), "models": matched}


_TEST_SOP = {
    "id": "sop_unit_test",
    "trigger": ["单测触发"],
    "max_retry": 2,
    "steps": [
        {"id": "ask_budget", "type": "ask", "slot": "budget",
         "ask": "预算多少？", "retry": "预算没听清，说个数字。", "extract": _extract_budget},
        {"id": "ask_pet", "type": "ask", "slot": "has_pet",
         "ask": "有宠物吗？", "retry": "有或没有？", "extract": _extract_has_pet},
        {"id": "do_search", "type": "action", "action": _mock_search},
        {"id": "reply", "type": "reply",
         "template": "共 $count 款：$list", "fallback": "抱歉，出了点小问题。"},
    ],
}

_BAD_SOP = {
    "id": "sop_bad_template",
    "trigger": ["坏模板"],
    "steps": [
        {"id": "ask_name", "type": "ask", "slot": "name",
         "ask": "叫什么？", "retry": "说名字。",
         "extract": lambda text, slots: ExtractResult("done", text.strip())
         if text.strip() else ExtractResult("fail")},
        {"id": "reply", "type": "reply",
         "template": "你好$name$missing_field", "fallback": "抱歉，出了点小问题。"},
    ],
}


def _boom_action(_slots):
    raise RuntimeError("step boom")


_BOOM_SOP = {
    "id": "sop_boom_action",
    "trigger": ["炸一下"],
    "steps": [
        {"id": "boom", "type": "action", "action": _boom_action},
        {"id": "reply", "type": "reply", "template": "不会走到这里"},
    ],
}


# ──────────────────────────────────────────────────────────────
# 1. 注册与匹配
# ──────────────────────────────────────────────────────────────

def test_register_and_match():
    service.sops.base.register(_TEST_SOP)
    _assert("sop_unit_test" in service.sops.base.SOPS, "注册成功")
    _assert_eq(service.sops.base.match_sop("单测触发一下"), "sop_unit_test", "trigger 命中")
    _assert_eq(service.sops.base.match_sop("你好呀"), None, "闲聊不触发")
    _assert_eq(service.sops.base.match_sop("边刷多久换一次"), None, "维护不触发")


def test_sop_name_from_definition():
    """名称只认 SOP 定义；匿名测试 SOP 缺省回 id，运行时不再维护第二张名称表。"""
    from service.sops.runtime import sop_name as runtime_sop_name

    _assert_eq(service.sops.base.sop_name("model_query"), "选购推荐", "model_query 名称来自定义")
    _assert_eq(service.sops.base.sop_name("repair"), "故障排查", "repair 名称来自定义")
    _assert_eq(service.sops.base.sop_name("service_point"), "售后网点查询", "service_point 名称来自定义")
    _assert_eq(service.sops.base.sop_name("sop_unit_test"), "sop_unit_test", "匿名 SOP 缺省 name=id")
    _assert_eq(runtime_sop_name("model_query"), "选购推荐", "runtime 薄转发同一份名称")
    _assert_eq(service.sops.base.sop_name("not_registered"), "not_registered", "未知 id 原样返回")


def _expect_register_error(sop, label):
    try:
        service.sops.base.register(sop)
    except ValueError as e:
        _assert("SOP" in str(e), "%s：错误信息带 SOP 定位" % label)
    else:
        _assert(False, "%s：应当拒绝注册" % label)


def test_register_validation():
    """注册前拦住缺字段/错类型/重复 step，避免运行到一半才 KeyError。"""
    base = {"id": "sop_validate", "steps": [{"id": "r", "type": "reply", "template": "ok"}]}
    _expect_register_error(dict(base, id=""), "id 为空")
    _expect_register_error(dict(base, steps=[]), "steps 为空")
    _expect_register_error(dict(base, steps=[
        {"id": "r", "type": "reply", "template": "ok"},
        {"id": "r", "type": "reply", "template": "again"},
    ]), "step id 重复")
    _expect_register_error(dict(base, steps=[{"id": "r", "type": "unknown"}]), "未知 step 类型")
    _expect_register_error(dict(base, steps=[
        {"id": "a", "type": "ask", "slot": "x", "ask": "问"},
    ]), "ask 缺 extract")
    _expect_register_error(dict(base, steps=[{"id": "a", "type": "action"}]), "action 缺 action")
    _expect_register_error(dict(base, steps=[{"id": "r", "type": "reply"}]), "reply 缺 template")
    _expect_register_error(dict(base, guards=[{"check": "not-callable"}]), "guard 不可调用")
    _expect_register_error(dict(base, max_retry=-1), "max_retry 为负")
    _expect_register_error(dict(base, enter_confirm="sometimes"), "enter_confirm 非法")

    service.sops.base.register(dict(base, steps=[{"id": "r", "type": "reply", "template": "ok"}]))
    _assert("sop_validate" in service.sops.base.SOPS, "合法定义可以注册")
    service.sops.base.SOPS.pop("sop_validate", None)


def test_extract_done_skip_and_drop_are_distinct():
    """done 合并 extra；skip 写 None+skipped；fail 超限只写 dropped，不把槽位塞进 slots。"""
    from tools.dto.sop import ExtractResult

    sop_id = "sop_extract_result"
    service.sops.base.register({
        "id": sop_id,
        "steps": [
            {"id": "a", "type": "ask", "slot": "a",
             "ask": "a?", "extract": lambda text, slots: ExtractResult(
                 "done", "A", extra={"candidates": ["x"], "choice_prompt": "选"})},
            {"id": "b", "type": "ask", "slot": "b",
             "ask": "b?", "extract": lambda text, slots: ExtractResult("skip")},
            {"id": "c", "type": "ask", "slot": "c",
             "ask": "c?", "extract": lambda text, slots: ExtractResult("fail")},
            {"id": "reply", "type": "reply", "template": "ok"},
        ],
        "max_retry": 0,
    })
    sid = "s-extract-result"
    try:
        service.sops.base.start_sop(sid, sop_id, "x")
        state = service.sops.base.get_sop_state(sid)
        _assert_eq(state["slots"].get("a"), "A", "done 写槽位")
        _assert_eq(state["slots"].get("candidates"), ["x"], "done 合并 extra")
        _assert_eq(state["slots"].get("choice_prompt"), "选", "extra 原子写入")

        service.sops.base.continue_sop(sid, "随便")
        state = service.sops.base.get_sop_state(sid)
        _assert_eq(state["slots"].get("b"), None, "skip 写 None")
        _assert_eq((state.get("slot_status") or {}).get("b"), "skipped", "skip 记 skipped")

        result = service.sops.base.continue_sop(sid, "还是不对")
        _assert(result.done, "fail 超过 max_retry 后推进到 reply")
        finished = service.sops.base.pop_finished(sid)
        _assert("c" not in (finished.get("slots") or {}), "dropped 槽位不放进 slots")
        _assert_eq((finished.get("slot_status") or {}).get("c"), "dropped",
                   "fail 超限记 dropped")
    finally:
        service.sops.base.end_sop(sid)
        service.sops.base.SOPS.pop(sop_id, None)
        _cleanup()


def test_action_non_dict_is_explicit_error():
    """action 返回非 dict 不能静默当空结果；要报错并结束会话。"""
    sop_id = "sop_action_bad_shape"
    service.sops.base.register({
        "id": sop_id,
        "steps": [
            {"id": "do", "type": "action", "action": lambda slots: ["not", "dict"]},
            {"id": "reply", "type": "reply", "template": "ok"},
        ],
    })
    sid = "s-action-bad"
    try:
        _assert_raises(ValueError, lambda: service.sops.base.start_sop(sid, sop_id, "x"),
                       "action 非 dict 显式报错")
        _assert(not service.sops.base.has_active_sop(sid), "异常后结束会话")
    finally:
        service.sops.base.end_sop(sid)
        service.sops.base.SOPS.pop(sop_id, None)
        _cleanup()


def test_non_model_action_has_no_n_models():
    """#14：只有 action 结果含 models 才写 n_models；repair/网点这类不能伪装成 0 款。"""
    sop_id = "sop_non_model_action"
    service.sops.base.register({
        "id": sop_id,
        "steps": [
            {"id": "do", "type": "action", "action": lambda slots: {"answer": "ok"}},
            {"id": "reply", "type": "reply", "template": "$answer"},
        ],
    })
    sid = "s-non-model-action"
    try:
        reset_trace(sid)
        trace_store.begin_turn(sid, "x")
        service.sops.base.start_sop(sid, sop_id, "x")
        trace_store.end_turn()
        row = trace_reader.load(trace_store._path(sid))[-1]
        flow = (row.get("sop") or {}).get("flow") or []
        action = next((f for f in flow if f.get("type") == "action"), {})
        _assert("n_models" not in action, "非型号 action 不写 n_models")
    finally:
        service.sops.base.end_sop(sid)
        service.sops.base.SOPS.pop(sop_id, None)
        _cleanup()


def test_reenter_hint_key_uses_word_dict():
    """SOP 定义只声明 key，固定话术仍从词表取。"""
    from config.word_dict_config import SOP_REENTER_HINT

    sop = {
        "id": "sop_hint_probe",
        "name": "提示测试",
        "reenter_hint_key": "model_query",
        "steps": [{"id": "r", "type": "reply", "template": "ok"}],
    }
    service.sops.base.register(sop)
    try:
        _assert_eq(service.sops.base.enter_decline_hint("sop_hint_probe"),
                   SOP_REENTER_HINT["model_query"], "按 reenter_hint_key 取词表话术")
    finally:
        service.sops.base.SOPS.pop("sop_hint_probe", None)


def test_model_query_trigger_and_guards():
    # 合并后通用选购入口由 model_query 承接
    _assert_eq(service.sops.base.match_sop("帮我推荐一款扫地机器人"), "model_query", "推荐→选购 SOP")
    _assert_eq(service.sops.base.match_sop("预算1000以内买一个"), "model_query", "预算/买→选购 SOP")
    # guard：选购咨询不触发选购 SOP（走 RAG）
    _assert(service.sops.base.match_sop("选购扫地机器人要注意什么") is None, "选购咨询 guard 拦截")
    _assert(service.sops.base.match_sop("扫地机器人保修多久") is None, "售后 guard 拦截")
    _assert(service.sops.base.match_sop("不染一尘有什么优势") is None, "品牌 guard 拦截")


# ──────────────────────────────────────────────────────────────
# 2. 会话生命周期
# ──────────────────────────────────────────────────────────────

def test_start_and_continue():
    _cleanup()
    reply, done = _pair(service.sops.base.start_sop("s1", "sop_unit_test", "帮我看看"))
    _assert_eq(reply, "预算多少？", "首轮返回 ask")
    _assert(not done, "未结束")
    _assert(service.sops.base.has_active_sop("s1"), "有活跃会话")
    reply, done = _pair(service.sops.base.continue_sop("s1", "1000以内"))
    _assert_eq(reply, "有宠物吗？", "推进到 ask_pet")
    _cleanup()


def test_continue_no_session():
    _cleanup()
    _assert_eq(service.sops.base.continue_sop("s1", "随便").kind, "noop", "无会话→noop")


def test_end_sop():
    _cleanup()
    service.sops.base.start_sop("s1", "sop_unit_test", "x")
    _assert(service.sops.base.has_active_sop("s1"), "启动后有会话")
    service.sops.base.end_sop("s1")
    _assert(not service.sops.base.has_active_sop("s1"), "结束后无会话")
    _assert_eq(service.sops.base.get_active_sop_id("s1"), None, "结束后 id=None")


def test_full_flow():
    _cleanup()
    service.sops.base.start_sop("s1", "sop_unit_test", "x")
    service.sops.base.continue_sop("s1", "1000以内")
    reply, done = _pair(service.sops.base.continue_sop("s1", "没有"))
    _assert(done, "SOP 结束")
    _assert_in("共 2 款", reply, "1000以内命中 2 款")
    _assert_in("型号A", reply, "含型号A")
    _assert_in("型号B", reply, "含型号B")
    _assert_not_in("型号C", reply, "不含超预算型号C")
    _assert(not service.sops.base.has_active_sop("s1"), "会话已结束")
    _cleanup()


# ──────────────────────────────────────────────────────────────
# 3. 提取 / 重试 / 放弃 / fallback
# ──────────────────────────────────────────────────────────────

def test_extract_budget_bare_number():
    from service.sops.model_query_sop import _extract_budget
    _assert_eq(_extract_budget("800块以内", {}).value, (None, 800), "带量词照旧走通用解析")
    _assert_eq(_extract_budget("2000左右", {}).kind, "done", "区间照旧走通用解析")
    _assert_eq(_extract_budget("800", {}).value, (None, 800), "纯数字短答按上限收（BC-20261015-03）")
    _assert_eq(_extract_budget("预算 5000 元", {}).value, (None, 5000), "带「预算」照旧")
    _assert_eq(_extract_budget("不知道", {}).kind, "fail", "非预算表达→fail")
    _assert_eq(_extract_budget("跳过", {}).kind, "skip", "显式跳过→skip")


def test_result_reply():
    from config.word_dict_config import MODEL_QUERY_EMPTY_REPLIES
    from service.sops.model_query_sop import MODEL_QUERY_SOP
    reply = MODEL_QUERY_SOP["steps"][-1]["template"]
    full = reply({"count": 2, "list": "- A\n- B"})
    _assert("在您选择的条件内" in full and "对比其中两款" in full, "有结果时给出列表并追问对比")
    empty = reply({"count": 0, "list": "", "models": []})
    _assert_eq(empty, MODEL_QUERY_EMPTY_REPLIES[0], "0 款时使用唯一的空结果话术")
    _assert(not re.search(r"\d", empty), "0 款时不给价格数字（没有依据的推荐）")
    _assert("对比" not in empty, "0 款时不追问对比")


def test_search_rejects_malformed_slots():
    """预算/时间槽形状不对或重试耗尽时，不能静默按无条件筛选。"""
    import service.sops.model_query_sop as mq
    from config.word_dict_config import MODEL_QUERY_CONSTRAINT_RETRY_REPLY

    from tools.dto.model_query import FindModelsResult, ModelQueryConstraints
    saved = mq.find_models_result
    calls = []
    mq.find_models_result = lambda args: calls.append(args) or FindModelsResult(
        [], [], [], ModelQueryConstraints())
    try:
        out = mq._search({"budget": "3000以内"})
        _assert_eq(out.get("error"), "invalid_slots", "字符串预算不能被切片后照常查询")
        _assert("budget" in out.get("dropped", []), "记录被丢弃的槽位")
        _assert_eq(calls, [], "非法槽位不调用 find_models")

        out = mq._search({"budget": (None, 3000)})
        _assert(not out.get("error"), "正常二元组预算照常查询")
        _assert_eq(calls[-1].get("budget_max"), 3000, "预算上限传入")

        out = mq._search({"budget": None, "_slot_status": {"budget": "dropped"}})
        _assert_eq(out.get("error"), "invalid_slots", "重试耗尽也走显式失败")
        _assert_eq(mq.MODEL_QUERY_SOP["steps"][-1]["template"](out),
                   MODEL_QUERY_CONSTRAINT_RETRY_REPLY,
                   "失败回复要求用户重新说明条件")

        out = mq._search({"_slot_status": {"series": "dropped"}})
        _assert_eq(out.get("error"), "invalid_slots", "系列重试耗尽也不能静默跳过")
        out = mq._search({"publish_range": "2025"})
        _assert_eq(out.get("error"), "invalid_slots", "日期形状错误不能静默跳过")
    finally:
        mq.find_models_result = saved


def test_retry_then_abandon():
    _cleanup()
    # start_sop 的触发词会先尝试一次提取（失败 → retry=1 → ask 话术）
    r = _pair(service.sops.base.start_sop("s1", "sop_unit_test", "x"))[0]
    _assert_eq(r, "预算多少？", "触发词提取失败→ask")
    # 第 2 次失败（retry=2）→ retry 话术
    r = _pair(service.sops.base.continue_sop("s1", "不知道"))[0]
    _assert_eq(r, "预算没听清，说个数字。", "重试失败→retry")
    # 第 3 次失败（retry=3 > max_retry=2）→ 放弃预算槽位推进到下一问
    r = _pair(service.sops.base.continue_sop("s1", "还是不知道"))[0]
    _assert_eq(r, "有宠物吗？", "超重试→放弃预算槽位")
    state = service.sops.base.get_sop_state("s1")
    _assert_eq(state.get("step"), 1, "已推进到第 2 步")
    _assert_eq(state.get("retry_count"), 0, "跨 step 时 retry_count 清零，不继承上一槽")
    _assert_eq((state.get("slot_status") or {}).get("budget"), "dropped",
               "重试耗尽要标记 dropped，不能和用户主动跳过混淆")
    # 第 2 步第一次失败：从 0 重新计数，而不是接着第 1 步的 3
    r = _pair(service.sops.base.continue_sop("s1", "不知道"))[0]
    _assert_eq(r, "有宠物吗？", "第 2 步首次失败仍用 ask 话术")
    state = service.sops.base.get_sop_state("s1")
    _assert_eq(state.get("retry_count"), 1, "第 2 步 retry 从 1 开始")
    _cleanup()


def test_model_query_dropped_budget_does_not_return_all():
    """预算重试耗尽后进入 action：不能把预算当没提过，返回全量列表。"""
    from config.word_dict_config import MODEL_QUERY_CONSTRAINT_RETRY_REPLY

    _cleanup()
    try:
        service.sops.base._save_session("s-dropped", service.sops.base.make_session(
            "model_query", step=4, slots={"budget": None},
            retry_count=0))
        state = service.sops.base.get_sop_state("s-dropped")
        state["slot_status"] = {"budget": "dropped"}
        service.sops.base._save_session("s-dropped", state)
        result = service.sops.base.continue_sop("s-dropped", "继续")
        _assert(result.done, "失败约束也结束流程")
        _assert_eq(result.reply, MODEL_QUERY_CONSTRAINT_RETRY_REPLY,
                   "重试耗尽后要求用户重新说明条件")
    finally:
        _cleanup()


def test_template_fallback():
    _cleanup()
    service.sops.base.register(_BAD_SOP)
    service.sops.base.start_sop("s1", "sop_bad_template", "")
    reply, done = _pair(service.sops.base.continue_sop("s1", "张三"))
    _assert(done, "SOP 结束")
    _assert_eq(reply, "抱歉，出了点小问题。", "缺字段触发 fallback")
    _cleanup()


def test_missing_ask_placeholder_uses_retry():
    """ask 模板占位符缺失：生产模式回退 retry，不把 $missing 发给用户。"""
    sop_id = "sop_missing_ask"
    service.sops.base.register({
        "id": sop_id,
        "steps": [{
            "id": "ask_name", "type": "ask", "slot": "name",
            "ask": "$missing", "retry": "请再说一次名字～",
            "extract": lambda text, slots: ExtractResult("fail"),
        }],
    })
    try:
        result = service.sops.base.start_sop("s-missing-ask", sop_id, "x")
        _assert_eq(result.reply, "请再说一次名字～", "ask 缺参回退 retry")
        _assert_not_in("{", result.reply, "不回显未替换占位符")
        _assert_eq(result.kind, "ask", "仍在 ask 状态")
    finally:
        service.sops.base.end_sop("s-missing-ask")
        service.sops.base.SOPS.pop(sop_id, None)


def test_missing_reply_placeholder_uses_fallback():
    """reply 模板占位符缺失：生产模式走步骤 fallback，不抛异常。"""
    sop_id = "sop_missing_reply"
    service.sops.base.register({
        "id": sop_id,
        "steps": [{
            "id": "reply", "type": "reply",
            "template": "$missing", "fallback": "兜底回复",
        }],
    })
    try:
        result = service.sops.base.start_sop("s-missing-reply", sop_id, "x")
        _assert_eq(result.reply, "兜底回复", "reply 缺参走 fallback")
        _assert_not_in("{", result.reply, "不回显未替换占位符")
        _assert(result.done, "reply 仍结束 SOP")
    finally:
        service.sops.base.end_sop("s-missing-reply")
        service.sops.base.SOPS.pop(sop_id, None)


def test_empty_reply_default_fallback():
    """没有步骤级 fallback 时，空 reply 使用统一兜底话术。"""
    from config.word_dict_config import SOP_EMPTY_REPLY_FALLBACK

    sop_id = "sop_empty_default"
    service.sops.base.register({
        "id": sop_id,
        "steps": [{"id": "reply", "type": "reply", "template": lambda ctx: ""}],
    })
    try:
        reply, done = _pair(service.sops.base.start_sop("s-empty", sop_id, ""))
        _assert(done, "SOP 结束")
        _assert_eq(reply, SOP_EMPTY_REPLY_FALLBACK, "无步骤 fallback 时使用统一兜底")
    finally:
        service.sops.base.SOPS.pop(sop_id, None)
        service.sops.base.end_sop("s-empty")


def test_unknown_sop_id_dropped():
    """会话里记的 sop_id 已不存在（SOP 被改名/删掉）→ 状态作废、按「没有活跃 SOP」处理，不再抛 KeyError。"""
    _cleanup()
    service.sops.base.register(_TEST_SOP)
    service.sops.base._start("s1", "nonexistent_sop")
    _assert(not service.sops.base.has_active_sop("s1"), "旧状态被作废")
    _assert_eq(service.sops.base.continue_sop("s1", None).kind, "noop",
               "continue_sop 返回 noop，本轮回落普通作答")
    _cleanup()


def test_stale_def_state_dropped():
    """代码改过 SOP 定义（步骤/槽位变了）→ Redis 里的旧状态作废，不拿旧下标续跑。"""
    _cleanup()
    service.sops.base.register(_TEST_SOP)
    service.sops.base.start_sop("s1", "sop_unit_test", "")
    _assert(service.sops.base.has_active_sop("s1"), "先有活跃状态")
    # 模拟一次发版：同一 sop_id 的定义多出一步 → 指纹变
    changed = dict(_TEST_SOP, steps=list(_TEST_SOP["steps"]) +
                   [{"id": "extra", "type": "reply", "template": "补充说明"}])
    service.sops.base.register(changed)
    _assert(not service.sops.base.has_active_sop("s1"), "定义变了 → 旧状态作废")
    _assert(service.sops.base._load_session("s1") is None, "Redis 里那份也已被删掉")
    _assert_eq(service.sops.base.continue_sop("s1", "1000以内").kind, "noop",
               "不再拿旧下标续跑")
    service.sops.base.register(_TEST_SOP)          # 还原注册表，别影响后面的用例
    _cleanup()


def test_injected_state_keeps_fingerprint():
    """测试注入的状态（test_scripts.test_hooks.set_active_sop）带指纹，能正常续跑——行为轨的 stateful 用例靠它。"""
    _cleanup()
    service.sops.base.register(_TEST_SOP)
    from test_scripts.test_hooks import set_active_sop
    set_active_sop("s1", "sop_unit_test", step=1, slots={"budget": [None, 1000]})
    _assert(service.sops.base.has_active_sop("s1"), "注入的状态没被当成过期")
    reply, done = _pair(service.sops.base.continue_sop("s1", "没有"))
    _assert(done, "答完剩下槽位后一轮走完 action + reply")
    _assert_in("型号A", reply, "回复里带推荐结果")
    _cleanup()


def test_stale_def_state_recorded_in_trace():
    """作废要留在 trace 里（AGENTS：降级必须 WARNING + trace）：sop.active / skipped / by=stale_def。"""
    _cleanup()
    service.sops.base.register(_TEST_SOP)
    service.sops.base.start_sop("s1", "sop_unit_test", "")
    reset_trace("s1")
    trace_store.begin_turn("s1", "1000以内")
    # 发版：定义多一步 → 读状态时作废（这一步落节点）
    service.sops.base.register(dict(_TEST_SOP, steps=list(_TEST_SOP["steps"]) +
                            [{"id": "extra", "type": "reply", "template": "补充说明"}]))
    _assert(not service.sops.base.has_active_sop("s1"), "状态被作废")
    trace_store.end_turn(reply="（单测）")
    rows = trace_reader.load(trace_store._path("s1"))      # 写入侧那条路径；别按会话名捞（会捞到旧日期的目录）
    path = (rows[0].get("path") or []) if rows else []
    _assert(any(n.get("node") == "sop.active" and n.get("status") == "skipped"
                and n.get("by") == "stale_def" for n in path),
            "trace 里有 sop.active / skipped / by=stale_def")
    service.sops.base.register(_TEST_SOP)
    _cleanup()


def test_local_session_expires_without_redis():
    """无 Redis 的降级路径按 sop_ttl 滑动过期（Redis 路径由 Redis 自己过期，不走这段）。"""
    _cleanup()
    from gateway import redis_store
    service.sops.base.register(_TEST_SOP)
    real = redis_store.get_redis
    redis_store.get_redis = lambda: None          # 强制走降级分支
    try:
        service.sops.base.start_sop("s1", "sop_unit_test", "")
        _assert(service.sops.base.has_active_sop("s1"), "刚写的降级态可用")
        _assert("ts" in service.sops.base._sessions["s1"], "降级态带 ts（Redis 路径不带）")
        service.sops.base._sessions["s1"]["ts"] = time.time() - 100000      # 手动拨旧（sop_ttl=900）
        _assert(not service.sops.base.has_active_sop("s1"), "超过 sop_ttl → 作废")
        _assert_eq(service.sops.base._load_session("s1"), None, "降级态那份也删了")
    finally:
        redis_store.get_redis = real
        service.sops.base._sessions.pop("s1", None)
    _cleanup()


def test_step_exception_ends_sop():
    """步骤函数抛异常：向上抛（外层兜底）+ 结束会话（不留半截状态、下轮不重跑同一步）+ trace 留痕。"""
    _cleanup()
    service.sops.base.register(_BOOM_SOP)
    reset_trace("s1")
    trace_store.begin_turn("s1", "随便")
    _assert_raises(RuntimeError, lambda: service.sops.base.start_sop("s1", "sop_boom_action", "随便"),
                   "步骤异常照常向上抛")
    _assert(not service.sops.base.has_active_sop("s1"), "异常后会话已结束（不会被 finally 写回去）")
    trace_store.end_turn()
    rows = trace_reader.load(trace_store._path("s1"))      # 同写入侧路径（见上一条用例的说明）
    path = (rows[0].get("path") or []) if rows else []
    node = next((n for n in path if n.get("node") == "sop.step_error"), None)
    _assert(node is not None, "trace 里有 sop.step_error 节点（能看出是第几步、哪个步骤炸的）")
    _assert_eq((node or {}).get("status"), "error", "节点状态是 error")
    _assert_eq(((node or {}).get("in") or {}).get("type"), "action", "in 带失败步骤的类型")
    _assert_eq(service.sops.base.continue_sop("s1", "再说一遍").kind, "noop",
               "下一轮不再停在同一步重复失败")
    _cleanup()


# ──────────────────────────────────────────────────────────────
# 4. 选购 SOP 全流程（真实 enumerate_models，数据可用时）
# ──────────────────────────────────────────────────────────────

def test_model_query_sop_full_flow():
    if not _models_ready():
        _skip("无型号数据（Chroma/pkl 缓存缺失）")
        return
    _cleanup()
    sid = "model_query_flow_test"
    reply, done = _pair(service.sops.base.start_sop(sid, "model_query", "帮我推荐一款"))
    _assert(service.sops.base.has_active_sop(sid), "model_query SOP 启动")
    _assert_in("预算", reply, "开场问预算")

    reply, done = _pair(service.sops.base.continue_sop(sid, "1000以内"))
    _assert(not done, "预算后未结束")
    _assert_in("系列", reply, "预算后问系列")

    reply, done = _pair(service.sops.base.continue_sop(sid, "跳过"))
    _assert_in("发布时间", reply, "系列跳过后问时间")
    reply, done = _pair(service.sops.base.continue_sop(sid, "跳过"))
    _assert_in("特点", reply, "时间跳过后问标签")
    reply, done = _pair(service.sops.base.continue_sop(sid, "跳过"))
    _assert(done, "标签跳过后 SOP 结束")
    _assert_in("在您选择的条件内", reply, "推荐含统计")
    _assert_in("净白 S1", reply, "1000以内含净白 S1(899)")
    _assert_not_in("净界 P1", reply, "1000以内不含净界 P1(1799)")
    _assert(not service.sops.base.has_active_sop(sid), "会话已结束")
    _cleanup()


TESTS = [
    test_register_and_match,
    test_sop_name_from_definition,
    test_register_validation,
    test_extract_done_skip_and_drop_are_distinct,
    test_action_non_dict_is_explicit_error,
    test_non_model_action_has_no_n_models,
    test_reenter_hint_key_uses_word_dict,
    test_model_query_trigger_and_guards,
    test_start_and_continue,
    test_continue_no_session,
    test_end_sop,
    test_full_flow,
    test_extract_budget_bare_number,
    test_result_reply,
    test_search_rejects_malformed_slots,
    test_retry_then_abandon,
    test_model_query_dropped_budget_does_not_return_all,
    test_template_fallback,
    test_missing_ask_placeholder_uses_retry,
    test_missing_reply_placeholder_uses_fallback,
    test_empty_reply_default_fallback,
    test_unknown_sop_id_dropped,
    test_stale_def_state_dropped,
    test_injected_state_keeps_fingerprint,
    test_stale_def_state_recorded_in_trace,
    test_step_exception_ends_sop,
    test_local_session_expires_without_redis,
    test_model_query_sop_full_flow,
]


def run():
    reset()
    return run_tests("选购/型号查询 SOP 测试（service.sops.base + model_query）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
