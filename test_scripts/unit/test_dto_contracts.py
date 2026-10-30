"""DTO 契约测试：构造校验、闭集与“领域结果不进传输层”的边界。"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *

from tools.dto.clarify import CLARIFY_DONE, ClarifyResult
from tools.dto.executor import NotMine, Reply
from tools.dto.model_query import ModelQueryResult
from tools.dto.request import ChatRequest
from tools.dto.sop import SopStepResult
from tools.dto.stage import Continue, Finish
from tools.dto.transport import TurnEvent


def test_stage_outcome():
    _assert_eq(Continue(prefix="先说明").prefix, "先说明", "Continue 带前缀")
    _assert_eq(Finish.text_reply("答案").text, "答案", "Finish.text_reply")
    stream = Finish.stream_reply(iter(["x"])).stream
    _assert_true(hasattr(stream, "__next__"), "Finish.stream_reply")
    _assert_true(Finish.silent().text is None, "silent 不带话术")
    _assert_raises(ValueError, lambda: Finish(text="x", stream=iter(["y"])),
                   "text 与 stream 不能同时存在")


def test_exec_outcome():
    _assert_eq(Reply("答案").text, "答案", "Reply 带话术")
    _assert_eq(Reply().text, "", "Reply 可为空（接管但无话术）")
    _assert(isinstance(NotMine(), NotMine), "NotMine 是显式类型")
    _assert_raises(ValueError, lambda: Reply(None), "Reply.text 必须是 str")


def test_clarify_result():
    _assert_eq(ClarifyResult.done("好的").code, CLARIFY_DONE, "done 形态")
    _assert_raises(ValueError, lambda: ClarifyResult("bad"), "未知 code 报错")
    _assert_raises(ValueError, lambda: ClarifyResult.done(""), "done 空 message 报错")
    _assert_raises(ValueError, lambda: ClarifyResult.resume(""), "resume 空 query 报错")


def test_turn_event():
    _assert_eq(TurnEvent("delta", text="x").kind, "delta", "delta 事件")
    _assert_raises(ValueError, lambda: TurnEvent("delta"), "delta 必须有 text")
    _assert_raises(ValueError, lambda: TurnEvent("meta"), "meta 必须有 dict")
    _assert_eq(TurnEvent("done").kind, "done", "done 可无 text/meta")
    _assert_raises(ValueError, lambda: TurnEvent("bogus"), "未知 kind 不能构造")
    _assert_raises(ValueError, lambda: TurnEvent("delta", text=1), "text 必须是 str")


def test_ops_dispatch_rejects_unknown_outcome():
    """执行体返回非 Reply/NotMine 时，派发器要显式报契约错误。"""
    import tools.agent as agent_mod
    from tools.turn_context import TurnContext

    saved_exec, saved_pipeline = agent_mod._OP_EXEC, agent_mod._PIPELINE_OPS
    agent_mod._OP_EXEC = {"bad": lambda _ctx: "not a dto"}
    agent_mod._PIPELINE_OPS = ()
    try:
        ctx = TurnContext.from_request(ChatRequest("q", "t-dto"))
        ctx.op_note = {"usable": ["bad"]}
        _assert_raises(TypeError, lambda: agent_mod._stage_ops_dispatch(ctx),
                       "执行体返回错类型不能当 Reply 用")
    finally:
        agent_mod._OP_EXEC, agent_mod._PIPELINE_OPS = saved_exec, saved_pipeline


def test_sop_result():
    _assert_eq(SopStepResult("ask", "预算多少？").reply, "预算多少？", "ask 带话术")
    _assert_true(SopStepResult("reply", "结果", done=True).done, "reply 可标记 done")
    _assert_true(SopStepResult("noop").reply == "", "noop 无话术")
    _assert_raises(ValueError, lambda: SopStepResult("bad"), "未知 kind 报错")
    _assert_raises(ValueError, lambda: SopStepResult("ask", done=True), "ask 不能 done")
    _assert_raises(ValueError, lambda: SopStepResult("noop", "x"), "noop 不能带话术")


def test_model_query_result():
    r = ModelQueryResult(models=[{"name": "X"}], kind="budget", constraints={"budget_max": 1000})
    _assert_eq(r.kind, "budget", "模型查询 kind")
    _assert_eq(len(r.models), 1, "模型查询结果")


def test_run_stages_rejects_unknown_outcome():
    """阶段返回体不认识时，主链要显式报错，不能当 Continue 继续跑。"""
    import tools.agent as agent_mod
    from tools.turn_context import TurnContext

    saved = agent_mod._STAGES
    agent_mod._STAGES = (
        agent_mod.StageSpec("normalize", "stage.normalize",
                            lambda _ctx: "not an outcome"),
    )
    try:
        ctx = TurnContext.from_request(ChatRequest("q", "t-dto"))
        _assert_raises(TypeError,
                       lambda: list(agent_mod._run_stages(ctx)),
                       "未知阶段返回体必须报错")
    finally:
        agent_mod._STAGES = saved


def test_chat_request():
    _assert_eq(ChatRequest("q", "sid").query, "q", "请求体保留 query")
    _assert_raises(ValueError, lambda: ChatRequest("", "sid"), "空 query 报错")
    _assert_raises(ValueError, lambda: ChatRequest("q", ""), "空 session_id 报错")
    _assert_raises(ValueError, lambda: ChatRequest("q", "sid", lng=1.0),
                   "lng/lat 必须成对")
    _assert_raises(ValueError, lambda: ChatRequest("q", "sid", lng=181.0, lat=0.0),
                   "经度越界报错")
    _assert_raises(ValueError, lambda: ChatRequest("q", "sid", lng=0.0, lat=91.0),
                   "纬度越界报错")
    _assert_raises(ValueError, lambda: ChatRequest("q", "sid", lng=float("nan"), lat=0.0),
                   "非有限坐标报错")
    _assert_raises(ValueError, lambda: ChatRequest("q", "sid", lng=True, lat=0.0),
                   "坐标不接受 bool")
    _assert_raises(ValueError, lambda: ChatRequest("q", "sid", retry="yes"),
                   "retry 必须是 bool")


def test_stage_registry():
    """阶段注册表：id/node 唯一、实现可调用、节点标签已登记。"""
    from config.word_dict_config import TRACE_NODE_LABELS
    from tools.agent import _STAGES

    ids = [spec.id for spec in _STAGES]
    nodes = [spec.node for spec in _STAGES]
    _assert_eq(len(ids), len(set(ids)), "阶段 id 不重复")
    _assert_eq(len(nodes), len(set(nodes)), "阶段节点不重复")
    for spec in _STAGES:
        _assert_true(callable(spec.fn), "%s 的实现可调用" % spec.id)
        _assert_in(spec.node, TRACE_NODE_LABELS, "%s 已登记 trace 标签" % spec.node)


TESTS = [
    test_stage_outcome,
    test_exec_outcome,
    test_clarify_result,
    test_turn_event,
    test_ops_dispatch_rejects_unknown_outcome,
    test_run_stages_rejects_unknown_outcome,
    test_sop_result,
    test_model_query_result,
    test_chat_request,
    test_stage_registry,
]


def run():
    reset()
    return run_tests("DTO 契约测试（控制流 / 执行体 / 澄清 / 传输）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
