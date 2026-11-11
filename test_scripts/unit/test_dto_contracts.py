"""DTO 契约测试：构造校验、闭集与“领域结果不进传输层”的边界。"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *

from utils.dto.clarify import CLARIFY_DONE, CLARIFY_DROP, CLARIFY_RESUME, ClarifyResult
from utils.dto.executor import NotMine, Reply
from utils.dto.handoff import Handoff
from utils.dto.model_query import FindModelsResult, ModelQueryConstraints
from utils.dto.feedback import FeedbackRequest, FeedbackResult
from utils.dto.knowledge import IngestResult, KnowledgeStats, ResetResult
from utils.dto.request import ChatRequest
from utils.dto.sop import ExtractResult, SopStepResult
from utils.dto.stage import Continue, Finish
from utils.dto.transport import TurnEvent


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
    handoff = Handoff("retrieve", "机器人不移动", "repair")
    _assert_eq(Reply(handoff=handoff).handoff.target, "retrieve", "Reply 可带 handoff")
    _assert(isinstance(NotMine(), NotMine), "NotMine 是显式类型")
    _assert_raises(ValueError, lambda: Reply(None), "Reply.text 必须是 str")
    _assert_raises(ValueError, lambda: Handoff(""), "Handoff.target 不能为空")


def test_clarify_result():
    _assert_eq(ClarifyResult.done("好的").code, CLARIFY_DONE, "done 形态")
    handoff = Handoff("retrieve", "机器人不移动", "repair")
    _assert_eq(ClarifyResult.done_handoff(handoff).handoff.target, "retrieve",
               "done 可带 handoff")
    _assert_eq(ClarifyResult.drop().code, CLARIFY_DROP, "drop 形态")
    resumed = ClarifyResult.resume("原句", prefix="先说",
                                   prefer_op="followup", enter_declined=True)
    _assert_eq((resumed.code, resumed.message, resumed.data.query,
                resumed.data.prefer_op, resumed.data.enter_declined),
               (CLARIFY_RESUME, "先说", "原句", "followup", True),
               "resume 载荷原样带出")
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
    import service.agent as agent_mod
    from service.turn_context import TurnContext

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
    handoff = Handoff("retrieve", "机器人不移动", "repair")
    _assert_eq(SopStepResult("reply", done=True, handoff=handoff).handoff.target,
               "retrieve", "reply 可带 handoff")
    _assert_true(SopStepResult("noop").reply == "", "noop 无话术")
    _assert_raises(ValueError, lambda: SopStepResult("bad"), "未知 kind 报错")
    _assert_raises(ValueError, lambda: SopStepResult("ask", done=True), "ask 不能 done")
    _assert_raises(ValueError, lambda: SopStepResult("ask", handoff=handoff),
                   "ask 不能带 handoff")
    _assert_raises(ValueError, lambda: SopStepResult("noop", "x"), "noop 不能带话术")


def test_extract_result():
    _assert_eq(ExtractResult("done", "x").value, "x", "done 带值")
    _assert_eq(ExtractResult("skip").kind, "skip", "skip 形态")
    _assert_eq(ExtractResult("fail", reason="no_x").reason, "no_x", "fail 带原因")
    _assert_raises(ValueError, lambda: ExtractResult("bad"), "未知 kind 报错")
    _assert_raises(ValueError, lambda: ExtractResult("done", extra=[]), "extra 必须是 dict")


def test_model_query_result():
    r = FindModelsResult(models=[{"name": "X"}], kind="budget",
                         constraints=ModelQueryConstraints())
    _assert_eq(r.kind, "budget", "模型查询 kind")
    _assert_eq(len(r.models), 1, "模型查询结果")


def test_run_stages_rejects_unknown_outcome():
    """阶段返回体不认识时，主链要显式报错，不能当 Continue 继续跑。"""
    import service.agent as agent_mod
    from service.turn_context import TurnContext

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


def test_feedback_request_and_result():
    req = FeedbackRequest(
        "边刷多久换一次", "sid", "useless",
        reason="kb_gap", note="没讲清楚", date="2026-11-11", turn_index=2,
    )
    _assert_eq(req.reason, "kb_gap", "请求体保留无用原因")
    _assert_eq(req.turn_index, 2, "请求体保留轮次")
    _assert_raises(ValueError, lambda: FeedbackRequest("", "sid", "useful"), "空 query 报错")
    _assert_raises(ValueError, lambda: FeedbackRequest("q", "", "useful"), "空 session_id 报错")
    _assert_raises(ValueError, lambda: FeedbackRequest("q", "sid", None), "rating 类型错误")
    _assert_raises(ValueError, lambda: FeedbackRequest("q", "sid", "useful", reason=None),
                   "reason 类型错误")
    _assert_raises(ValueError, lambda: FeedbackRequest("q", "sid", "useful", date="2026-11-11"),
                   "date 与 turn_index 必须成对")
    _assert_raises(ValueError, lambda: FeedbackRequest("q", "sid", "useful", turn_index=2),
                   "date 与 turn_index 必须成对")
    _assert_raises(ValueError, lambda: FeedbackRequest("q", "sid", "useful", turn_index=True),
                   "turn_index 不接受 bool")

    result = FeedbackResult("ok", date="2026-11-11", turn_index=2, updated=True)
    _assert_eq(result.payload()["turn_index"], 2, "结果回传轮次")
    _assert_eq(FeedbackResult("error", message="bad").payload(),
               {"status": "error", "message": "bad"}, "错误结果不带 success 字段")
    _assert_raises(ValueError, lambda: FeedbackResult("unknown"), "结果状态必须在闭集内")


def test_knowledge_results():
    _assert_eq(KnowledgeStats("kb", 3).payload(),
               {"collection_name": "kb", "chunk_count": 3}, "统计结果保留字段")
    _assert_eq(IngestResult("ok", file="a.md", chunks=2, sha256="x", elapsed=0.1).payload(),
               {"status": "ok", "file": "a.md", "chunks": 2, "sha256": "x", "elapsed": 0.1},
               "成功入库结果")
    _assert_eq(IngestResult("error", file="a.md", message="bad").payload(),
               {"status": "error", "file": "a.md", "message": "bad"},
               "失败入库结果")
    _assert_eq(ResetResult("busy").payload(), {"status": "busy"}, "busy 不再用 None 哨兵")
    _assert_eq(ResetResult("ok", deleted=4).payload(),
               {"status": "ok", "deleted": 4}, "重置结果")
    _assert_raises(ValueError, lambda: IngestResult("unknown"), "入库状态闭集")
    _assert_raises(ValueError, lambda: ResetResult("unknown"), "重置状态闭集")


def test_stage_registry():
    """阶段注册表：id/node 唯一、实现可调用、节点标签已登记。"""
    from config.word_dict_config import TRACE_NODE_LABELS
    from service.agent import _STAGES

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
    test_extract_result,
    test_model_query_result,
    test_chat_request,
    test_feedback_request_and_result,
    test_knowledge_results,
    test_stage_registry,
]


def run():
    reset()
    return run_tests("DTO 契约测试（控制流 / 执行体 / 澄清 / 传输）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
