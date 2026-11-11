"""阶段 handoff 契约：只允许前向跳，跳过的阶段可解释，retrieve 真正使用 domain。"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *

from config.word_dict_config import domain_filter
from service.agent import StageSpec, _run_stages, _stage_retrieve
from utils.dto.executor import Reply
from utils.dto.handoff import Handoff
from utils.dto.request import ChatRequest
from utils.dto.stage import Continue, Finish
from service.turn_context import TurnContext


def _ctx():
    return TurnContext.from_request(ChatRequest("继续", "t-handoff"))


def test_run_stages_forward_handoff_skips_middle():
    import service.agent as agent_mod

    saved = agent_mod._STAGES
    calls = []
    agent_mod._STAGES = (
        StageSpec("a", "stage.a", lambda ctx: Continue(handoff=Handoff("c"))),
        StageSpec("b", "stage.b", lambda ctx: calls.append("b") or Continue()),
        StageSpec("c", "stage.c", lambda ctx: calls.append("c") or Finish(text="ok")),
    )
    try:
        out = list(_run_stages(_ctx()))
    finally:
        agent_mod._STAGES = saved
    _assert_eq(out, ["ok"], "跳到 c 并输出结果")
    _assert_eq(calls, ["c"], "b 被跳过，不能执行")


def test_run_stages_rejects_backward_handoff():
    import service.agent as agent_mod

    saved = agent_mod._STAGES
    agent_mod._STAGES = (
        StageSpec("a", "stage.a", lambda ctx: Continue(handoff=Handoff("a"))),
        StageSpec("b", "stage.b", lambda ctx: Continue()),
    )
    try:
        _assert_raises(ValueError, lambda: list(_run_stages(_ctx())),
                       "同阶段/回跳必须报错")
    finally:
        agent_mod._STAGES = saved


class _FakeRetriever:
    def __init__(self):
        self.filters = []

    def search_with_scores(self, query, filter=None, trace_out=None):
        self.filters.append(filter)
        return []


def test_retrieve_uses_handoff_domain_and_query():
    import service.agent as agent_mod

    fake = _FakeRetriever()
    saved_get = agent_mod._get_retriever
    agent_mod._get_retriever = lambda: fake
    ctx = _ctx()
    ctx.handoff = Handoff("retrieve", "机器人不移动怎么办", "repair")
    try:
        _stage_retrieve(ctx)
    finally:
        agent_mod._get_retriever = saved_get
    _assert_eq(fake.filters, [{"file_name": domain_filter("repair")}],
               "handoff 的 query/domain 进入 retrieve")


def test_generate_uses_handoff_query():
    """clarify 答“是”后，retrieve 与 generate/verify 必须使用同一个 handoff query。"""
    import service.agent as agent_mod

    captured = {}
    saved = agent_mod.rag_stream

    def fake_stream(query, **_kwargs):
        captured["query"] = query
        return iter(())

    agent_mod.rag_stream = fake_stream
    try:
        ctx = _ctx()
        ctx.handoff = Handoff("retrieve", "机器人不移动怎么办", "repair")
        ctx.chunks = [type("Chunk", (), {"page_content": "证据"})()]
        outcome = agent_mod._stage_generate(ctx)
        list(outcome.stream)
    finally:
        agent_mod.rag_stream = saved
    _assert_eq(captured["query"], "机器人不移动怎么办",
               "generate 使用 handoff query，而不是确认句“是”")


def test_ops_dispatch_propagates_handoff():
    import service.agent as agent_mod

    saved_exec, saved_pipeline = agent_mod._OP_EXEC, agent_mod._PIPELINE_OPS
    agent_mod._OP_EXEC = {"x": lambda ctx: Reply(handoff=Handoff("retrieve", "q", "repair"))}
    agent_mod._PIPELINE_OPS = ()
    try:
        ctx = _ctx()
        ctx.op_note = {"usable": ["x"]}
        outcome = agent_mod._stage_ops_dispatch(ctx)
    finally:
        agent_mod._OP_EXEC, agent_mod._PIPELINE_OPS = saved_exec, saved_pipeline
    _assert(isinstance(outcome, Continue), "ops 返回 Continue")
    _assert_eq(outcome.handoff.target, "retrieve", "handoff 透传")
    _assert_eq(ctx.handoff.target, "retrieve", "ctx 同步")


TESTS = [
    test_run_stages_forward_handoff_skips_middle,
    test_run_stages_rejects_backward_handoff,
    test_retrieve_uses_handoff_domain_and_query,
    test_generate_uses_handoff_query,
    test_ops_dispatch_propagates_handoff,
]


def run():
    return run_tests("阶段 handoff 契约", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
