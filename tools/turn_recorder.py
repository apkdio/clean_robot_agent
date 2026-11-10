"""一轮对话的旁路记录器：行为标签与 trace 节点的统一入口。

主链路只调用这里，不直接散落 trace_store 调用；记录失败绝不影响对话。
AI 工具编排已移出主链，作为独立实验子项目维护，本模块不再承载 shadow 探针。
"""
from __future__ import annotations

from tools import trace_store
from tools.log_tool import get_logger

logger = get_logger(name="turn_recorder")


def log_behavior(tag: str, **detail) -> None:
    """行为观测点：把这一轮「处置成哪一类行为」记一行，供行为轨评测读取。"""
    logger.info("[Behavior] %s%s", tag, "".join(" %s=%s" % (k, v) for k, v in detail.items()))
    trace_store.note_behavior(tag, detail)


class TurnRecorder:
    """本轮的记录门面：只包 trace_store，不另存一份状态。"""

    def route(self, name: str, **facts) -> None:
        trace_store.route(name, **facts)

    def node(self, name: str, status: str = "ok", by: str = None,
             in_: dict = None, out: dict = None) -> None:
        trace_store.node(name, status=status, by=by, in_=in_, out=out)

    def step(self, name: str, **fields) -> None:
        trace_store.step(name, **fields)

    def retrieval(self, **facts) -> None:
        trace_store.retrieval(**facts)

    def behavior(self, tag: str, **detail) -> None:
        log_behavior(tag, **detail)
