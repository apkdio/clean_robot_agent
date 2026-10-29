"""一轮对话的旁路记录器：行为标签、trace 节点与影子探针的统一入口。

主链路只调用这里，不直接散落 trace_store 调用；记录失败绝不影响对话。
"""
from __future__ import annotations

import os

from tools import trace_store
from tools.log_tool import get_logger

logger = get_logger(name="turn_recorder")


def log_behavior(tag: str, **detail) -> None:
    """行为观测点：把这一轮「处置成哪一类行为」记一行，供行为轨评测与影子对照读取。"""
    logger.info("[Behavior] %s%s", tag, "".join(" %s=%s" % (k, v) for k, v in detail.items()))
    trace_store.note_behavior(tag, detail)


def shadow_probe(query: str, session_id: str, chain_category: str,
                 chain_detail: str = "") -> None:
    """影子模式：让模型并行做一次决策，与链路实际走的类目对照，只落日志、不改行为。"""
    from function_tools.registry import orchestration_mode, shadow_sample
    from tools.orchestrator import chain_fc_eligible, decide, shadow_line, shadow_record

    if orchestration_mode() != "shadow" or not chain_fc_eligible(chain_category):
        return
    import random

    if random.random() > shadow_sample():
        return

    ctx = trace_store.current_turn()
    shadow_path = trace_store._path(session_id, trace_store.SHADOW_KIND, create=True)
    trace_store.step("shadow", chain=chain_category, detail=chain_detail,
                     log=os.path.basename(shadow_path) if shadow_path else "")

    def _work():
        try:
            from config.word_dict_config import DOMAIN_FILES
            from tools.domain_router import domain_scores
            from tools.query_rewrite import history_block

            scores = domain_scores(query)
            domain_hint = scores[0][0] if scores and scores[0][0] in DOMAIN_FILES else None
            decision = decide(query, history_block=history_block(session_id), domain_hint=domain_hint)
            logger.info(shadow_line(decision, query, chain_category, chain_detail))
            trace_store.shadow_turn(ctx, shadow_record(decision, query, chain_category, chain_detail))
        except Exception as e:  # noqa: BLE001
            logger.warning("[Shadow] probe failed: %s", e)

    import threading

    threading.Thread(target=_work, name="shadow-probe", daemon=True).start()


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

    def shadow(self, query: str, session_id: str, chain_category: str,
               chain_detail: str = "") -> None:
        shadow_probe(query, session_id, chain_category, chain_detail)
