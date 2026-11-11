"""trace 公共入口（薄壳）：实现拆在 mapper/trace/{paths,write,read,sidecar}。

这里只 re-export 公共函数与稳定路径入口，不复制 `_local`、缓存或锁；
调用方统一使用 `mapper.trace_store`，测试仍可用 `TRACE_DIR` 隔离。
"""
from utils.log_tool import log_path

from mapper.trace.paths import (
    _path,
    _root_dir,
    _session_dir,
    session_dirs,
)
from mapper.trace.read import find_turn, find_turn_by_index, latest_turn
from mapper.trace.sidecar import (
    CHUNK_KIND,
    CHUNK_MAX_CHARS,
    EVENT_KIND,
    FEEDBACK_KIND,
    FEEDBACK_LOOKBACK_DAYS,
    effective_feedbacks,
    event,
    events_of,
    feedback_of,
    feedback_turn,
)
from mapper.trace.write import (
    TRACE_VERSION,
    add_ms,
    begin_span,
    begin_turn,
    brief,
    chunk_brief,
    chunks,
    end_span,
    end_turn,
    note_behavior,
    note_error,
    node,
    reset_sequence,
    retrieval,
    route,
    stage,
    step,
    turn_anchor,
    verify,
)

__all__ = [
    "log_path",
    "CHUNK_KIND", "CHUNK_MAX_CHARS", "EVENT_KIND", "FEEDBACK_KIND",
    "FEEDBACK_LOOKBACK_DAYS", "TRACE_VERSION",
    "find_turn", "find_turn_by_index", "latest_turn",
    "effective_feedbacks", "event", "events_of", "feedback_of", "feedback_turn",
    "add_ms", "begin_span", "begin_turn", "brief", "chunk_brief", "chunks",
    "end_span", "end_turn", "note_behavior", "note_error", "node",
    "reset_sequence", "retrieval", "route", "stage", "step", "turn_anchor", "verify",
    "session_dirs",
]
