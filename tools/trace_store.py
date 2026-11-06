"""trace 公共入口（薄壳）：实现拆在 repo/trace/{paths,write,read,sidecar}。

这里只 re-export 同一个函数/状态对象，不复制 `_local`、缓存或锁；
调用方继续 `from tools import trace_store`，测试仍可用 `TRACE_DIR` 隔离。
"""
from tools.log_tool import log_path

from repo.trace.paths import (
    _DATE_DIR,
    _dir_cache,
    _path,
    _root_dir,
    _session_dir,
    _stem_cache,
    _stem_of,
    _today,
    day_files,
    session_dirs,
    _date_of_file,
)
from repo.trace.read import find_turn, find_turn_by_index, latest_turn
from repo.trace.facts import to_fact, to_facts
from repo.trace.sidecar import (
    CHUNK_KIND,
    CHUNK_MAX_CHARS,
    EVENT_KIND,
    FEEDBACK_KIND,
    FEEDBACK_LOOKBACK_DAYS,
    SHADOW_KIND,
    SIDE_VERSION,
    _append,
    _event,
    effective_feedbacks,
    event,
    events_of,
    feedback_of,
    feedback_turn,
    shadow_turn,
)
from repo.trace.write import (
    TRACE_VERSION,
    _STEP_NAMES,
    _NODE_OF_STEP,
    _ROUTE_MIRRORED,
    _last,
    _local,
    _lock,
    _scalars,
    _seed,
    _seq,
    _t0,
    _finalize,
    add_ms,
    begin_span,
    begin_turn,
    end_span,
    brief,
    chunk_brief,
    chunk_records,
    chunks,
    current_turn,
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
    _as_triple,
    _num,
)
