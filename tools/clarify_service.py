"""统一澄清服务：只保留状态认领、分派、载荷处理和登记机制。

具体业务（SOP 进入/退出、操作澄清）在各自模块注册 handler，本模块不 import 它们：
  - SOP 门：`sops/sop_gate.py`
  - 操作澄清：`tools/operations.py`
"""
from __future__ import annotations

import re
from collections.abc import Callable

from config.word_dict_config import OPERATION_LABELS as _OPERATION_LABELS
from tools import pending_store, trace_store
from tools.dto.clarify import ClarifyData, ClarifyResult
from tools.dto import clarify as _clarify_dto
from tools.log_tool import get_logger
from tools.selection import parse_index

logger = get_logger(name="clarify_service")

CLARIFY_DONE = _clarify_dto.CLARIFY_DONE
CLARIFY_RESUME = _clarify_dto.CLARIFY_RESUME
CLARIFY_DROP = _clarify_dto.CLARIFY_DROP
_CLARIFY_KINDS = _clarify_dto.CLARIFY_KINDS
_CONFIRM_CLEAN_RE = re.compile(r"[\s，,。.！!？?~～、]+")

_HANDLERS: dict[str, Callable] = {}
_BUILTINS_LOADED = False


def register_clarify_handler(biz: str, handler: Callable) -> None:
    """登记一个 biz 的处理器；重复登记由后注册者覆盖并留 WARNING。"""
    if not isinstance(biz, str) or not biz.strip():
        raise ValueError("clarify biz 不能为空")
    if not callable(handler):
        raise ValueError("clarify handler 必须可调用：%s" % biz)
    if biz in _HANDLERS and _HANDLERS[biz] is not handler:
        logger.warning("[Clarify] 重复登记 biz=%s，覆盖旧处理器", biz)
    _HANDLERS[biz] = handler


def registered_handlers() -> dict:
    """当前登记表（副本）；评测/契约测试按它校验覆盖关系。"""
    _ensure_builtin_handlers()
    return dict(_HANDLERS)


def clarify_handler(biz: str) -> Callable | None:
    """按 biz 取处理器；需要时先加载内置注册模块。"""
    _ensure_builtin_handlers()
    return _HANDLERS.get(biz)


def _ensure_builtin_handlers() -> None:
    """内置 handler 分布在 SOP 门和操作模块，按需只加载一次，避免顶层循环依赖。"""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    try:
        from sops import sop_gate  # noqa: F401  导入即注册 SOP 门
        from tools import operations  # noqa: F401  导入即注册 ops 澄清
    except Exception:
        _BUILTINS_LOADED = False
        raise


def dispatch_clarify(session_id: str, biz: str, query: str, q: str) -> ClarifyResult:
    """按 biz 分派。未知 biz 必须显式消费状态，避免下一轮反复卡在同一条 pending。"""
    pending = pending_store.peek(session_id) or {}
    facts = {"biz": biz, "retry": int(pending.get("retry") or 0)}
    handler = clarify_handler(biz)
    if handler is None:
        logger.warning("[Clarify] 未登记的业务 %r，关闭状态 session=%s", biz, session_id)
        trace_store.node("clarify", status="skipped", in_=facts,
                         out={"by": "no_handler"})
        pending_store.close(session_id, biz)
        return ClarifyResult.drop()
    trace_store.node("clarify", in_=facts)
    return handler(session_id, query, q)


def apply_resume(data: ClarifyData, query: str, q: str, enter_declined_turn: bool,
                 prefer_op: str | None) -> tuple:
    """把 resume 载荷应用到本轮状态，返回 (query, q, declined, prefer)。"""
    if data.enter_declined:
        enter_declined_turn = True
    if data.prefer_op:
        prefer_op = data.prefer_op
    if data.query != query:
        query = data.query
        from tools.query_rewrite import normalize_enabled

        if normalize_enabled():
            from tools.query_normalizer import normalize

            q = normalize(query)
            trace_store.step("rewrite", normalize={"changed": q != query, "text": q[:120]})
        else:
            q = query
    return query, q, enter_declined_turn, prefer_op


def match_op_choice(text: str, options: list):
    """解析操作澄清的选择：序号走统一 parse_index，认不出再按选项别名匹配。"""
    status, idx = parse_index(text, len(options))
    if status == "ok":
        return options[idx - 1]
    t = _CONFIRM_CLEAN_RE.sub("", text or "")
    if not t:
        return None
    for op in options:
        labels = _OPERATION_LABELS.get(op, ())
        if any(lb in t for lb in labels):
            return op
    return None


def match_yes_no(query: str, gate: str = "exit") -> bool | None:
    """确认门判据：True=是、False=不是、None=认不出（调用方再问一次，有界两次）。"""
    from config.word_dict_config import (
        CONFIRM_YES_WORDS, CONFIRM_NO_WORDS, CONFIRM_SHORT_YES, CONFIRM_EXIT_ONLY,
    )

    text = _CONFIRM_CLEAN_RE.sub("", query or "")
    if not text:
        return None
    if any(w in text for w in CONFIRM_NO_WORDS):
        return False
    if gate == "exit":
        for w, val in sorted(CONFIRM_EXIT_ONLY.items(), key=lambda kv: len(kv[0]), reverse=True):
            if w in text:
                return val
    if any(w in text for w in CONFIRM_YES_WORDS):
        return True
    if text in CONFIRM_SHORT_YES:
        return True
    return None


def open_clarify(session_id: str, biz: str, state: dict = None,
                 trace_in: dict = None, **data) -> dict:
    """开澄清并落统一节点：提问轮能追到「开了哪个澄清、覆盖了谁的」。"""
    prev = pending_store.peek(session_id)
    rec = pending_store.ask(session_id, biz, state=state, **data)
    opts = data.get("options") or []
    facts = {"biz": biz, "retry": int(rec.get("retry") or 0),
             "options": "/".join(opts) if opts else "-"}
    if trace_in:
        facts.update(trace_in)
    trace_store.node("clarify.open", in_=facts,
                     out={"replaces": (prev or {}).get("biz") or "-"})
    return rec
