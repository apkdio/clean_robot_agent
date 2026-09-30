"""仅供测试/评测使用的状态注入助手（生产链路不调用）。

评测 harness 不该直接摸内部结构——会话记录里的 `danger` 标记、安全承接的"连续承接次数"、
SOP 会话状态与统一待确认状态——所以把最小的 setup 接口集中在这里。

对应 project_detail.md §4.15 的状态型用例（`kind: stateful`；决策见 ADR-25）：

    setup = {"danger_word": "糊味", "carry_count": 1, "intent": "unknown"}
    → set_danger_window(sid, "糊味"); set_carry_count(sid, 1)

用到的私有接口（`agent._SAFETY_CARRY_PREFIX`、`sops.base._save_session`）都在本文件里封住：
将来内部结构变了，只改这一处。
"""

from __future__ import annotations

from typing import Dict, Optional

from tools import pending_store


def set_danger_window(session_id: str, word: str = "糊味", text: str = "") -> None:
    """在会话里造一条带 `danger` 标记的用户消息（安全承接的前提之一）。"""
    from tools.context_store import append_message

    append_message(session_id, "user", text or ("我的机器人底下好像有%s出现，还有味道" % word), danger=True)


def set_carry_count(session_id: str, n: int) -> None:
    """把「最近连续安全承接次数」设成 n。

    判定机制：`agent._recent_carry_count()` 从尾部数**连续**以 `_SAFETY_CARRY_PREFIX`
    开头的 assistant 回复，所以这里就追加以该前缀开头的回复。
    """
    from tools.agent import _SAFETY_CARRY_PREFIX
    from tools.context_store import append_message

    for _ in range(max(0, int(n))):
        append_message(session_id, "assistant",
                       _SAFETY_CARRY_PREFIX + "情况属于安全隐患，请保持断电停机并联系官方售后。")


def set_active_sop(session_id: str, sop_id: str, step: int = 0,
                   slots: Optional[Dict] = None, retry_count: int = 0) -> None:
    """直接写入一个活跃 SOP 会话（不跑首轮引导，便于精确构造场景）。"""
    from sops.base import _save_session

    _save_session(session_id, {"sop_id": sop_id, "step": int(step),
                               "slots": dict(slots or {}), "retry_count": int(retry_count)})


def set_pending(session_id: str, kind: str, **data) -> None:
    """写入统一待确认状态：`kind` ∈ {confirm_exit, confirm_enter}。"""
    pending_store.set(session_id, kind, **data)


def reset(session_id: str) -> None:
    """清掉该会话的全部状态（会话记录与 meta / SOP / 待确认），用于用例间隔离。"""
    from sops.base import _delete_session

    from tools.context_store import delete_session

    delete_session(session_id)
    _delete_session(session_id)
    pending_store.clear(session_id)
