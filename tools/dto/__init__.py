"""跨层返回体契约（DTO）。

本包是叶子包：只放返回体，不 import 业务模块。
"""
from tools.dto.clarify import (
    CLARIFY_DONE, CLARIFY_DROP, CLARIFY_KINDS, CLARIFY_RESUME,
    ClarifyData, ClarifyResult,
)
from tools.dto.executor import ExecOutcome, NotMine, Reply
from tools.dto.model_query import ModelQueryResult
from tools.dto.sop import SopStepResult
from tools.dto.stage import Continue, Finish, StageOutcome
from tools.dto.transport import TurnEvent

__all__ = [
    "Continue", "Finish", "StageOutcome",
    "Reply", "NotMine", "ExecOutcome",
    "ClarifyData", "ClarifyResult",
    "CLARIFY_DONE", "CLARIFY_RESUME", "CLARIFY_DROP", "CLARIFY_KINDS",
    "ModelQueryResult",
    "SopStepResult",
    "TurnEvent",
]
