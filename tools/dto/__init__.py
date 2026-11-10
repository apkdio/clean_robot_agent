"""跨层返回体契约（DTO）。

本包是叶子包：只放返回体，不 import 业务模块。
"""
from tools.dto.clarify import (
    CLARIFY_DONE, CLARIFY_DROP, CLARIFY_KINDS, CLARIFY_RESUME,
    ClarifyData, ClarifyResult,
)
from tools.dto.executor import ExecOutcome, NotMine, Reply
from tools.dto.handoff import Handoff
from tools.dto.model_query import (FindModelsResult, ModelQueryConstraints,
                                   ParamFilter, PriceRange, PublishRange)
from tools.dto.request import ChatRequest
from tools.dto.sop import ExtractResult, SopStepResult
from tools.dto.stage import Continue, Finish, StageOutcome
from tools.dto.transport import TurnEvent

__all__ = [
    "Continue", "Finish", "StageOutcome",
    "Reply", "NotMine", "ExecOutcome", "Handoff",
    "ClarifyData", "ClarifyResult",
    "CLARIFY_DONE", "CLARIFY_RESUME", "CLARIFY_DROP", "CLARIFY_KINDS",
    "FindModelsResult", "ModelQueryConstraints",
    "PriceRange", "PublishRange", "ParamFilter",
    "ChatRequest",
    "ExtractResult", "SopStepResult",
    "TurnEvent",
]
