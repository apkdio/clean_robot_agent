"""跨层返回体契约（DTO）。

本包是叶子包：只放返回体，不 import 业务模块。
"""
from utils.dto.clarify import (
    CLARIFY_DONE, CLARIFY_DROP, CLARIFY_KINDS, CLARIFY_RESUME,
    ClarifyData, ClarifyResult,
)
from utils.dto.executor import ExecOutcome, NotMine, Reply
from utils.dto.handoff import Handoff
from utils.dto.model_query import (FindModelsResult, ModelQueryConstraints,
                                   ParamFilter, PriceRange, PublishRange)
from utils.dto.request import ChatRequest
from utils.dto.sop import ExtractResult, SopStepResult
from utils.dto.stage import Continue, Finish, StageOutcome
from utils.dto.transport import TurnEvent

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
