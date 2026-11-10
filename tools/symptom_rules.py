"""故障编号纯能力：编号 → 标准检索 query。"""
from __future__ import annotations

from typing import Dict, Optional

from config.word_dict_config import SYMPTOM_QUERY_MAP


def symptom_id_to_query(args: Dict) -> Optional[str]:
    """把 LLM 返回的 {symptom_id} 映射成标准检索 query。

    编号 0 / 空 / 非数字 / 越界 → None（视为未识别，SOP 走 retry 重新问）。
    """
    if not args:
        return None

    sid = args.get("symptom_id")
    if sid is None:
        return None

    try:
        sid = int(sid)
    except (TypeError, ValueError):
        return None

    return SYMPTOM_QUERY_MAP.get(sid)
