"""型号指代与型号名命中判断：供操作判别器与主链共用。"""
from __future__ import annotations

from tools.log_tool import get_logger

logger = get_logger(name="model_ref")


def has_model_ref(query: str) -> bool:
    """判断 query 是否指向某个具体型号（指代词或直接报型号名）。"""
    from config.word_dict_config import MODEL_REF_WORDS

    if any(w in query for w in MODEL_REF_WORDS):
        return True
    try:
        from function_tools.model_tool import model_name_in_query

        return model_name_in_query(query)
    except Exception as e:  # noqa: BLE001
        logger.warning("[ModelRef] Model name check failed: %s", e)
        return False
