"""主链单次抽取：一次 FC 只取一个窄参数。

schema 在 `function_tools/schemas.py`；纯规则在 `tools/*_rules.py` / `tools/service_point.py`。
这里只负责发一次 function call、取 args 和做最小转换，不承担流程与业务门控。
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from function_tools.schemas import (
    EXTRACT_BUDGET_SCHEMA,
    EXTRACT_DATE_SCHEMA,
    EXTRACT_LOCATION_SCHEMA,
    EXTRACT_MODELS_SCHEMA,
    EXTRACT_SYMPTOM_SCHEMA,
)
from tools import llm_tool
from tools.date_rules import calc_date_range
from tools.log_tool import get_logger
from tools.metadata_extractor import budget_args_to_range
from tools.service_point import geocode_city
from tools.symptom_rules import symptom_id_to_query

logger = get_logger(name="llm_extract")


def extract_date_range(query: str) -> tuple | None:
    """LLM 抽取日期表达并换算；失败返回 None。"""
    try:
        args = llm_tool.call_tool_args(
            [HumanMessage(content=query)],
            EXTRACT_DATE_SCHEMA,
            model=llm_tool.get_chat_model_name(),
        )
        if not args:
            return None
        result = calc_date_range(args.get("expression", ""))
        if "error" not in result:
            return result["start_date"], result["end_date"]
    except Exception as e:  # noqa: BLE001
        logger.warning("[ExtractDate] tool calling failed: %s", e)
    return None


def extract_budget_range(query: str) -> tuple | None:
    """LLM 抽取预算上限；失败返回 None。"""
    try:
        args = llm_tool.call_tool_args(
            [HumanMessage(content=query)],
            EXTRACT_BUDGET_SCHEMA,
            model=llm_tool.get_small_model_name(),
        )
        rng = budget_args_to_range(args or {})
        if rng is not None:
            logger.info("[ExtractBudget] resolved via LLM tool: %s", args)
        return rng
    except Exception as e:  # noqa: BLE001
        logger.warning("[ExtractBudget] tool calling failed: %s", e)
        return None


def extract_model_names(query: str, history: str = "") -> str:
    """从当前问题和上下文提取型号名；失败返回空字符串。"""
    content = query if not history else "对话历史：\n%s\n\n用户当前问题：%s" % (history, query)
    try:
        args = llm_tool.call_tool_args(
            [HumanMessage(content=content)],
            EXTRACT_MODELS_SCHEMA,
            model=llm_tool.get_chat_model_name(),
        )
        return str((args or {}).get("model_names") or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("[ExtractModel] tool calling failed: %s", e)
        return ""


def classify_symptom(text: str) -> str | None:
    """返回标准检索 query；无法归类返回 None。"""
    try:
        args = llm_tool.call_tool_args(
            [HumanMessage(content=text)],
            EXTRACT_SYMPTOM_SCHEMA,
            model=llm_tool.get_small_model_name(),
        )
        return symptom_id_to_query(args or {})
    except Exception as e:  # noqa: BLE001
        logger.warning("[ExtractSymptom] tool calling failed: %s", e)
        return None


def extract_city_candidates(text: str) -> list:
    """返回城市候选；失败返回空列表。"""
    try:
        args = llm_tool.call_tool_args(
            [HumanMessage(content=text)],
            EXTRACT_LOCATION_SCHEMA,
            model=llm_tool.get_chat_model_name(),
        )
        return geocode_city(str((args or {}).get("location") or "").strip())
    except Exception as e:  # noqa: BLE001
        logger.warning("[ExtractLocation] tool calling failed: %s", e)
        return []
